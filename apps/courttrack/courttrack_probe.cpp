/* Courtside: the court tracker inside DeepStream.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Buffer probe behind the secondary GIEs (after jersey_log, which hands over the jersey reads and removes the helper
 * objects). For every batch:
 *   detections   objects with an MV3DT world foot location; court filter on the box bottom (court + margin, boxes
 *                cut by the bottom image edge kept), the camera's fixed bias (camInfo courtTrackerBias_m) removed
 *   colour       every `appearance_every`-th frame, for good crops (not cut by the image border, >= 80 px tall) of the
 *                colour cameras: the torso (20-50 % of the box height, central half of its width) is cut out of the
 *                frame with NvBufSurfTransform into a small RGBA tile, and its HSV hue x saturation histogram
 *                (18 x 8) computed on the CPU; body height from the box for the height cameras
 *   tracking     once all cameras of a frame have arrived: the online team model labels the colour crops, the court
 *                tracker steps, and the team model learns from the tracker's assignment (online_teams.cpp,
 *                courttrack.cpp); every frame from the first one with detections is stepped
 *   output       one JSON message per frame on its own Kafka topic (players: court track id, identity, team,
 *                position, velocity, measured or predicted, cameras, contributing detections); optional CSV of the
 *                identity of every detection, and the tracker's inputs in the replay format (ct_replay)
 * Everything runs on the probe's streaming thread; the reported identities are final when sent.
 */
#include "courttrack_probe.h"

#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "courttrack.h"
#include "gstnvdsmeta.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "nvds_msgapi.h"
#include "online_teams.h"

using namespace courttrack;

namespace {

constexpr int TILE_W = 32, TILE_H = 48;   // torso tile (RGBA) the histogram is computed on
constexpr int MAX_CROPS = 64;             // crops per NvBufSurfTransform call

struct Camera {
  int source = -1;
  std::string name;
  int index = 0;          // tracker camera index (cameras in name order)
  bool colour = false, height = false;
  double bias_x = 0, bias_y = 0;
  double P[12];
  double Hinv[9];         // inverse of P[:, [0, 1, 3]]: pixel -> floor (z = 0)
};

struct Config {
  double xmin = 0, xmax = 0, yhalf = 0, margin = 0.3;
  int frame_w = 1920, frame_h = 1080;
  int appearance_every = 3;
  double min_box_h = 80;
  std::vector<Camera> cams;               // by source id
  std::string kafka_lib, kafka_conn, kafka_topic, kafka_cfg;
  std::string ident_log, input_log;
};

struct Crop {
  int batch_id;
  NvBufSurfTransformRect rect;
  int frame, cam, det;                    // where the histogram goes
};

struct PendingFrame {
  std::vector<CameraFrame> cams;          // arrival order
  std::vector<int> sources;
  guint64 ntp = 0;
};

struct Stats {
  std::vector<double> step_ms, hist_ms, log_ms;   // compute per frame; histograms per batch; logs + Kafka per frame
  long crops = 0;
};

Config cfg;
std::unique_ptr<Tracker> tracker;
std::unique_ptr<OnlineTeams> teams;
std::map<int, int> index_to_source;
std::mutex reads_mutex;
std::map<std::tuple<guint, guint, guint64>, std::pair<std::string, float>> reads;
std::map<int, PendingFrame> pending;
int last_stepped = -1;
FILE *ident_log = nullptr;
std::ofstream input_log;
NvBufSurface *tiles = nullptr;
Stats stats;

/* Kafka through DeepStream's message API (the same adapter library the per-camera messages use) */
struct Kafka {
  void *lib = nullptr;
  NvDsMsgApiHandle h = nullptr;
  decltype(&nvds_msgapi_connect) connect = nullptr;
  decltype(&nvds_msgapi_send_async) send = nullptr;
  decltype(&nvds_msgapi_do_work) work = nullptr;
  decltype(&nvds_msgapi_disconnect) disconnect = nullptr;
  long sent = 0, failed = 0;
} kafka;

bool parse_config(const char *path) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line;
  std::vector<Camera> cams;
  while (std::getline(in, line)) {
    std::istringstream s(line);
    std::string key;
    if (!(s >> key) || key[0] == '#') continue;
    if (key == "court") s >> cfg.xmin >> cfg.xmax >> cfg.yhalf >> cfg.margin >> cfg.frame_w >> cfg.frame_h;
    else if (key == "appearance") s >> cfg.appearance_every >> cfg.min_box_h;
    else if (key == "camera") {
      Camera c;
      int colour, height;
      s >> c.source >> c.name >> c.index >> colour >> height >> c.bias_x >> c.bias_y;
      for (double &v : c.P) s >> v;
      c.colour = colour;
      c.height = height;
      if (!s) return false;
      cams.push_back(c);
    } else if (key == "kafka") s >> cfg.kafka_lib >> cfg.kafka_conn >> cfg.kafka_topic >> cfg.kafka_cfg;
    else if (key == "ident_log") s >> cfg.ident_log;
    else if (key == "input_log") s >> cfg.input_log;
  }
  int n = 0;
  for (auto &c : cams) n = std::max(n, c.source + 1);
  cfg.cams.assign(n, Camera{});
  for (auto &c : cams) {
    const double *P = c.P;   // H = P[:, [0, 1, 3]] (row-major 3x3), inverted by cofactors
    double a = P[0], b = P[1], d = P[3], e = P[4], f = P[5], g = P[7], h = P[8], i = P[9], k = P[11];
    double A = f * k - g * i, B = -(e * k - g * h), C = e * i - f * h;
    double det = a * A + b * B + d * C;
    double inv[9] = {A, -(b * k - d * i), b * g - d * f, B, a * k - d * h, -(a * g - d * e), C, -(a * i - b * h),
                     a * f - b * e};
    for (int j = 0; j < 9; ++j) c.Hinv[j] = inv[j] / det;
    cfg.cams[c.source] = c;
  }
  return !cams.empty() && cfg.xmax > cfg.xmin;
}

/* tools/stitch.floor_point: pixel -> floor point, and whether it lies in front of the camera */
bool floor_point(const Camera &c, double u, double v, double *x, double *y) {
  const double *H = c.Hinv;
  double X0 = H[0] * u + H[1] * v + H[2], X1 = H[3] * u + H[4] * v + H[5], X2 = H[6] * u + H[7] * v + H[8];
  *x = X0 / X2;
  *y = X1 / X2;
  double depth = c.P[8] * *x + c.P[9] * *y + c.P[11];
  return depth > 0;
}

/* tools/appearance.person_height */
double person_height(const Camera &c, double l, double t, double r, double b) {
  if (t <= 5 || b >= cfg.frame_h - 5 || l <= 2 || r >= cfg.frame_w - 2) return NAN;
  double x, y;
  if (!floor_point(c, (l + r) / 2, b, &x, &y)) return NAN;
  const double *P = c.P;
  int best = 0;
  double bd = INFINITY;
  for (int i = 0; i < 211; ++i) {
    double hh = 0.5 + i * (2.1 / 210);
    double v = (P[4] * x + P[5] * y + P[6] * hh + P[7]) / (P[8] * x + P[9] * y + P[10] * hh + P[11]);
    if (std::fabs(v - t) < bd) {
      bd = std::fabs(v - t);
      best = i;
    }
  }
  return best > 0 && best < 210 ? 0.5 + best * (2.1 / 210) : NAN;
}

/* OpenCV's 8-bit RGB -> HSV (hue 0-179, saturation 0-255), then calcHist with 18 x 8 uniform bins */
void hsv_hist(const uint8_t *rgba, int pitch, Hist *out) {
  static int sdiv[256], hdiv[256];
  static bool init = false;
  const int shift = 12;
  if (!init) {
    sdiv[0] = hdiv[0] = 0;
    for (int i = 1; i < 256; ++i) {
      sdiv[i] = (int)std::lround((255 << shift) / (1. * i));
      hdiv[i] = (int)std::lround((180 << shift) / (6. * i));
    }
    init = true;
  }
  out->fill(0);
  double n = 0;
  for (int yy = 0; yy < TILE_H; ++yy) {
    const uint8_t *p = rgba + yy * pitch;
    for (int xx = 0; xx < TILE_W; ++xx, p += 4) {
      int r = p[0], g = p[1], b = p[2];
      int v = std::max(r, std::max(g, b)), vmin = std::min(r, std::min(g, b)), diff = v - vmin;
      int vr = v == r ? -1 : 0, vg = v == g ? -1 : 0;
      int s = (diff * sdiv[v] + (1 << (shift - 1))) >> shift;
      int h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
      h = (h * hdiv[diff] + (1 << (shift - 1))) >> shift;
      h += h < 0 ? 180 : 0;
      if (h >= 180) continue;    // calcHist ignores values at the upper bound
      (*out)[(h / 10) * 8 + s / 32] += 1;
      n += 1;
    }
  }
  for (auto &x : *out) x = (float)(x / (n + 1e-9));
}

void compute_histograms(NvBufSurface *surf, std::vector<Crop> &crops, std::map<int, PendingFrame> &frames) {
  if (crops.empty()) return;
  auto t0 = std::chrono::steady_clock::now();
  if (!tiles) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, surf->gpuId);
    NvBufSurfaceCreateParams cp{};
    cp.gpuId = surf->gpuId;
    cp.width = TILE_W;
    cp.height = TILE_H;
    cp.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    cp.layout = NVBUF_LAYOUT_PITCH;
    cp.memType = prop.integrated ? NVBUF_MEM_DEFAULT : NVBUF_MEM_CUDA_UNIFIED;   // CPU-mappable on both
    if (NvBufSurfaceCreate(&tiles, MAX_CROPS, &cp) != 0) {
      g_printerr("courttrack: cannot allocate torso tiles\n");
      tiles = nullptr;
      return;
    }
  }
  NvBufSurfTransformConfigParams tc{};
  tc.compute_mode = NvBufSurfTransformCompute_Default;
  tc.gpu_id = surf->gpuId;
  tc.cuda_stream = nullptr;
  NvBufSurfTransformSetSessionParams(&tc);
  for (size_t at = 0; at < crops.size(); at += MAX_CROPS) {
    size_t n = std::min(crops.size() - at, (size_t)MAX_CROPS);
    std::vector<bool> ok(n, false);
    for (size_t i = 0; i < n; ++i) {   // one crop per call, through single-entry views (as NVIDIA's gst-dsexample)
      NvBufSurface src = *surf, dst = *tiles;
      src.surfaceList = &surf->surfaceList[crops[at + i].batch_id];
      dst.surfaceList = &tiles->surfaceList[i];
      src.batchSize = src.numFilled = dst.batchSize = dst.numFilled = 1;
      NvBufSurfTransformRect sr = crops[at + i].rect, dr{0, 0, TILE_W, TILE_H};
      NvBufSurfTransformParams tp{};
      tp.transform_flag = NVBUFSURF_TRANSFORM_CROP_SRC | NVBUFSURF_TRANSFORM_CROP_DST | NVBUFSURF_TRANSFORM_FILTER;
      tp.transform_filter = NvBufSurfTransformInter_Bilinear;
      tp.src_rect = &sr;
      tp.dst_rect = &dr;
      ok[i] = NvBufSurfTransform(&src, &dst, &tp) == NvBufSurfTransformError_Success;
      if (!ok[i]) g_printerr("courttrack: NvBufSurfTransform failed\n");
    }
    tiles->numFilled = (uint32_t)n;
    if (NvBufSurfaceMap(tiles, -1, -1, NVBUF_MAP_READ) != 0) continue;
    NvBufSurfaceSyncForCpu(tiles, -1, -1);
    for (size_t i = 0; i < n; ++i) {
      if (!ok[i]) continue;
      const Crop &c = crops[at + i];
      Detection &d = frames[c.frame].cams[c.cam].dets[c.det];
      hsv_hist((const uint8_t *)tiles->surfaceList[i].mappedAddr.addr[0], tiles->surfaceList[i].planeParams.pitch[0],
               &d.feat);
      d.has_feat = true;   // until labelling, has_feat/feat hold the torso histogram (as in ct_replay's input)
    }
    NvBufSurfaceUnMap(tiles, -1, -1);
    stats.crops += n;
  }
  stats.hist_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

void write_input_header() {
  input_log.write("CTIN2", 5);
  std::vector<const Camera *> order;
  for (auto &c : cfg.cams)
    if (c.source >= 0) order.push_back(&c);
  std::sort(order.begin(), order.end(), [](const Camera *a, const Camera *b) { return a->index < b->index; });
  int32_t n = (int32_t)order.size();
  input_log.write((const char *)&n, 4);
  for (auto *c : order) {
    double P[12];
    std::copy(c->P, c->P + 12, P);
    // camera centre on the floor: -inv(P[:, :3]) P[:, 3], first two coordinates (as tools/world_tracker.track_run)
    double M[9] = {P[0], P[1], P[2], P[4], P[5], P[6], P[8], P[9], P[10]}, t[3] = {P[3], P[7], P[11]};
    double A = M[4] * M[8] - M[5] * M[7], B = -(M[3] * M[8] - M[5] * M[6]), C = M[3] * M[7] - M[4] * M[6];
    double det = M[0] * A + M[1] * B + M[2] * C;
    double inv[9] = {A, -(M[1] * M[8] - M[2] * M[7]), M[1] * M[5] - M[2] * M[4], B, M[0] * M[8] - M[2] * M[6],
                     -(M[0] * M[5] - M[2] * M[3]), C, -(M[0] * M[7] - M[1] * M[6]), M[0] * M[4] - M[1] * M[3]};
    double x = -(inv[0] * t[0] + inv[1] * t[1] + inv[2] * t[2]) / det;
    double y = -(inv[3] * t[0] + inv[4] * t[1] + inv[5] * t[2]) / det;
    input_log.write((const char *)&x, 8);
    input_log.write((const char *)&y, 8);
    int32_t len = (int32_t)c->name.size();
    input_log.write((const char *)&len, 4);
    input_log.write(c->name.data(), len);
  }
}

void write_input_frame(int frame, const std::vector<CameraFrame> &cams) {
  auto w32 = [](int32_t v) { input_log.write((const char *)&v, 4); };
  auto wd = [](double v) { input_log.write((const char *)&v, 8); };
  auto ws = [&](const std::string &s) {
    w32((int32_t)s.size());
    input_log.write(s.data(), s.size());
  };
  w32(frame);
  w32((int32_t)cams.size());
  for (auto &cf : cams) {
    w32(cf.cam);
    w32((int32_t)cf.dets.size());
    for (auto &d : cf.dets) {
      wd(d.x);
      wd(d.y);
      wd(0.0);
      wd(d.height);
      int64_t gid = d.gid;
      input_log.write((const char *)&gid, 8);
      ws("");
      w32(d.has_read);
      wd(d.read_score);
      ws(d.number);
      w32(d.has_feat);
      if (d.has_feat) input_log.write((const char *)d.feat.data(), sizeof(float) * FEAT_DIM);
    }
  }
}

std::string json_str(const std::string &s) { return s.empty() ? "null" : "\"" + s + "\""; }

void send_frame(int frame, guint64 ntp) {
  std::ostringstream o;
  o.setf(std::ios::fixed);
  o.precision(3);
  o << "{\"version\":\"1.0\",\"frame\":" << frame << ",\"ntpTimestamp\":" << ntp << ",\"players\":[";
  bool first = true;
  for (auto &t : tracker->live()) {
    // team: from the identity when there is one (it may come from the roster before the colour votes are firm)
    std::string team = !t.identity.empty() ? t.identity.substr(0, t.identity.find(':'))
                                           : (t.team == "A" || t.team == "B" ? t.team : "");
    o << (first ? "" : ",") << "{\"track\":" << t.tid << ",\"identity\":" << json_str(t.identity)
      << ",\"team\":" << json_str(team) << ",\"x\":" << t.x << ",\"y\":" << t.y << ",\"vx\":" << t.vx
      << ",\"vy\":" << t.vy << ",\"measured\":" << (t.measured ? "true" : "false") << ",\"cameras\":" << t.n_cams
      << ",\"detections\":[";
    for (size_t i = 0; i < t.dets.size(); ++i)
      o << (i ? "," : "") << "{\"sensor\":\"" << cfg.cams[index_to_source[t.dets[i].cam]].name
        << "\",\"objectId\":\"" << t.dets[i].gid << "\"}";
    o << "]}";
    first = false;
  }
  o << "]}";
  std::string msg = o.str();
  if (kafka.h) {
    auto cb = [](void *, NvDsMsgApiErrorType st) {
      if (st == NVDS_MSGAPI_OK) ++kafka.sent;
      else ++kafka.failed;
    };
    if (kafka.send(kafka.h, (char *)cfg.kafka_topic.c_str(), (const uint8_t *)msg.data(), msg.size(), cb, nullptr) !=
        NVDS_MSGAPI_OK)
      ++kafka.failed;
    kafka.work(kafka.h);
  }
}

void step_frame(int frame, PendingFrame &pf) {
  auto l0 = std::chrono::steady_clock::now();
  if (!cfg.input_log.empty()) write_input_frame(frame, pf.cams);
  double log_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - l0).count();
  auto t0 = std::chrono::steady_clock::now();
  struct Colour {
    bool has = false;
    Hist s{};
    bool read = false;
  };
  std::map<DetKey, Colour> colour;
  teams->update(frame);
  for (auto &cf : pf.cams)
    for (auto &d : cf.dets) {
      Colour &col = colour[{frame, cf.cam, d.gid}];
      col.read = d.has_read;
      d.team.clear();
      d.team_p = 0;
      if (d.has_feat) {
        col.has = true;
        col.s = OnlineTeams::sample(d.feat);
        auto lab = teams->label(cf.cam, col.s);
        d.team = lab.first;
        d.team_p = lab.second;
        d.has_feat = teams->feature(cf.cam, col.s, &d.feat);
        teams->observe_crop(cf.cam, col.s);
      }
    }
  auto ids = tracker->step(frame, pf.cams);
  for (auto &di : ids) {
    const Colour &col = colour[di.key];
    teams->observe_track(di.tid, frame, di.key.cam, col.has ? &col.s : nullptr, col.read);
  }
  stats.step_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  auto l1 = std::chrono::steady_clock::now();
  if (ident_log)
    for (auto &di : ids)
      std::fprintf(ident_log, "%d,%s,%lld,%d,%s\n", frame, cfg.cams[index_to_source[di.key.cam]].name.c_str(),
                   (long long)di.key.gid, di.tid, di.identity.c_str());
  send_frame(frame, pf.ntp);
  stats.log_ms.push_back(log_ms +
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - l1).count());
  last_stepped = frame;
}

/* Step every complete frame in order (and frames that will not complete because a later one has arrived). */
void step_ready(bool flush) {
  int nsrc = 0;
  for (auto &c : cfg.cams) nsrc += c.source >= 0;
  while (!pending.empty()) {
    auto it = pending.begin();
    bool complete = (int)it->second.sources.size() >= nsrc;
    bool overtaken = pending.rbegin()->first > it->first + 2;
    if (!complete && !overtaken && !flush) break;
    int frame = it->first;
    bool any = false;
    for (auto &cf : it->second.cams) any |= !cf.dets.empty();
    if (last_stepped < 0 && !any) {   // start with the first frame that has detections (as track_run)
      pending.erase(it);
      continue;
    }
    if (last_stepped >= 0)
      for (int f = last_stepped + 1; f < frame; ++f) {   // frames no camera delivered: still stepped
        PendingFrame empty;
        step_frame(f, empty);
      }
    step_frame(frame, it->second);
    pending.erase(it);
  }
  std::lock_guard<std::mutex> lk(reads_mutex);   // reads of stepped frames are no longer needed
  for (auto it = reads.begin(); it != reads.end() && (int)std::get<0>(it->first) <= last_stepped;)
    it = reads.erase(it);
}

GstPadProbeReturn probe(GstPad *, GstPadProbeInfo *info, gpointer) {
  GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
  NvDsBatchMeta *batch = gst_buffer_get_nvds_batch_meta(buf);
  if (!batch) return GST_PAD_PROBE_OK;
  GstMapInfo map;
  NvBufSurface *surf = nullptr;
  if (gst_buffer_map(buf, &map, GST_MAP_READ)) surf = (NvBufSurface *)map.data;
  std::vector<Crop> crops;
  for (NvDsMetaList *lf = batch->frame_meta_list; lf; lf = lf->next) {
    NvDsFrameMeta *fm = (NvDsFrameMeta *)lf->data;
    if (fm->source_id >= cfg.cams.size() || cfg.cams[fm->source_id].source < 0) continue;
    const Camera &cam = cfg.cams[fm->source_id];
    int frame = (int)fm->frame_num;
    if (frame <= last_stepped) continue;   // late: its frame was already stepped
    PendingFrame &pf = pending[frame];
    if (!pf.ntp) pf.ntp = fm->ntp_timestamp;
    pf.sources.push_back(cam.source);
    pf.cams.push_back(CameraFrame{cam.index, {}});
    int ci = (int)pf.cams.size() - 1;
    bool appearance = cam.colour && frame % cfg.appearance_every == 0;
    std::vector<guint64> seen;   // MV3DT occasionally gives two people one id in a frame: keep the first
    for (NvDsMetaList *lo = fm->obj_meta_list; lo; lo = lo->next) {
      NvDsObjectMeta *obj = (NvDsObjectMeta *)lo->data;
      const float *foot = nullptr;
      for (NvDsMetaList *lu = obj->obj_user_meta_list; lu; lu = lu->next) {
        NvDsUserMeta *um = (NvDsUserMeta *)lu->data;
        if (um->base_meta.meta_type == NVDS_OBJ_WORLD_FOOT_LOCATION) foot = (const float *)um->user_meta_data;
      }
      if (!foot) continue;
      const NvOSD_RectParams &rp = obj->rect_params;
      double l = rp.left, t = rp.top, r = rp.left + rp.width, b = rp.top + rp.height;
      double fx, fy;
      bool front = floor_point(cam, (l + r) / 2, b, &fx, &fy);
      bool on = front && fx >= cfg.xmin - cfg.margin && fx <= cfg.xmax + cfg.margin &&
                std::fabs(fy) <= cfg.yhalf + cfg.margin;
      if (!on && b < cfg.frame_h - 5) continue;   // feet below the image edge: position unknown, kept
      if (std::find(seen.begin(), seen.end(), obj->object_id) != seen.end()) continue;
      seen.push_back(obj->object_id);
      Detection d;
      d.x = foot[0] - cam.bias_x;
      d.y = foot[1] - cam.bias_y;
      d.gid = (int64_t)obj->object_id;
      {
        std::lock_guard<std::mutex> lk(reads_mutex);
        auto it = reads.find({fm->frame_num, fm->source_id, obj->object_id});
        if (it != reads.end() && !it->second.first.empty()) {
          d.has_read = true;
          d.number = it->second.first;
          d.read_score = it->second.second;
        }
      }
      bool cut = t <= 5 || b >= cfg.frame_h - 5 || l <= 2 || r >= cfg.frame_w - 2;
      if (appearance && !cut && b - t >= cfg.min_box_h && surf) {
        int li = (int)std::max(0.0, l), ti = (int)std::max(0.0, t);
        int ri = (int)std::min(cfg.frame_w - 1.0, r), bi = (int)std::min(cfg.frame_h - 1.0, b);
        int h = bi - ti, w = ri - li;
        if (h >= 60 && w >= 15) {   // tools/appearance.torso_hist
          int top = ti + (int)(0.2 * h), bottom = ti + (int)(0.5 * h);
          int left = li + (int)(0.25 * w), right = ri - (int)(0.25 * w);
          if (bottom > top && right > left) {
            crops.push_back({(int)fm->batch_id,
                             {(uint32_t)top, (uint32_t)left, (uint32_t)(right - left), (uint32_t)(bottom - top)},
                             frame, ci, (int)pf.cams[ci].dets.size()});
            if (cam.height) d.height = person_height(cam, l, t, r, b);
          }
        }
      }
      pf.cams[ci].dets.push_back(d);
    }
  }
  if (surf) {
    compute_histograms(surf, crops, pending);
    gst_buffer_unmap(buf, &map);
  }
  step_ready(false);
  return GST_PAD_PROBE_OK;
}

double pct(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, (size_t)(q / 100 * v.size()))];
}

}  // namespace

extern "C" gboolean courttrack_attach(GstPad *pad) {
  const char *path = g_getenv("COURTTRACK_CONFIG");
  if (!path || !*path || !pad) return FALSE;
  if (!parse_config(path)) {
    g_printerr("courttrack: cannot read config %s\n", path);
    return FALSE;
  }
  std::vector<std::pair<double, double>> cam_xy(cfg.cams.size());
  for (auto &c : cfg.cams)
    if (c.source >= 0) {
      index_to_source[c.index] = c.source;
      // camera centre on the floor (aniso measurement noise): -inv(P[:, :3]) P[:, 3]
      const double *P = c.P;
      double M[9] = {P[0], P[1], P[2], P[4], P[5], P[6], P[8], P[9], P[10]}, t[3] = {P[3], P[7], P[11]};
      double A = M[4] * M[8] - M[5] * M[7], B = -(M[3] * M[8] - M[5] * M[6]), C = M[3] * M[7] - M[4] * M[6];
      double det = M[0] * A + M[1] * B + M[2] * C;
      double inv[6] = {A, -(M[1] * M[8] - M[2] * M[7]), M[1] * M[5] - M[2] * M[4], B, M[0] * M[8] - M[2] * M[6],
                       -(M[0] * M[5] - M[2] * M[3])};
      cam_xy[c.index] = {-(inv[0] * t[0] + inv[1] * t[1] + inv[2] * t[2]) / det,
                         -(inv[3] * t[0] + inv[4] * t[1] + inv[5] * t[2]) / det};
    }
  tracker.reset(new Tracker(Params{}, cam_xy));
  teams.reset(new OnlineTeams(OTParams{}));
  if (!cfg.ident_log.empty()) {
    ident_log = std::fopen(cfg.ident_log.c_str(), "w");
    if (ident_log) {
      setvbuf(ident_log, nullptr, _IOFBF, 1 << 20);
      std::fprintf(ident_log, "frame,sensor,object_id,track,identity\n");
    }
  }
  if (!cfg.input_log.empty()) {
    input_log.open(cfg.input_log, std::ios::binary);
    write_input_header();
  }
  if (!cfg.kafka_lib.empty()) {
    kafka.lib = dlopen(cfg.kafka_lib.c_str(), RTLD_LAZY);
    if (kafka.lib) {
      kafka.connect = (decltype(kafka.connect))dlsym(kafka.lib, "nvds_msgapi_connect");
      kafka.send = (decltype(kafka.send))dlsym(kafka.lib, "nvds_msgapi_send_async");
      kafka.work = (decltype(kafka.work))dlsym(kafka.lib, "nvds_msgapi_do_work");
      kafka.disconnect = (decltype(kafka.disconnect))dlsym(kafka.lib, "nvds_msgapi_disconnect");
    }
    if (kafka.connect && kafka.send && kafka.work && kafka.disconnect)
      kafka.h = kafka.connect((char *)cfg.kafka_conn.c_str(), [](NvDsMsgApiHandle, NvDsMsgApiEventType) {},
                              cfg.kafka_cfg == "-" ? nullptr : (char *)cfg.kafka_cfg.c_str());
    if (!kafka.h) g_printerr("courttrack: no Kafka connection (%s, %s)\n", cfg.kafka_lib.c_str(), cfg.kafka_conn.c_str());
  }
  gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, probe, nullptr, nullptr);
  g_print("courttrack: %zu cameras, court tracker + online team model, Kafka topic %s%s\n", index_to_source.size(),
          cfg.kafka_topic.empty() ? "(none)" : cfg.kafka_topic.c_str(), kafka.h ? "" : " (not connected)");
  return TRUE;
}

extern "C" void courttrack_add_read(guint frame, guint source, guint64 object_id, const char *number, float conf) {
  if (!tracker) return;
  std::string n = number ? number : "";
  float sc = n.empty() ? 0.f : conf;   // as tools/jersey.from_ds: the most confident read per object and frame
  std::lock_guard<std::mutex> lk(reads_mutex);
  auto key = std::make_tuple(frame, source, object_id);
  auto it = reads.find(key);
  if (it == reads.end() || sc > it->second.second) reads[key] = {n, sc};
}

extern "C" void courttrack_close(void) {
  if (!tracker) return;
  step_ready(true);
  if (ident_log) std::fclose(ident_log);
  ident_log = nullptr;
  if (input_log.is_open()) {
    int32_t end = -1;
    input_log.write((const char *)&end, 4);
    input_log.close();
  }
  if (kafka.h) {
    for (int i = 0; i < 50; ++i) kafka.work(kafka.h);
    kafka.disconnect(kafka.h);
  }
  auto mean = [](const std::vector<double> &v) {
    return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  };
  g_print("courttrack: logs + Kafka %.3f ms mean, %.3f ms p99, %.3f ms max per frame\n", mean(stats.log_ms),
          pct(stats.log_ms, 99), stats.log_ms.empty() ? 0.0 : *std::max_element(stats.log_ms.begin(), stats.log_ms.end()));
  g_print("courttrack: %zu frames stepped; tracker + team model %.3f ms mean, %.3f ms p99, %.3f ms max per frame; "
          "%ld torso crops, histograms %.3f ms mean, %.3f ms max per batch; Kafka sent %ld, failed %ld\n",
          stats.step_ms.size(),
          stats.step_ms.empty() ? 0.0 : std::accumulate(stats.step_ms.begin(), stats.step_ms.end(), 0.0) / stats.step_ms.size(),
          pct(stats.step_ms, 99), stats.step_ms.empty() ? 0.0 : *std::max_element(stats.step_ms.begin(), stats.step_ms.end()),
          stats.crops,
          stats.hist_ms.empty() ? 0.0 : std::accumulate(stats.hist_ms.begin(), stats.hist_ms.end(), 0.0) / stats.hist_ms.size(),
          stats.hist_ms.empty() ? 0.0 : *std::max_element(stats.hist_ms.begin(), stats.hist_ms.end()),
          kafka.sent, kafka.failed);
  tracker.reset();
}
