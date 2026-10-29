/* Gym MV3DT court tracker: C++ port of the default configuration of tools/world_tracker.py.
 * Comments name the Python method each part mirrors; behaviour that decides ties (insertion-ordered counters, stable
 * sorts, list order) follows the Python so that both produce the same output on the same input.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "courttrack.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <cstdio>
#include <cstdlib>
#include <set>

namespace courttrack {

const char *const UNKNOWN_TEAM = "other";

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();

/* ---- small fixed-size linear algebra ---- */
using V2 = std::array<double, 2>;
using M2 = std::array<double, 4>;        // row-major 2x2
using V4 = std::array<double, 4>;
using M4 = std::array<double, 16>;       // row-major 4x4

M2 m2_add(const M2 &a, const M2 &b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]}; }
double m2_det(const M2 &a) { return a[0] * a[3] - a[1] * a[2]; }
M2 m2_inv(const M2 &a) {
  double d = m2_det(a);
  return {a[3] / d, -a[1] / d, -a[2] / d, a[0] / d};
}
V2 m2_mul(const M2 &a, const V2 &v) { return {a[0] * v[0] + a[1] * v[1], a[2] * v[0] + a[3] * v[1]}; }
double mahal(const V2 &d, const M2 &S) {   // d^T S^-1 d
  V2 s = m2_mul(m2_inv(S), d);
  return d[0] * s[0] + d[1] * s[1];
}
M4 m4_mul(const M4 &a, const M4 &b) {
  M4 c{};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      double s = 0;
      for (int k = 0; k < 4; ++k) s += a[i * 4 + k] * b[k * 4 + j];
      c[i * 4 + j] = s;
    }
  return c;
}
M4 m4_t(const M4 &a) {
  M4 c{};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) c[j * 4 + i] = a[i * 4 + j];
  return c;
}
M2 top_left(const M4 &P) { return {P[0], P[1], P[4], P[5]}; }

/* ---- insertion-ordered counter (Python collections.Counter semantics for most_common(1)) ---- */
template <typename K>
struct OCounter {
  std::vector<std::pair<K, double>> items;
  void add(const K &k, double w = 1.0) {
    for (auto &kv : items)
      if (kv.first == k) {
        kv.second += w;
        return;
      }
    items.emplace_back(k, w);
  }
  double total() const {
    double s = 0;
    for (auto &kv : items) s += kv.second;
    return s;
  }
  /* first key with the largest count (max() over items in insertion order) */
  const std::pair<K, double> *most_common() const {
    const std::pair<K, double> *best = nullptr;
    for (auto &kv : items)
      if (!best || kv.second > best->second) best = &kv;
    return best;
  }
  double get(const K &k) const {
    for (auto &kv : items)
      if (kv.first == k) return kv.second;
    return 0;
  }
  /* Counter a + b: a's keys first, then b's new keys; non-positive counts dropped */
  static OCounter sum(const OCounter &a, const OCounter &b) {
    OCounter c = a;
    for (auto &kv : b.items) c.add(kv.first, kv.second);
    c.items.erase(std::remove_if(c.items.begin(), c.items.end(), [](auto &kv) { return kv.second <= 0; }),
                  c.items.end());
    return c;
  }
};

struct Read {
  int frame;
  std::string number;
  double score;
};

struct Feat {
  int frame;
  std::array<float, FEAT_DIM> v;
};

bool compatible(const std::string &a, const std::string &b) { return a.empty() || b.empty() || a == b; }
bool conflict(const std::string &a, const std::string &b) { return !a.empty() && !b.empty() && a != b; }

}  // namespace

struct Track {
  static uint64_t next_uid;
  uint64_t uid = next_uid++;   // never reused (an address can be, once a merged track is freed)
  int tid = 0;
  V4 x{};
  M4 P{};
  int born = 0;
  std::vector<int> hits;
  int last_hit = 0;
  bool confirmed = false;
  std::vector<OutPoint> out;
  std::map<int, uint32_t> cams;                    // frame -> camera bit mask
  std::vector<std::pair<int, std::string>> labels; // (frame, team)
  std::vector<std::pair<int, double>> heights;
  OCounter<std::string> votes;
  std::vector<DetKey> dets;
  std::vector<Read> reads;
  OCounter<std::string> nvotes;
  bool has_R = false;
  M2 R{};
  bool id_linked = false;
  std::string ident;
  std::vector<Feat> feats;

  void add_reads(const std::vector<Read> &r) {
    for (auto &x : r) {
      reads.push_back(x);
      nvotes.add(x.number);
    }
  }
  void add_labels(const std::vector<std::pair<int, std::string>> &l) {
    for (auto &x : l) {
      labels.push_back(x);
      votes.add(x.second);
    }
  }
  std::string number(const Params &p) const {
    double n = nvotes.total();
    if (n < p.num_min_reads) return "";
    auto *mc = nvotes.most_common();
    return mc->second / n >= p.num_share ? mc->first : "";
  }
  std::string team(const Params &p) const {
    double n = votes.total();
    if (n < p.team_votes) return "";
    auto *mc = votes.most_common();
    return mc->second / n >= p.team_share ? mc->first : "";
  }
  std::string firm_team(const Params &p) const {
    double n = votes.total();
    if (n < p.team_firm_votes) return "";
    auto *mc = votes.most_common();
    return mc->second / n >= p.team_firm_share ? mc->first : "";
  }
  bool height(const Params &p, double *h) const {
    if ((int)heights.size() < p.height_min_n) return false;
    std::vector<double> v;
    for (auto &x : heights) v.push_back(x.second);
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    *h = n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    return true;
  }
};

uint64_t Track::next_uid = 0;

using TrackP = std::shared_ptr<Track>;

namespace {
/* A detection as it moves through one step: (cam, z, label, height, key) plus read and feature. */
struct Cand {
  int cam;
  V2 z;
  std::string lab;
  double h;
  DetKey key;
  const Detection *det;
};
}  // namespace

struct Tracker::Impl {
  Params p;
  std::vector<std::pair<double, double>> cam_xy;
  M4 F{}, Q{};
  std::vector<TrackP> tracks, finished;
  int next_id = 0;
  std::vector<std::pair<std::pair<int, int>, int>> close;           // (tid_a, tid_b) -> frames, insertion order
  std::vector<std::pair<std::string, OCounter<std::string>>> roster; // number -> team -> tracks
  std::set<std::tuple<uint64_t, std::string, std::string>> rostered;   // (track uid, team, number) counted
  struct Proto {
    std::array<float, FEAT_DIM> sum{};
    int n = 0;
  };
  std::map<std::string, Proto> proto, team_proto;
  std::map<DetKey, const Detection *> frame_dets;                   // detections of the current frame
  std::vector<DetIdentity> frame_out;

  explicit Impl(const Params &prm, const std::vector<std::pair<double, double>> &cxy) : p(prm), cam_xy(cxy) {
    double dt = 1.0 / FPS;
    F = {1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1};
    double G[4][2] = {{dt * dt / 2, 0}, {0, dt * dt / 2}, {dt, 0}, {0, dt}};
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) Q[i * 4 + j] = (G[i][0] * G[j][0] + G[i][1] * G[j][1]) * p.accel_sigma * p.accel_sigma;
  }

  /* _ray / _R */
  bool ray(int cam, const V2 &z, V2 *u) const {
    if (cam < 0 || cam >= (int)cam_xy.size()) return false;
    double vx = z[0] - cam_xy[cam].first, vy = z[1] - cam_xy[cam].second;
    double n = std::hypot(vx, vy) + 1e-9;
    *u = {vx / n, vy / n};
    return true;
  }
  M2 Rm(int cam, const V2 &z) const {
    V2 u;
    if (!ray(cam, z, &u)) {
      double s = p.meas_sigma * p.meas_sigma;
      return {s, 0, 0, s};
    }
    double r2 = p.sigma_radial * p.sigma_radial, t2 = p.sigma_tangential * p.sigma_tangential;
    M2 uu = {u[0] * u[0], u[0] * u[1], u[1] * u[0], u[1] * u[1]};
    return {r2 * uu[0] + t2 * (1 - uu[0]), r2 * uu[1] - t2 * uu[1], r2 * uu[2] - t2 * uu[2], r2 * uu[3] + t2 * (1 - uu[3])};
  }

  /* _gate_matrix (with cost_logdet) */
  std::vector<double> gate_matrix(const std::vector<V2> &Z, int frame, int cam) const {
    size_t T = tracks.size(), N = Z.size();
    std::vector<double> C(T * N, INF);
    for (size_t i = 0; i < T; ++i) {
      const Track &t = *tracks[i];
      double coast = std::max(0, frame - t.last_hit - 1) / double(FPS);
      double radius = std::min(p.gate_base_m + p.gate_speed * coast, p.gate_max_m);
      M2 Pt = top_left(t.P);
      for (size_t j = 0; j < N; ++j) {
        M2 S = m2_add(Pt, Rm(cam, Z[j]));
        V2 D = {Z[j][0] - t.x[0], Z[j][1] - t.x[1]};
        double m = mahal(D, S);
        V2 u;
        double dist;
        if (ray(cam, Z[j], &u)) {
          double dr = D[0] * u[0] + D[1] * u[1];
          double dtn = D[0] * -u[1] + D[1] * u[0];
          dist = std::hypot(dr * p.sigma_tangential / p.sigma_radial, dtn);
        } else {
          dist = std::hypot(D[0], D[1]);
        }
        if (dist <= radius && m <= p.gate_chi2) C[i * N + j] = m + std::log(m2_det(S));
      }
    }
    return C;
  }

  /* _fuse */
  void fuse(const std::vector<std::pair<int, V2>> &cz, V2 *z, M2 *R) const {
    M2 info{0, 0, 0, 0};
    V2 iz{0, 0};
    for (auto &c : cz) {
      M2 Ri = m2_inv(Rm(c.first, c.second));
      info = m2_add(info, Ri);
      V2 a = m2_mul(Ri, c.second);
      iz[0] += a[0];
      iz[1] += a[1];
    }
    *R = m2_inv(info);
    *z = m2_mul(*R, iz);
  }

  int index_of(const Track *t) const {
    for (size_t i = 0; i < tracks.size(); ++i)
      if (tracks[i].get() == t) return (int)i;
    return -1;
  }
  void remove_finished(const Track *t) {
    for (auto it = finished.begin(); it != finished.end(); ++it)
      if (it->get() == t) {
        finished.erase(it);
        return;
      }
  }

  int last_frame = -1;

  void step(int frame, const std::vector<CameraFrame> &cams) {
    last_frame = frame;
    frame_out.clear();
    frame_dets.clear();
    for (auto &cf : cams)
      for (auto &d : cf.dets) frame_dets[{frame, cf.cam, d.gid}] = &d;
    for (auto &tp : tracks) {     // predict
      Track &t = *tp;
      V4 nx{};
      for (int i = 0; i < 4; ++i) {
        double s = 0;
        for (int k = 0; k < 4; ++k) s += F[i * 4 + k] * t.x[k];
        nx[i] = s;
      }
      t.x = nx;
      M4 FP = m4_mul(F, t.P), FPF = m4_mul(FP, m4_t(F));
      for (int i = 0; i < 16; ++i) t.P[i] = FPF[i] + Q[i];
    }
    std::vector<std::string> teams, nums;
    for (auto &tp : tracks) {
      teams.push_back(tp->firm_team(p));
      nums.push_back(tp->number(p));
    }
    std::vector<std::pair<int, std::vector<Cand>>> assigned;   // track index -> candidates, insertion order
    auto assigned_at = [&](int i) -> std::vector<Cand> & {
      for (auto &a : assigned)
        if (a.first == i) return a.second;
      assigned.emplace_back(i, std::vector<Cand>{});
      return assigned.back().second;
    };
    std::vector<Cand> leftovers;
    for (auto &cf : cams) {
      if (cf.dets.empty()) continue;
      std::vector<Cand> cand;
      std::vector<V2> Z;
      for (auto &d : cf.dets) {
        cand.push_back({cf.cam, {d.x, d.y}, d.team, d.height, {frame, cf.cam, d.gid}, &d});
        Z.push_back({d.x, d.y});
      }
      if (tracks.empty()) {
        leftovers.insert(leftovers.end(), cand.begin(), cand.end());
        continue;
      }
      std::vector<double> C = gate_matrix(Z, frame, cf.cam);
      size_t N = Z.size(), T = tracks.size();
      for (size_t j = 0; j < N; ++j) {
        const Detection &d = cf.dets[j];
        if (!d.team.empty() && d.team_p >= p.team_penalty_p)
          for (size_t i = 0; i < T; ++i)
            if (!teams[i].empty() && teams[i] != d.team && std::isfinite(C[i * N + j])) C[i * N + j] += p.team_penalty;
      }
      for (size_t j = 0; j < N; ++j) {
        const Detection &d = cf.dets[j];
        if (d.has_read && d.read_score >= p.num_penalty_score)
          for (size_t i = 0; i < T; ++i)
            if (!nums[i].empty() && nums[i] != d.number && std::isfinite(C[i * N + j])) C[i * N + j] += p.num_penalty;
      }
      const double big = 1e9;
      std::vector<double> Cb(C);
      for (auto &c : Cb)
        if (!std::isfinite(c)) c = big;
      std::vector<bool> used(N, false);
      for (auto &rc : linear_sum_assignment(Cb, (int)T, (int)N))
        if (C[rc.first * N + rc.second] < big) {
          assigned_at(rc.first).push_back(cand[rc.second]);
          used[rc.second] = true;
        }
      for (size_t j = 0; j < N; ++j)
        if (!used[j]) leftovers.push_back(cand[j]);
    }
    /* consistency: keep detections consistent with the one closest to the prediction */
    for (auto &a : assigned) {
      auto &cz = a.second;
      if (cz.size() <= 1) continue;
      const Track &t = *tracks[a.first];
      std::stable_sort(cz.begin(), cz.end(), [&](const Cand &u, const Cand &v) {
        return std::hypot(u.z[0] - t.x[0], u.z[1] - t.x[1]) < std::hypot(v.z[0] - t.x[0], v.z[1] - t.x[1]);
      });
      V2 ref = cz[0].z;
      M2 Rr = Rm(cz[0].cam, cz[0].z);
      std::vector<Cand> keep;
      for (auto &c : cz) {
        V2 d = {c.z[0] - ref[0], c.z[1] - ref[1]};
        if (mahal(d, m2_add(Rm(c.cam, c.z), Rr)) <= p.consistency_chi2) keep.push_back(c);
        else leftovers.push_back(c);
      }
      cz = keep;
    }
    /* update: unidentified tracks first (stable) */
    std::stable_sort(assigned.begin(), assigned.end(),
                     [&](const auto &u, const auto &v) { return nums[u.first].empty() && !nums[v.first].empty(); });
    for (auto &a : assigned) {
      TrackP tp = tracks[a.first];
      Track *t = tp.get();
      auto &cz = a.second;
      std::vector<std::pair<int, V2>> czz;
      for (auto &c : cz) czz.emplace_back(c.cam, c.z);
      V2 z;
      M2 R;
      fuse(czz, &z, &R);
      t->R = R;
      t->has_R = true;
      M2 S = m2_add(top_left(t->P), R);
      M2 Si = m2_inv(S);
      double K[4][2];   // P H^T S^-1
      for (int i = 0; i < 4; ++i) {
        double ph0 = t->P[i * 4 + 0], ph1 = t->P[i * 4 + 1];
        K[i][0] = ph0 * Si[0] + ph1 * Si[2];
        K[i][1] = ph0 * Si[1] + ph1 * Si[3];
      }
      V2 inn = {z[0] - t->x[0], z[1] - t->x[1]};
      for (int i = 0; i < 4; ++i) t->x[i] += K[i][0] * inn[0] + K[i][1] * inn[1];
      M4 IKH{};
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) IKH[i * 4 + j] = (i == j ? 1.0 : 0.0) - (j == 0 ? K[i][0] : j == 1 ? K[i][1] : 0.0);
      t->P = m4_mul(IKH, t->P);
      t->hits.push_back(frame);
      t->last_hit = frame;
      t->out.push_back({frame, t->x[0], t->x[1], (int)cz.size()});
      uint32_t mask = 0;
      for (auto &c : cz) mask |= 1u << c.cam;
      t->cams[frame] = mask;
      std::vector<std::pair<int, std::string>> labs;
      for (auto &c : cz)
        if (!c.lab.empty()) labs.emplace_back(frame, c.lab);
      t->add_labels(labs);
      for (auto &c : cz)
        if (std::isfinite(c.h)) t->heights.emplace_back(frame, c.h);
      for (auto &c : cz) t->dets.push_back(c.key);
      std::string prev_num = t->number(p);
      std::vector<Read> rd;   /* _schedule_reads without a read budget: every read of an assigned detection */
      for (auto &c : cz)
        if (c.det->has_read) rd.push_back({frame, c.det->number, c.det->read_score});
      t->add_reads(rd);
      int recent = 0;
      for (int h : t->hits)
        if (h > frame - p.confirm_window) ++recent;
      if (!t->confirmed && recent >= p.confirm_hits) {
        t->confirmed = true;
        TrackP r = relink(tp, frame);
        tp = r;
        t = tp.get();
        prev_num = "";
      }
      if (t->confirmed) number_events(tp, frame, prev_num);
    }
    births(frame, leftovers);
    merge(frame);
    label_identities(frame);
    std::vector<TrackP> keep;
    for (auto &tp : tracks) {
      int limit = tp->confirmed ? p.max_coast : p.tentative_coast;
      if (frame - tp->last_hit <= limit) keep.push_back(tp);
      else finished.push_back(tp);
    }
    tracks = keep;
  }

  /* _relink: attach a newly confirmed track to the best recently lost track */
  TrackP relink(TrackP tp, int frame) {
    (void)frame;
    Track &t = *tp;
    auto mean_xy = [](const std::vector<OutPoint> &o, size_t a, size_t b) {
      V2 m{0, 0};
      for (size_t i = a; i < b; ++i) {
        m[0] += o[i].x;
        m[1] += o[i].y;
      }
      m[0] /= double(b - a);
      m[1] /= double(b - a);
      return m;
    };
    V2 start = mean_xy(t.out, 0, std::min<size_t>(5, t.out.size()));
    std::string tt = t.team(p);
    double th;
    bool has_th = t.height(p, &th);
    TrackP best;
    double best_cost = INF;
    for (auto &lp : finished) {
      Track &lost = *lp;
      if (!lost.confirmed) continue;
      std::string lt = lost.team(p);
      double lh;
      bool has_lh = lost.height(p, &lh);
      if (!compatible(lt, tt) || (has_lh && has_th && std::fabs(lh - th) > p.height_tol)) continue;
      if (conflict(lost.number(p), t.number(p))) continue;
      bool same_team = !lt.empty() && lt == tt;
      int gap = t.out.front().frame - lost.out.back().frame;
      int limit = same_team ? p.stitch_gap_team : p.stitch_gap;
      if (gap < 1 || gap > limit) continue;
      size_t n = lost.out.size();
      V2 end = mean_xy(lost.out, n > 5 ? n - 5 : 0, n);
      double dist = std::hypot(start[0] - end[0], start[1] - end[1]);
      double allow = p.stitch_d0 + p.stitch_vmax * gap / double(FPS);
      if (dist > allow) continue;
      double cost = dist / allow + double(gap) / limit + (same_team ? 0.0 : 0.3);
      if (cost < best_cost) {
        best = lp;
        best_cost = cost;
      }
    }
    if (!best) return tp;
    Track &b = *best;
    b.out.insert(b.out.end(), t.out.begin(), t.out.end());
    for (auto &kv : t.cams) b.cams[kv.first] = kv.second;
    b.add_labels(t.labels);
    b.add_reads(t.reads);
    b.dets.insert(b.dets.end(), t.dets.begin(), t.dets.end());
    b.heights.insert(b.heights.end(), t.heights.begin(), t.heights.end());
    b.hits.insert(b.hits.end(), t.hits.begin(), t.hits.end());
    b.x = t.x;
    b.P = t.P;
    b.last_hit = t.last_hit;
    remove_finished(best.get());
    tracks[index_of(&t)] = best;
    return best;
  }

  /* _number_events (split_majority) */
  void number_events(TrackP tp, int frame, const std::string &prev_num) {
    Track &t = *tp;
    std::string num = t.number(p);
    if (!t.id_linked && !num.empty() && !t.firm_team(p).empty()) {
      t.id_linked = true;
      id_relink(tp);
      return;
    }
    if (prev_num.empty()) return;
    std::vector<const Read *> win;
    for (auto &r : t.reads)
      if (frame - p.num_switch_window < r.frame && r.frame <= frame) win.push_back(&r);
    if ((int)win.size() < p.num_switch_reads) return;
    OCounter<std::string> c;
    for (auto *r : win) c.add(r->number);
    auto *mc = c.most_common();
    if (mc->first != prev_num && mc->second / win.size() >= p.split_share) {
      int since = std::numeric_limits<int>::max();
      for (auto *r : win)
        if (r->number == mc->first) since = std::min(since, r->frame);
      split(tp, since);
    }
  }

  /* _split: frames >= since belong to a new track */
  void split(TrackP tp, int since) {
    Track &t = *tp;
    auto nt = std::make_shared<Track>();
    Track &n = *nt;
    n.tid = next_id++;
    n.x = t.x;
    n.P = t.P;
    n.born = since;
    for (int h : t.hits)
      if (h >= since) n.hits.push_back(h);
    n.last_hit = t.last_hit;
    n.confirmed = true;
    for (auto &o : t.out)
      if (o.frame >= since) n.out.push_back(o);
    for (auto &kv : t.cams)
      if (kv.first >= since) n.cams[kv.first] = kv.second;
    std::vector<std::pair<int, std::string>> nl, ol;
    for (auto &l : t.labels) (l.first >= since ? nl : ol).push_back(l);
    n.add_labels(nl);
    std::vector<Read> nr, orr;
    for (auto &r : t.reads) (r.frame >= since ? nr : orr).push_back(r);
    n.add_reads(nr);
    for (auto &h : t.heights)
      if (h.first >= since) n.heights.push_back(h);
    for (auto &k : t.dets)
      if (k.frame >= since) n.dets.push_back(k);
    std::vector<OutPoint> oo;
    for (auto &o : t.out)
      if (o.frame < since) oo.push_back(o);
    t.out = oo;
    std::vector<int> oh;
    for (int h : t.hits)
      if (h < since) oh.push_back(h);
    t.hits = oh;
    for (auto it = t.cams.begin(); it != t.cams.end();) it = it->first >= since ? t.cams.erase(it) : std::next(it);
    t.labels.clear();
    t.votes = {};
    t.add_labels(ol);
    t.reads.clear();
    t.nvotes = {};
    t.add_reads(orr);
    std::vector<std::pair<int, double>> ohh;
    for (auto &h : t.heights)
      if (h.first < since) ohh.push_back(h);
    t.heights = ohh;
    std::vector<DetKey> od;
    for (auto &k : t.dets)
      if (k.frame < since) od.push_back(k);
    t.dets = od;
    t.last_hit = t.out.empty() ? since - 1 : t.out.back().frame;
    tracks[index_of(&t)] = nt;
    if (!t.out.empty()) finished.push_back(tp);
    id_relink(nt);
  }

  /* prune: a finished track can only be picked by relink (at a track's confirmation) or id_relink (when a track's
   * identity completes), and only if it ended at most stitch_gap_team / id_gap frames before the picking track's first
   * frame. Tracks that may still pick: unconfirmed and not yet identity-linked live tracks, new tracks (born later, or
   * split off a live track from up to num_switch_window frames back), and finished confirmed tracks not yet identity-linked that a re-link may revive (closure). Unconfirmed
   * finished tracks are never picked. */
  size_t prune(int frame) {
    const int g = std::max({p.stitch_gap, p.stitch_gap_team, p.id_gap});
    int m = frame - p.num_switch_window;   // a split's new track starts up to num_switch_window frames back
    for (auto &tp : tracks)
      if ((!tp->confirmed || !tp->id_linked) && !tp->out.empty()) m = std::min(m, tp->out.front().frame);
    for (bool changed = true; changed;) {
      changed = false;
      for (auto &tp : finished)
        if (tp->confirmed && !tp->id_linked && !tp->out.empty() && tp->out.back().frame >= m - g &&
            tp->out.front().frame < m) {
          m = tp->out.front().frame;
          changed = true;
        }
    }
    std::vector<TrackP> keep;
    for (auto &tp : finished)
      if (tp->confirmed && !tp->out.empty() && tp->out.back().frame >= m - g) keep.push_back(tp);
    finished.swap(keep);
    return finished.size();
  }

  /* _id_relink: re-link to the last lost track with the same identity (team + number) */
  void id_relink(TrackP tp) {
    Track &t = *tp;
    std::string num = t.number(p), team = t.firm_team(p);
    TrackP best;
    for (auto &lp : finished) {
      Track &lost = *lp;
      if (!lost.confirmed || lost.number(p) != num || !compatible(lost.firm_team(p), team)) continue;
      if (team.empty() || lost.firm_team(p) != team) continue;
      int gap = t.out.front().frame - lost.out.back().frame;
      if (gap >= 1 && gap <= p.id_gap && (!best || lost.out.back().frame > best->out.back().frame)) best = lp;
    }
    if (!best) return;
    Track &b = *best;
    std::vector<OutPoint> out = b.out;
    out.insert(out.end(), t.out.begin(), t.out.end());
    t.out = out;
    std::map<int, uint32_t> cams = b.cams;
    for (auto &kv : t.cams) cams[kv.first] = kv.second;
    t.cams = cams;
    auto labels = b.labels;
    labels.insert(labels.end(), t.labels.begin(), t.labels.end());
    t.labels = labels;
    t.votes = OCounter<std::string>::sum(b.votes, t.votes);
    auto reads = b.reads;
    reads.insert(reads.end(), t.reads.begin(), t.reads.end());
    t.reads = reads;
    t.nvotes = OCounter<std::string>::sum(b.nvotes, t.nvotes);
    auto heights = b.heights;
    heights.insert(heights.end(), t.heights.begin(), t.heights.end());
    t.heights = heights;
    auto dets = b.dets;
    dets.insert(dets.end(), t.dets.begin(), t.dets.end());
    t.dets = dets;
    auto hits = b.hits;
    hits.insert(hits.end(), t.hits.begin(), t.hits.end());
    t.hits = hits;
    t.born = b.born;
    t.tid = b.tid;
    remove_finished(best.get());
  }

  /* _merge: duplicates (disjoint cameras, close) and same identity */
  void merge(int frame) {
    std::vector<TrackP> live;
    for (auto &tp : tracks)
      if (tp->confirmed && tp->last_hit == frame) live.push_back(tp);
    if (live.empty()) return;
    std::set<std::pair<int, int>> seen;
    for (size_t a = 0; a < live.size(); ++a)
      for (size_t b = a + 1; b < live.size(); ++b) {
        Track &ta = *live[a], &tb = *live[b];
        std::pair<int, int> key = {std::min(ta.tid, tb.tid), std::max(ta.tid, tb.tid)};
        V2 dv = {ta.x[0] - tb.x[0], ta.x[1] - tb.x[1]};
        double d = std::hypot(dv[0], dv[1]);
        bool cl = d < 0.9;   // merge_dist (isotropic, tracks without a fused measurement)
        if (ta.has_R && tb.has_R) cl = mahal(dv, m2_add(ta.R, tb.R)) <= p.merge_chi2;
        std::string na = ta.number(p), nb = tb.number(p);
        if (conflict(na, nb)) continue;
        std::string tma = ta.team(p), tmb = tb.team(p);
        bool same_id = !na.empty() && na == nb && compatible(tma, tmb) && !tma.empty() && tma == tmb;
        auto ca = ta.cams.find(frame), cb = tb.cams.find(frame);
        bool disjoint = ca == ta.cams.end() || cb == tb.cams.end() || (ca->second & cb->second) == 0;
        if ((cl && disjoint) || (same_id && d < p.id_merge_dist)) {
          bool found = false;
          for (auto &kv : close)
            if (kv.first == key) {
              ++kv.second;
              found = true;
            }
          if (!found) close.emplace_back(key, 1);
          seen.insert(key);
        }
      }
    close.erase(std::remove_if(close.begin(), close.end(), [&](auto &kv) { return !seen.count(kv.first); }),
                close.end());
    auto snapshot = close;
    for (auto &kv : snapshot) {
      int n = kv.second;
      if (n < p.merge_frames) continue;
      TrackP ta, tb;
      for (auto &tp : tracks) {
        if (!ta && tp->tid == kv.first.first) ta = tp;
        if (!tb && tp->tid == kv.first.second) tb = tp;
      }
      if (!ta || !tb) continue;
      TrackP oldp = ta->born <= tb->born ? ta : tb, newp = ta->born <= tb->born ? tb : ta;
      Track &old = *oldp, &nw = *newp;
      int since = frame - n + 1;
      std::map<int, const OutPoint *> old_at;
      for (auto &o : old.out) old_at[o.frame] = &o;
      std::set<int> since_set;
      for (auto &o : nw.out) {
        if (o.frame >= since) since_set.insert(o.frame);
        else {
          auto it = old_at.find(o.frame);
          if (it != old_at.end() && std::hypot(o.x - it->second->x, o.y - it->second->y) < 1.5) since_set.insert(o.frame);
        }
      }
      std::vector<OutPoint> prefix;
      for (auto &o : nw.out)
        if (!since_set.count(o.frame)) prefix.push_back(o);
      if (!prefix.empty()) {   // keep the younger track's earlier history as a finished track
        auto pre = std::make_shared<Track>();
        pre->tid = nw.tid;
        pre->x = nw.x;
        pre->P = nw.P;
        pre->born = nw.born;
        for (int h : nw.hits)
          if (!since_set.count(h)) pre->hits.push_back(h);
        pre->last_hit = prefix.back().frame;
        pre->confirmed = true;
        pre->out = prefix;
        for (auto &kv2 : nw.cams)
          if (!since_set.count(kv2.first)) pre->cams[kv2.first] = kv2.second;
        std::vector<std::pair<int, std::string>> pl;
        for (auto &l : nw.labels)
          if (!since_set.count(l.first)) pl.push_back(l);
        pre->add_labels(pl);
        std::vector<Read> pr;
        for (auto &r : nw.reads)
          if (!since_set.count(r.frame)) pr.push_back(r);
        pre->add_reads(pr);
        for (auto &k : nw.dets)
          if (!since_set.count(k.frame)) pre->dets.push_back(k);
        for (auto &h : nw.heights)
          if (!since_set.count(h.first)) pre->heights.push_back(h);
        finished.push_back(pre);
      }
      std::map<int, std::vector<const OutPoint *>> per;
      for (auto &o : old.out) per[o.frame].push_back(&o);
      for (auto &o : nw.out)
        if (since_set.count(o.frame)) per[o.frame].push_back(&o);
      std::vector<OutPoint> merged;
      for (auto &kv2 : per) {
        double sx = 0, sy = 0;
        int sk = 0;
        for (auto *o : kv2.second) {
          sx += o->x;
          sy += o->y;
          sk += o->n_cams;
        }
        merged.push_back({kv2.first, sx / kv2.second.size(), sy / kv2.second.size(), sk});
      }
      old.out = merged;
      for (auto &kv2 : nw.cams)
        if (since_set.count(kv2.first)) old.cams[kv2.first] |= kv2.second;
      std::vector<std::pair<int, std::string>> ml;
      for (auto &l : nw.labels)
        if (since_set.count(l.first)) ml.push_back(l);
      old.add_labels(ml);
      std::vector<Read> mr;
      for (auto &r : nw.reads)
        if (since_set.count(r.frame)) mr.push_back(r);
      old.add_reads(mr);
      for (auto &k : nw.dets)
        if (since_set.count(k.frame)) old.dets.push_back(k);
      for (auto &h : nw.heights)
        if (since_set.count(h.first)) old.heights.push_back(h);
      std::set<int> hs(old.hits.begin(), old.hits.end());
      for (int h : nw.hits)
        if (since_set.count(h)) hs.insert(h);
      old.hits.assign(hs.begin(), hs.end());
      old.last_hit = std::max(old.last_hit, nw.last_hit);
      for (int i = 0; i < 4; ++i) old.x[i] = (old.x[i] + nw.x[i]) / 2;
      for (int i = 0; i < 16; ++i) old.P[i] = (old.P[i] + nw.P[i]) / 2;
      tracks.erase(std::remove(tracks.begin(), tracks.end(), newp), tracks.end());
      close.erase(std::remove_if(close.begin(), close.end(), [&](auto &c) { return c.first == kv.first; }),
                  close.end());
    }
  }

  /* _births */
  void births(int frame, std::vector<Cand> leftovers) {
    std::stable_sort(leftovers.begin(), leftovers.end(),
                     [](const Cand &a, const Cand &b) { return a.cam < b.cam; });   // camera indices follow name order
    struct Group {
      uint32_t cams = 0;
      std::vector<Cand> members;
    };
    std::vector<Group> groups;
    for (auto &c : leftovers) {
      Group *best = nullptr;
      for (auto &g : groups) {
        if (g.cams & (1u << c.cam)) continue;
        std::vector<std::pair<int, V2>> gz;
        for (auto &m : g.members) gz.emplace_back(m.cam, m.z);
        V2 zg;
        M2 Rg;
        fuse(gz, &zg, &Rg);
        V2 d = {c.z[0] - zg[0], c.z[1] - zg[1]};
        if (mahal(d, m2_add(Rg, Rm(c.cam, c.z))) <= p.consistency_chi2) {
          best = &g;
          break;
        }
      }
      if (best) {
        best->cams |= 1u << c.cam;
        best->members.push_back(c);
      } else {
        groups.push_back({1u << c.cam, {c}});
      }
    }
    for (auto &g : groups) {
      std::vector<std::pair<int, V2>> gz;
      for (auto &m : g.members) gz.emplace_back(m.cam, m.z);
      V2 z;
      M2 R;
      fuse(gz, &z, &R);
      auto t = std::make_shared<Track>();
      t->tid = next_id++;
      t->x = {z[0], z[1], 0, 0};
      t->P = {R[0], R[1], 0, 0, R[2], R[3], 0, 0, 0, 0, 4, 0, 0, 0, 0, 4};
      t->born = frame;
      t->hits = {frame};
      t->last_hit = frame;
      t->R = R;
      t->has_R = true;
      t->out.push_back({frame, z[0], z[1], (int)g.members.size()});
      t->cams[frame] = g.cams;
      std::vector<std::pair<int, std::string>> labs;
      for (auto &m : g.members)
        if (!m.lab.empty()) labs.emplace_back(frame, m.lab);
      t->add_labels(labs);
      for (auto &m : g.members)
        if (std::isfinite(m.h)) t->heights.emplace_back(frame, m.h);
      for (auto &m : g.members) t->dets.push_back(m.key);
      std::vector<Read> rd;
      for (auto &m : g.members)
        if (m.det->has_read) rd.push_back({frame, m.det->number, m.det->read_score});
      t->add_reads(rd);
      tracks.push_back(t);
    }
  }

  /* roster_teams */
  std::vector<std::string> roster_teams(const std::string &num) const {
    std::vector<std::string> out;
    for (auto &r : roster)
      if (r.first == num)
        for (auto &kv : r.second.items)
          if (kv.second >= p.roster_min_tracks) out.push_back(kv.first);
    return out;
  }

  /* recent_number */
  std::string recent_number(const Track &t, int frame) const {
    OCounter<std::string> w;
    int n = 0;
    for (auto it = t.reads.rbegin(); it != t.reads.rend(); ++it) {
      if (it->frame < frame - p.id_window) break;
      if (it->frame <= frame) {
        w.add(it->number, std::exp(-(frame - it->frame) / p.id_tau));
        ++n;
      }
    }
    if (n < p.num_min_reads) return "";
    auto *mc = w.most_common();
    return mc->second / w.total() >= p.num_share ? mc->first : "";
  }

  const Proto *proto_for(const std::string &ident, const std::string &team) const {
    auto a = proto.find(ident);
    if (a != proto.end() && a->second.n >= p.proto_min) return &a->second;
    auto b = team_proto.find(team);
    if (b != team_proto.end() && b->second.n >= p.proto_min) return &b->second;
    return nullptr;
  }

  /* _closest_player */
  std::string closest_player(const Track &t, const std::string &num, const std::vector<std::string> &teams) const {
    int last = t.out.empty() ? 0 : t.out.back().frame;
    std::array<float, FEAT_DIM> m{};
    int k = 0;
    for (auto &f : t.feats)
      if (f.frame > last - p.app_recent) {
        for (int i = 0; i < FEAT_DIM; ++i) m[i] += f.v[i];
        ++k;
      }
    if (k < p.app_min_samples) return "";
    for (auto &x : m) x /= k;
    std::vector<std::pair<double, std::string>> d;
    for (auto &tm : teams) {
      const Proto *pr = proto_for(tm + ":" + num, tm);
      if (!pr) return "";
      double s = 0;
      for (int i = 0; i < FEAT_DIM; ++i) {
        double e = m[i] - pr->sum[i] / pr->n;
        s += e * e;
      }
      d.emplace_back(std::sqrt(s), tm);
    }
    std::sort(d.begin(), d.end());
    return d[1].first - d[0].first >= p.app_margin ? d[0].second + ":" + num : "";
  }

  /* identity (id_recent, player appearance) */
  std::string identity(const Track &t, int frame) const {
    std::string num = recent_number(t, frame);
    if (num.empty() || !t.confirmed) return "";
    std::string team = t.firm_team(p);
    static const char *dbg = getenv("CT_DEBUG_FRAME");   // debugging: identity decisions at one frame
    if (dbg && atoi(dbg) == frame) {
      auto ts = roster_teams(num);
      fprintf(stderr, "identity tid %d num %s firm %s roster", t.tid, num.c_str(), team.c_str());
      for (auto &x : ts) fprintf(stderr, " %s", x.c_str());
      int last = t.out.empty() ? 0 : t.out.back().frame, k = 0;
      for (auto &f : t.feats) k += f.frame > last - p.app_recent;
      fprintf(stderr, " feats %d look '%s' ident '%s'\n", k, ts.size() > 1 ? closest_player(t, num, ts).c_str() : "",
              t.ident.c_str());
      if (ts.size() > 1)
        for (auto &tm : ts) {
          const Proto *pr = proto_for(tm + ":" + num, tm);
          std::array<float, FEAT_DIM> m{};
          int kk = 0;
          for (auto &f : t.feats)
            if (f.frame > last - p.app_recent) {
              for (int i = 0; i < FEAT_DIM; ++i) m[i] += f.v[i];
              ++kk;
            }
          for (auto &x : m) x /= kk;
          double ss = 0;
          if (pr)
            for (int i = 0; i < FEAT_DIM; ++i) {
              double e = m[i] - pr->sum[i] / pr->n;
              ss += e * e;
            }
          fprintf(stderr, "   %s proto n %d dist %.17g\n", tm.c_str(), pr ? pr->n : -1, std::sqrt(ss));
        }
    }
    auto teams = roster_teams(num);
    if (teams.size() > 1) {
      std::string by_look = closest_player(t, num, teams);
      if (!by_look.empty()) return by_look;
    }
    if (!team.empty() && team != UNKNOWN_TEAM && team != "ref") return team + ":" + num;
    if (team.empty() && teams.size() == 1) return teams[0] + ":" + num;
    return "";
  }

  /* _learn */
  void learn(const Track &t, const std::string &num, const std::string &team, const std::vector<const float *> &vs) {
    if (!t.confirmed || num.empty()) return;
    auto teams = roster_teams(num);
    std::string player_team = (team == UNKNOWN_TEAM || team == "ref") ? "" : team;
    std::string tm;
    bool unique;
    if (teams.size() == 1) {
      tm = teams[0];
      if (!player_team.empty() && player_team != tm) return;
      unique = true;
    } else if (!player_team.empty() && std::find(teams.begin(), teams.end(), player_team) != teams.end()) {
      tm = player_team;
      unique = false;
    } else {
      return;
    }
    std::array<float, FEAT_DIM> v{};
    for (auto *f : vs)
      for (int i = 0; i < FEAT_DIM; ++i) v[i] += f[i];
    auto add = [&](Proto &pr) {
      for (int i = 0; i < FEAT_DIM; ++i) pr.sum[i] += v[i];
      pr.n += (int)vs.size();
    };
    add(proto[tm + ":" + num]);
    if (unique) add(team_proto[tm]);
  }

  /* _label_identities */
  void label_identities(int frame) {
    for (auto &tp : tracks) {
      Track &t = *tp;
      if (t.last_hit != frame) continue;
      std::string num = t.number(p), team = t.firm_team(p);
      if (t.confirmed && !num.empty() && !team.empty() && team != UNKNOWN_TEAM && team != "ref") {
        auto key = std::make_tuple(t.uid, team, num);
        if (!rostered.count(key)) {
          rostered.insert(key);
          bool found = false;
          for (auto &r : roster)
            if (r.first == num) {
              r.second.add(team);
              found = true;
            }
          if (!found) {
            roster.emplace_back(num, OCounter<std::string>{});
            roster.back().second.add(team);
          }
        }
      }
      std::vector<Feat> fs;
      size_t start = t.dets.size() > 8 ? t.dets.size() - 8 : 0;
      for (size_t i = start; i < t.dets.size(); ++i) {
        const DetKey &k = t.dets[i];
        if (k.frame != frame) continue;
        auto it = frame_dets.find(k);
        if (it != frame_dets.end() && it->second->has_feat) fs.push_back({frame, it->second->feat});
      }
      if (!fs.empty()) {
        std::vector<Feat> kept;
        for (auto &f : t.feats)
          if (f.frame > frame - p.app_recent) kept.push_back(f);
        kept.insert(kept.end(), fs.begin(), fs.end());
        t.feats = kept;
        std::vector<const float *> vs;
        for (auto &f : fs) vs.push_back(f.v.data());
        learn(t, num, team, vs);
      }
      std::string nw = identity(t, frame);
      if (!nw.empty()) t.ident = nw;
      for (auto it = t.dets.rbegin(); it != t.dets.rend(); ++it) {
        if (it->frame != frame) break;
        frame_out.push_back({*it, t.ident, t.tid});
      }
    }
  }
};

Params for_stride(Params p, int stride) {
  if (stride <= 1) return p;
  p.confirm_window *= stride;
  p.tentative_coast *= stride;
  p.merge_frames = std::max(1, (int)std::lround(p.merge_frames / double(stride)));
  return p;
}

Tracker::Tracker(const Params &p, const std::vector<std::pair<double, double>> &cam_xy)
    : impl_(new Impl(p, cam_xy)) {}
Tracker::~Tracker() = default;

std::vector<DetIdentity> Tracker::step(int frame, const std::vector<CameraFrame> &cams) {
  impl_->step(frame, cams);
  return impl_->frame_out;
}

size_t Tracker::prune(int frame) { return impl_->prune(frame); }

std::vector<Tracker::TrackView> Tracker::result() const {
  std::vector<TrackView> out;
  for (auto *list : {&impl_->finished, &impl_->tracks})
    for (auto &tp : *list)
      if (tp->confirmed) out.push_back({tp->tid, &tp->out, &tp->dets});
  return out;
}

std::vector<LiveTrack> Tracker::live() const {
  std::vector<LiveTrack> out;
  for (auto &tp : impl_->tracks) {
    const Track &t = *tp;
    if (!t.confirmed) continue;
    LiveTrack l{t.tid, t.x[0], t.x[1], t.x[2], t.x[3], false, 0, t.ident, t.firm_team(impl_->p), {}};
    if (!t.out.empty() && t.out.back().frame == t.last_hit && t.last_hit == impl_->last_frame) {
      l.measured = true;
      l.n_cams = t.out.back().n_cams;
      for (auto it = t.dets.rbegin(); it != t.dets.rend() && it->frame == t.last_hit; ++it) l.dets.push_back(*it);
    }
    out.push_back(std::move(l));
  }
  return out;
}

std::vector<std::pair<std::string, std::vector<std::pair<std::string, int>>>> Tracker::roster() const {
  std::vector<std::pair<std::string, std::vector<std::pair<std::string, int>>>> out;
  for (auto &r : impl_->roster) {
    std::vector<std::pair<std::string, int>> v;
    for (auto &kv : r.second.items) v.emplace_back(kv.first, (int)kv.second);
    out.emplace_back(r.first, v);
  }
  return out;
}

/* ---- linear_sum_assignment: shortest augmenting path (Crouse 2016, as SciPy) ---- */
std::vector<std::pair<int, int>> linear_sum_assignment(const std::vector<double> &cost_in, int nr, int nc) {
  std::vector<std::pair<int, int>> res;
  if (nr == 0 || nc == 0) return res;
  bool transpose = nc < nr;
  std::vector<double> cost(cost_in);
  if (transpose) {
    cost.assign((size_t)nr * nc, 0);
    for (int i = 0; i < nr; ++i)
      for (int j = 0; j < nc; ++j) cost[(size_t)j * nr + i] = cost_in[(size_t)i * nc + j];
    std::swap(nr, nc);
  }
  std::vector<double> u(nr, 0), v(nc, 0), spc(nc);
  std::vector<int> path(nc, -1), col4row(nr, -1), row4col(nc, -1), remaining(nc);
  std::vector<char> SR(nr), SC(nc);
  for (int cur = 0; cur < nr; ++cur) {
    double minVal = 0;
    int i = cur, sink = -1, num_remaining = nc;
    for (int it = 0; it < nc; ++it) remaining[it] = nc - it - 1;
    std::fill(SR.begin(), SR.end(), 0);
    std::fill(SC.begin(), SC.end(), 0);
    std::fill(spc.begin(), spc.end(), INF);
    while (sink == -1) {
      int index = -1;
      double lowest = INF;
      SR[i] = 1;
      for (int it = 0; it < num_remaining; ++it) {
        int j = remaining[it];
        double r = minVal + cost[(size_t)i * nc + j] - u[i] - v[j];
        if (r < spc[j]) {
          path[j] = i;
          spc[j] = r;
        }
        if (spc[j] < lowest || (spc[j] == lowest && row4col[j] == -1)) {
          lowest = spc[j];
          index = it;
        }
      }
      minVal = lowest;
      if (minVal == INF) return res;   // infeasible (cannot happen with finite costs)
      int j = remaining[index];
      if (row4col[j] == -1) sink = j;
      else i = row4col[j];
      SC[j] = 1;
      remaining[index] = remaining[--num_remaining];
    }
    u[cur] += minVal;
    for (int r = 0; r < nr; ++r)
      if (SR[r] && r != cur) u[r] += minVal - spc[col4row[r]];
    for (int c = 0; c < nc; ++c)
      if (SC[c]) v[c] -= minVal - spc[c];
    int j = sink;
    while (true) {
      int r = path[j];
      row4col[j] = r;
      std::swap(col4row[r], j);
      if (r == cur) break;
    }
  }
  if (transpose) {   // rows of the original = columns here
    for (int r = 0; r < nr; ++r) res.emplace_back(col4row[r], r);
    std::sort(res.begin(), res.end());
  } else {
    for (int r = 0; r < nr; ++r) res.emplace_back(r, col4row[r]);
  }
  return res;
}

}  // namespace courttrack
