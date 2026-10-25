/* Replays recorded court-tracker inputs (tools/courttrack_export.py) through the C++ tracker and writes its output
 * for comparison with the Python version.
 *   ct_replay inputs.bin cpp_out.txt
 * CTIN1 inputs carry team labels and colour features (offline team model); CTIN2 inputs carry torso histograms, and
 * the online team model (online_teams.cpp) labels them, as in the deployed pipeline.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <algorithm>
#include <chrono>
#include <tuple>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>

#include "courttrack.h"
#include "online_teams.h"

using namespace courttrack;

namespace {
struct Reader {
  std::ifstream in;
  template <typename T>
  T get() {
    T v;
    in.read(reinterpret_cast<char *>(&v), sizeof v);
    if (!in) throw std::runtime_error("truncated input");
    return v;
  }
  std::string str() {
    int32_t n = get<int32_t>();
    std::string s(n, '\0');
    in.read(&s[0], n);
    return s;
  }
};
}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: ct_replay inputs.bin cpp_out.txt\n";
    return 2;
  }
  Reader r{std::ifstream(argv[1], std::ios::binary)};
  char magic[5];
  r.in.read(magic, 5);
  bool online = r.in && !std::memcmp(magic, "CTIN2", 5);
  if (!r.in || (!online && std::memcmp(magic, "CTIN1", 5))) {
    std::cerr << "not a court-tracker input file\n";
    return 1;
  }
  int ncams = r.get<int32_t>();
  std::vector<std::pair<double, double>> cam_xy;
  for (int i = 0; i < ncams; ++i) {
    double x = r.get<double>(), y = r.get<double>();
    cam_xy.emplace_back(x, y);
    std::cerr << "camera " << i << " " << r.str() << "\n";
  }
  Tracker tracker{Params{}, cam_xy};
  OnlineTeams teams;
  struct Colour {
    bool has = false;
    Hist s{};
    bool read = false;
  };
  FILE *out = std::fopen(argv[2], "w");
  FILE *labels = getenv("CT_LABELS") ? std::fopen(getenv("CT_LABELS"), "w") : nullptr;   // team label per crop
  double busy = 0;
  std::vector<std::tuple<double, int, double>> slow;   // (ms, frame, ms in the team model update)
  long frames = 0, ndets = 0;
  while (true) {
    int frame = r.get<int32_t>();
    if (frame < 0) break;
    std::vector<CameraFrame> cams(r.get<int32_t>());
    std::map<DetKey, Colour> colour;
    for (auto &cf : cams) {
      cf.cam = r.get<int32_t>();
      cf.dets.resize(r.get<int32_t>());
      for (auto &d : cf.dets) {
        d.x = r.get<double>();
        d.y = r.get<double>();
        d.team_p = r.get<double>();
        d.height = r.get<double>();
        d.gid = r.get<int64_t>();
        d.team = r.str();
        d.has_read = r.get<int32_t>();
        d.read_score = r.get<double>();
        d.number = r.str();
        d.has_feat = r.get<int32_t>();
        if (d.has_feat) r.in.read(reinterpret_cast<char *>(d.feat.data()), sizeof(float) * FEAT_DIM);
        ++ndets;
      }
      // keep the first detection of each object id (recordings made before the probe did this itself)
      std::vector<Detection> uniq;
      for (auto &d : cf.dets)
        if (std::none_of(uniq.begin(), uniq.end(), [&](const Detection &u) { return u.gid == d.gid; }))
          uniq.push_back(d);
      cf.dets.swap(uniq);
    }
    auto t0 = std::chrono::steady_clock::now();
    double upd = 0;
    if (online) {   // has_feat/feat hold a torso histogram: label, feature, then learn the camera colour
      auto u0 = std::chrono::steady_clock::now();
      teams.update(frame);
      upd = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - u0).count();
      for (auto &cf : cams)
        for (auto &d : cf.dets) {
          Colour &col = colour[{frame, cf.cam, d.gid}];
          col.read = d.has_read;
          d.team.clear();
          d.team_p = 0;
          if (d.has_feat) {
            col.has = true;
            col.s = OnlineTeams::sample(d.feat);
            auto lab = teams.label(cf.cam, col.s);
            d.team = lab.first;
            d.team_p = lab.second;
            d.has_feat = teams.feature(cf.cam, col.s, &d.feat);
            if (labels) std::fprintf(labels, "%d %d %lld %s %.17g\n", frame, cf.cam, (long long)d.gid,
                                     d.team.empty() ? "-" : d.team.c_str(), d.team_p);
            teams.observe_crop(cf.cam, col.s);
          }
        }
    }
    auto ids = tracker.step(frame, cams);
    if (online)
      for (auto &di : ids) {
        const Colour &col = colour[di.key];
        teams.observe_track(di.tid, frame, di.key.cam, col.has ? &col.s : nullptr, col.read);
      }
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    busy += dt;
    slow.emplace_back(1000 * dt, frame, upd);
    ++frames;
    for (auto &di : ids)
      std::fprintf(out, "D %d %d %lld %s\n", di.key.frame, di.key.cam, (long long)di.key.gid,
                   di.identity.empty() ? "-" : di.identity.c_str());
  }
  auto res = tracker.result();
  for (auto &t : res) {
    std::fprintf(out, "T %d\n", t.tid);
    for (auto &o : *t.out) std::fprintf(out, "O %d %.17g %.17g %d\n", o.frame, o.x, o.y, o.n_cams);
    for (auto &k : *t.dets) std::fprintf(out, "K %d %d %lld\n", k.frame, k.cam, (long long)k.gid);
  }
  std::fclose(out);
  if (labels) std::fclose(labels);
  std::sort(slow.rbegin(), slow.rend());
  std::cerr << "slowest frames (ms, frame, of which team model update):";
  for (size_t i = 0; i < std::min<size_t>(8, slow.size()); ++i)
    std::cerr << " (" << std::get<0>(slow[i]) << ", " << std::get<1>(slow[i]) << ", " << std::get<2>(slow[i]) << ")";
  std::cerr << "\n";
  std::cerr << frames << " frames, " << ndets << " detections, " << res.size() << " confirmed tracks; tracker "
            << 1000 * busy / frames << " ms/frame\n";
  return 0;
}
