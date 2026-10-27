/* Gym MV3DT online team segmentation: two teams and everyone else from torso colour, without an operator.
 * C++ port of tools/online_teams.py (Python stays the reference). Causal: labels for frame f come from a model built
 * from frames before f.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef COURTTRACK_ONLINE_TEAMS_H
#define COURTTRACK_ONLINE_TEAMS_H

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "courttrack.h"

namespace courttrack {

using Hist = std::array<float, FEAT_DIM>;   // torso HSV histogram (18 hue x 8 saturation), sums to 1

struct OTParams {                      // defaults = tools/online_teams.py OTParams defaults
  int k = 3;                           // clusters: two teams, everyone else
  int every = 150;                     // frames between re-clusterings (5 s)
  int window = 5400;                   // frames of track history used (3 min)
  int warmup = 900;                    // frames before the first model (30 s)
  int min_tracks = 8;
  int min_samples = 10;
  int track_recent = 900;
  double temperature = 0.05;
  double label_p = 0.8;
  double radius_q = 80;
  double min_team_share = 0.1;
  double team_match = 2.0;             // later models: team names follow the nearest clusters (see .py)
  uint64_t seed = 0;
  bool warm_only = true;               // after the first model, re-cluster only from the previous centres
  std::string ref_label = "other";     // label of the third cluster ("other" = no label)
};

/* Deterministic generator shared with the Python reference (tools/online_teams.SplitMix64). */
struct SplitMix64 {
  uint64_t s;
  explicit SplitMix64(uint64_t seed = 0) : s(seed) {}
  uint64_t next();
  double uniform();
  int choice(const std::vector<double> &p);
};

class OnlineTeams {
 public:
  explicit OnlineTeams(const OTParams &p = OTParams{});
  /* Re-cluster if due (call once per frame, before labelling that frame). */
  void update(int frame);
  /* sqrt of a good crop's histogram (the sample used everywhere below). */
  static Hist sample(const Hist &hist);
  /* Team label ("A", "B" or "" for none) and its probability, from the current model. */
  std::pair<std::string, double> label(int cam, const Hist &s) const;
  /* Unit-norm colour feature (camera-centred) for the per-player look; false before the camera has any crop. */
  bool feature(int cam, const Hist &s, Hist *out) const;
  /* Learn the camera's mean colour from a good crop (after labelling it). */
  void observe_crop(int cam, const Hist &s);
  /* Feedback from the tracker: one detection of court track `tid` (s = nullptr if no good crop). */
  void observe_track(int tid, int frame, int cam, const Hist *s, bool has_read);
  bool ready() const { return !centres_.empty(); }
  /* Last clustering: frame, roles and shares, for logs. */
  std::string summary() const { return summary_; }

 private:
  struct Sample {
    int frame;
    int cam;
    Hist s;
  };
  Hist centred(int cam, const Hist &s) const;
  OTParams p_;
  SplitMix64 rng_;
  std::vector<int> cam_order_;              // cameras in the order first seen (Python dict order)
  std::map<int, Hist> cam_sum_;
  std::map<int, int> cam_n_;
  std::vector<int> tid_order_;              // track ids in the order first seen
  std::map<int, std::vector<Sample>> samples_;
  std::map<int, std::vector<std::pair<int, bool>>> dets_;
  std::vector<std::vector<double>> centres_;
  std::vector<std::string> roles_;
  std::vector<double> radius_;
  std::vector<std::vector<double>> prev_a_, prev_b_;   // previous team centres (0 or 1 entries)
  std::string summary_;
};

}  // namespace courttrack

#endif
