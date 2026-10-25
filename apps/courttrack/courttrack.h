/* Courtside court tracker: multi-camera player tracking on the court plane with identity output.
 * C++ port of the default configuration of tools/world_tracker.py (Python stays the reference; options that were
 * tested and not adopted are not ported). Causal: everything it reports for a frame is final.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef COURTTRACK_H
#define COURTTRACK_H

#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace courttrack {

constexpr int FPS = 30;
constexpr int FEAT_DIM = 144;            // per-camera-centred sqrt torso HSV histogram (18 x 8), unit norm
extern const char *const UNKNOWN_TEAM;   // colour label that carries no team information ("other")

struct Params {                          // defaults = tools/world_tracker.py WTParams defaults
  double accel_sigma = 6.0;
  double gate_chi2 = 13.8, gate_base_m = 1.2, gate_speed = 5.0, gate_max_m = 3.0;
  double sigma_radial = 0.55, sigma_tangential = 0.25, meas_sigma = 0.35;
  double consistency_chi2 = 9.21, merge_chi2 = 5.99;
  int merge_frames = 15, confirm_hits = 8, confirm_window = 12, max_coast = 60, tentative_coast = 3;
  double team_penalty = 4.0, team_penalty_p = 0.8, team_firm_share = 0.85, team_share = 0.75, team_votes = 3.0;
  int team_firm_votes = 10;
  int stitch_gap = 90, stitch_gap_team = 240;
  double stitch_d0 = 1.0, stitch_vmax = 5.0, height_tol = 0.12;
  int height_min_n = 5;
  int num_min_reads = 2;
  double num_share = 0.6, num_penalty = 6.0, num_penalty_score = 0.8;
  int num_switch_reads = 3, num_switch_window = 90;
  double split_share = 0.66;
  int id_gap = 300;
  double id_merge_dist = 2.0;
  int roster_min_tracks = 2;
  double app_margin = 0.05;
  int app_recent = 90, app_min_samples = 3, proto_min = 20;
  int id_window = 300;
  double id_tau = 150.0;
};

/* Scale the rules that count detections in frame windows when detections arrive every `stride` frames. */
Params for_stride(Params p, int stride);

struct DetKey {
  int frame = 0;
  int cam = 0;                           // camera index
  int64_t gid = 0;                       // MV3DT object id (per camera)
  bool operator<(const DetKey &o) const {
    return frame != o.frame ? frame < o.frame : cam != o.cam ? cam < o.cam : gid < o.gid;
  }
  bool operator==(const DetKey &o) const { return frame == o.frame && cam == o.cam && gid == o.gid; }
};

/* One detection of one camera in one frame. */
struct Detection {
  double x = 0, y = 0;                   // floor position (m), camera bias already removed
  std::string team;                      // confident team label ("" = none)
  double team_p = 0;                     // its probability
  double height = NAN;                   // body height (m) if known
  int64_t gid = 0;
  bool has_read = false;                 // jersey read of this detection (if any)
  std::string number;
  double read_score = 0;
  bool has_feat = false;                 // colour feature (good crops only)
  std::array<float, FEAT_DIM> feat{};
};

struct CameraFrame {
  int cam = 0;
  std::vector<Detection> dets;
};

struct OutPoint {
  int frame;
  double x, y;
  int n_cams;
};

struct Track;

/* Identity reported for one detection (empty = no identity yet). */
struct DetIdentity {
  DetKey key;
  std::string identity;                  // "team:number" or ""
  int tid = -1;                          // court track the detection was assigned to
};

/* A live court track (confirmed, not yet given up), for the per-frame output. */
struct LiveTrack {
  int tid;
  double x, y, vx, vy;                   // Kalman state: floor position (m) and velocity (m/s)
  bool measured;                         // detections this frame (false: predicted while unseen)
  int n_cams;                            // cameras that saw it this frame
  std::string identity;                  // "team:number" or ""
  std::string team;                      // firm team label or ""
  std::vector<DetKey> dets;              // this frame's detections
};

class Tracker {
 public:
  Tracker(const Params &p, const std::vector<std::pair<double, double>> &cam_xy);
  ~Tracker();
  /* Process one frame: cameras in arrival order, each with its detections. Returns the identity reported for each
   * detection assigned to a track in this frame. */
  std::vector<DetIdentity> step(int frame, const std::vector<CameraFrame> &cams);
  /* Confirmed tracks (finished and live), for analysis: id, positions, detection keys. */
  struct TrackView {
    int tid;
    const std::vector<OutPoint> *out;
    const std::vector<DetKey> *dets;
  };
  std::vector<TrackView> result() const;
  /* Confirmed live tracks after the last step(). */
  std::vector<LiveTrack> live() const;
  /* Roster inferred so far: number -> team -> tracks that established it. */
  std::vector<std::pair<std::string, std::vector<std::pair<std::string, int>>>> roster() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/* Minimum-cost assignment of a dense rows x cols cost matrix (shortest augmenting path, as SciPy's
 * linear_sum_assignment). Returns (row, col) pairs sorted by row. */
std::vector<std::pair<int, int>> linear_sum_assignment(const std::vector<double> &cost, int rows, int cols);

}  // namespace courttrack

#endif
