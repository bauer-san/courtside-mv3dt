/* Gym MV3DT online team segmentation: C++ port of tools/online_teams.py.
 * Arithmetic follows the Python version where it decides outcomes: float32 colour samples and camera means, the
 * order of tracks (it decides which point the random draws pick), stable sorts, first-minimum ties and NumPy's
 * percentile interpolation.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "online_teams.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace courttrack {

namespace {

using VecD = std::vector<double>;

double sqdist(const Hist &x, const VecD &c) {
  double s = 0;
  for (int i = 0; i < FEAT_DIM; ++i) {
    double e = double(x[i]) - c[i];
    s += e * e;
  }
  return s;
}

VecD to_vec(const Hist &h) { return VecD(h.begin(), h.end()); }

/* numpy.percentile, method "linear" */
double percentile(std::vector<double> a, double q) {
  std::sort(a.begin(), a.end());
  double idx = (a.size() - 1) * (q / 100.0);   // as NumPy 2.5: (n - 1) * quantile
  size_t lo = (size_t)std::floor(idx), hi = std::min(lo + 1, a.size() - 1);
  double t = idx - lo, d = a[hi] - a[lo];
  return t >= 0.5 ? a[hi] - d * (1 - t) : a[lo] + d * t;
}

struct KMeansResult {
  std::vector<VecD> c;
  std::vector<int> lab;
  double inertia;
};

KMeansResult lloyd(const std::vector<Hist> &X, const VecD &w, std::vector<VecD> c, int iters) {
  size_t n = X.size(), k = c.size();
  std::vector<int> lab(n);
  auto assign = [&]() {
    for (size_t i = 0; i < n; ++i) {
      int b = 0;
      double bd = std::numeric_limits<double>::infinity();
      for (size_t j = 0; j < k; ++j) {
        double d = sqdist(X[i], c[j]);
        if (d < bd) {
          bd = d;
          b = (int)j;
        }
      }
      lab[i] = b;
    }
  };
  for (int it = 0; it < iters; ++it) {
    assign();
    std::vector<VecD> nc(k, VecD(FEAT_DIM, 0.0));
    VecD ws(k, 0.0);
    std::vector<bool> any(k, false);
    for (size_t i = 0; i < n; ++i) {
      for (int d = 0; d < FEAT_DIM; ++d) nc[lab[i]][d] += double(X[i][d]) * w[i];
      ws[lab[i]] += w[i];
      any[lab[i]] = true;
    }
    for (size_t j = 0; j < k; ++j) {
      if (!any[j]) nc[j] = c[j];
      else
        for (int d = 0; d < FEAT_DIM; ++d) nc[j][d] /= ws[j];
    }
    if (nc == c) break;
    c = nc;
  }
  assign();
  double inertia = 0;
  for (size_t i = 0; i < n; ++i) inertia += sqdist(X[i], c[lab[i]]) * w[i];
  return {c, lab, inertia};
}

/* weighted_kmeans: `attempts` weighted k-means++ starts plus, if given, a start from `init` */
KMeansResult weighted_kmeans(const std::vector<Hist> &X, const VecD &w, int k, SplitMix64 &rng,
                             const std::vector<VecD> &init, int attempts, int iters = 100) {
  std::vector<std::vector<VecD>> starts;
  for (int a = 0; a < attempts; ++a) {
    std::vector<VecD> c{to_vec(X[rng.choice(w)])};
    for (int j = 1; j < k; ++j) {
      VecD d(X.size());
      double tot = 0;
      for (size_t i = 0; i < X.size(); ++i) {
        double m = std::numeric_limits<double>::infinity();
        for (auto &cc : c) m = std::min(m, sqdist(X[i], cc));
        d[i] = m * w[i];
        tot += d[i];
      }
      c.push_back(to_vec(X[tot > 0 ? rng.choice(d) : rng.choice(w)]));
    }
    starts.push_back(c);
  }
  if (!init.empty()) starts.push_back(init);
  KMeansResult best;
  bool have = false;
  for (auto &c : starts) {
    KMeansResult r = lloyd(X, w, c, iters);
    if (!have || r.inertia < best.inertia) {
      best = r;
      have = true;
    }
  }
  return best;
}

}  // namespace

uint64_t SplitMix64::next() {
  s += 0x9E3779B97F4A7C15ULL;
  uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

double SplitMix64::uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }

int SplitMix64::choice(const std::vector<double> &p) {
  std::vector<double> c(p.size());
  std::partial_sum(p.begin(), p.end(), c.begin());
  double x = uniform() * c.back();
  size_t i = std::upper_bound(c.begin(), c.end(), x) - c.begin();   // searchsorted(side="right")
  return (int)std::min(i, p.size() - 1);
}

OnlineTeams::OnlineTeams(const OTParams &p) : p_(p), rng_(p.seed) {}

Hist OnlineTeams::sample(const Hist &hist) {
  Hist s;
  for (int i = 0; i < FEAT_DIM; ++i) s[i] = std::sqrt(hist[i]);
  return s;
}

Hist OnlineTeams::centred(int cam, const Hist &s) const {
  Hist g{};
  int total = 0;
  for (int c : cam_order_) {
    const Hist &cs = cam_sum_.at(c);
    for (int i = 0; i < FEAT_DIM; ++i) g[i] += cs[i];
    total += cam_n_.at(c);
  }
  float n = (float)std::max(total, 1), nc = (float)cam_n_.at(cam);
  const Hist &cs = cam_sum_.at(cam);
  Hist out;
  for (int i = 0; i < FEAT_DIM; ++i) out[i] = (s[i] - cs[i] / nc) + g[i] / n;
  return out;
}

std::pair<std::string, double> OnlineTeams::label(int cam, const Hist &s) const {
  auto it = cam_n_.find(cam);
  if (centres_.empty() || it == cam_n_.end() || it->second == 0) return {"", 0.0};
  Hist x = centred(cam, s);
  size_t k = centres_.size();
  VecD d(k);
  for (size_t j = 0; j < k; ++j) d[j] = sqdist(x, centres_[j]);
  size_t b = std::min_element(d.begin(), d.end()) - d.begin();
  VecD w(k);
  double sw = 0;
  for (size_t j = 0; j < k; ++j) sw += (w[j] = std::exp(-(d[j] - d[b]) / p_.temperature));
  for (auto &x2 : w) x2 /= sw;
  if (std::sqrt(d[b]) > radius_[b]) return {"", 0.0};
  const std::string &r = roles_[b];
  double pr = 0;
  for (size_t j = 0; j < k; ++j)
    if (roles_[j] == r) pr += w[j];
  if (r == UNKNOWN_TEAM || pr < p_.label_p) return {"", 0.0};
  return {r, pr};
}

bool OnlineTeams::feature(int cam, const Hist &s, Hist *out) const {
  auto it = cam_n_.find(cam);
  if (it == cam_n_.end() || it->second == 0) return false;
  Hist f = centred(cam, s);
  double nn = 0;   // float64, summed in order (as the Python reference)
  for (float v : f) nn += double(v) * v;
  float norm = (float)(std::sqrt(nn) + 1e-9);
  for (int i = 0; i < FEAT_DIM; ++i) (*out)[i] = f[i] / norm;
  return true;
}

void OnlineTeams::observe_crop(int cam, const Hist &s) {
  auto it = cam_sum_.find(cam);
  if (it == cam_sum_.end()) {
    cam_order_.push_back(cam);
    cam_sum_[cam] = s;
  } else {
    for (int i = 0; i < FEAT_DIM; ++i) it->second[i] += s[i];
  }
  ++cam_n_[cam];
}

void OnlineTeams::observe_track(int tid, int frame, int cam, const Hist *s, bool has_read) {
  dets_[tid].emplace_back(frame, has_read);
  if (s) {
    auto it = samples_.find(tid);
    if (it == samples_.end()) {
      tid_order_.push_back(tid);
      it = samples_.emplace(tid, std::vector<Sample>{}).first;
    }
    it->second.push_back({frame, cam, *s});
  }
}

void OnlineTeams::update(int frame) {
  if (frame < p_.warmup || frame % p_.every) return;
  int lo = frame - p_.window;
  for (auto &kv : samples_)   // history older than the window is never used again (frames are increasing)
    kv.second.erase(kv.second.begin(), std::find_if(kv.second.begin(), kv.second.end(),
                                                    [&](const Sample &x) { return x.frame > lo; }));
  for (auto &kv : dets_)
    kv.second.erase(kv.second.begin(), std::find_if(kv.second.begin(), kv.second.end(),
                                                    [&](const std::pair<int, bool> &x) { return x.first > lo; }));
  std::vector<Hist> X;
  VecD w;
  std::vector<std::pair<double, double>> rates;
  for (int tid : tid_order_) {
    const auto &ss = samples_[tid];
    if (ss.empty()) continue;
    int from = std::max(lo, ss.back().frame - p_.track_recent);
    Hist m{};
    int n = 0;
    for (auto &x : ss)
      if (x.frame > from) {
        Hist c = centred(x.cam, x.s);
        for (int i = 0; i < FEAT_DIM; ++i) m[i] += c[i];
        ++n;
      }
    if (n < p_.min_samples) continue;
    if (const char *dump = getenv("CT_PROTOS")) {   // debugging: clustering inputs
      static FILE *f = std::fopen(dump, "w");
      int r = 0, nd = 0;
      for (auto &d : dets_[tid]) r += d.second, ++nd;
      std::fprintf(f, "%d %d %d %d %d %.9g\n", frame, tid, n, r, nd, m[0]);
      std::fflush(f);
    }
    for (auto &v : m) v /= (float)n;
    X.push_back(m);
    w.push_back(n);
    double r = 0, nd = 0;
    for (auto &d : dets_[tid]) {
      r += d.second;
      nd += 1;
    }
    rates.emplace_back(r, nd);
  }
  if ((int)X.size() < std::max(p_.min_tracks, p_.k)) return;
  int attempts = p_.warm_only && !centres_.empty() ? 0 : 10;
  KMeansResult km = weighted_kmeans(X, w, p_.k, rng_, centres_, attempts);
  int k = p_.k;
  VecD share(k, 0), reads(k, 0), dets(k, 0), rate(k);
  double wt = 0;
  for (size_t i = 0; i < X.size(); ++i) {
    share[km.lab[i]] += w[i];
    wt += w[i];
    reads[km.lab[i]] += rates[i].first;
    dets[km.lab[i]] += rates[i].second;
  }
  for (int j = 0; j < k; ++j) {
    share[j] /= wt;
    rate[j] = reads[j] / std::max(dets[j], 1.0);
  }
  std::vector<int> order(k);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return -rate[a] < -rate[b]; });
  std::vector<int> cand;
  for (int j : order)
    if (share[j] >= p_.min_team_share) cand.push_back(j);
  if (cand.size() < 2) return;
  int ta = cand[0], tb = cand[1];
  std::vector<std::string> roles(k, UNKNOWN_TEAM);
  int ref = -1;
  for (int j = 0; j < k; ++j)
    if (j != ta && j != tb && (ref < 0 || rate[j] < rate[ref])) ref = j;
  if (ref >= 0) roles[ref] = p_.ref_label;
  if (!prev_a_.empty()) {
    auto sd = [](const VecD &a, const VecD &b) {
      double s = 0;
      for (size_t i = 0; i < a.size(); ++i) s += (a[i] - b[i]) * (a[i] - b[i]);
      return s;
    };
    double d0 = sd(km.c[ta], prev_a_[0]) + sd(km.c[tb], prev_b_[0]);
    double d1 = sd(km.c[ta], prev_b_[0]) + sd(km.c[tb], prev_a_[0]);
    if (d1 < d0) std::swap(ta, tb);
  }
  roles[ta] = "A";
  roles[tb] = "B";
  prev_a_ = {km.c[ta]};
  prev_b_ = {km.c[tb]};
  std::vector<std::vector<double>> dist(k);
  for (int tid : tid_order_) {
    const auto &ss = samples_[tid];
    size_t from = ss.size() > (size_t)p_.min_samples * 3 ? ss.size() - p_.min_samples * 3 : 0;
    for (size_t i = from; i < ss.size(); ++i) {
      if (ss[i].frame <= lo) continue;
      Hist x = centred(ss[i].cam, ss[i].s);
      int b = 0;
      double bd = std::numeric_limits<double>::infinity();
      for (int j = 0; j < k; ++j) {
        double d = sqdist(x, km.c[j]);
        if (d < bd) {
          bd = d;
          b = j;
        }
      }
      dist[b].push_back(std::sqrt(bd));
    }
  }
  radius_.assign(k, std::numeric_limits<double>::infinity());
  for (int j = 0; j < k; ++j)
    if (!dist[j].empty()) radius_[j] = percentile(dist[j], p_.radius_q);
  centres_ = km.c;
  roles_ = roles;
  if (const char *dump = getenv("CT_MODEL")) {   // debugging: every clustering's model
    static FILE *f = std::fopen(dump, "w");
    std::fprintf(f, "%d %zu", frame, X.size());
    for (int j = 0; j < k; ++j) {
      std::fprintf(f, " %s %.17g", roles[j].c_str(), radius_[j]);
      for (int d = 0; d < 3; ++d) std::fprintf(f, " %.17g", km.c[j][d]);
    }
    std::fprintf(f, "\n");
    std::fflush(f);
  }
  std::ostringstream o;
  o << "frame " << frame << ", " << X.size() << " tracks:";
  for (int j = 0; j < k; ++j) o << " " << roles[j] << " " << share[j] << "/" << rate[j];
  summary_ = o.str();
}

}  // namespace courttrack
