"""Check the online team model's labels crop by crop against the hand-named offline model (report section 4.6).

For each run, the court tracker runs in its main configuration (world_tracker.track_default) and every good crop's
online label (A, B or none) is compared with the offline model's label (red, maroon, ref, other; a label needs
p >= WTParams.team_min_p, as in the tracker), over the game (to 6:29, gt_eval.game_end; times in the original NL frames). Online team
names are matched to colours by majority. Reported per period: crops of each colour given the other team's name,
referee crops given a team name, and team names given to crops the offline model left unlabelled.

  python -m tools.team_label_check s_int8t10_r1 s_int8t10_r2 s_int8t10_r3
"""

import argparse
import collections

from tools import GAME_END, ROOT

PERIODS = ([(0, 900, "0:00-0:30"), (900, 1800, "0:30-1:00")]
           + [(1800 * m, 1800 * (m + 1), f"{m}:00-{m + 1}:00") for m in range(1, 6)] + [(10800, GAME_END + 1, "6:00-6:29")])


def check(run_name: str) -> list[str]:
    from tools.appearance import team_probs
    from tools.world_tracker import WTParams, track_default
    run = ROOT / "experiments" / run_name
    st = {}
    track_default(run, stats=st)
    given = st["given"]
    p_min = WTParams().team_min_p
    offline = {}
    for k, (probs, _) in team_probs(run).items():
        best = max(probs, key=probs.get)
        offline[k] = best if best != "other" and probs[best] >= p_min else None
    from tools.gt_eval import frame_offsets
    nl = frame_offsets(run).get("NL", 0)   # time-aligned clips start later: report in the original NL frames
    given = {(k[0] + nl, k[1], k[2]): v for k, v in given.items()}
    offline = {(k[0] + nl, k[1], k[2]): v for k, v in offline.items()}
    keys = [k for k in given if k[0] <= GAME_END and k in offline]
    votes = collections.Counter((given[k], offline[k]) for k in keys if given[k] and offline[k] in ("red", "maroon"))
    colour = {}
    for name in ("A", "B"):
        colour[name] = max(("red", "maroon"), key=lambda c: votes[(name, c)])
    out = [f"{run_name}: online names A = {colour['A']}, B = {colour['B']}; {len(keys)} good crops over the game"]
    for f0, f1, label in PERIODS:
        ks = [k for k in keys if f0 <= k[0] < f1]
        parts = []
        for c in ("red", "maroon"):
            n = [k for k in ks if offline[k] == c]
            wrong = sum(1 for k in n if given[k] and colour[given[k]] != c)
            parts.append(f"{c} {wrong}/{len(n)} other team")
        refs = [k for k in ks if offline[k] == "ref"]
        parts.append(f"ref {sum(1 for k in refs if given[k])}/{len(refs)} team-labelled")
        unl = [k for k in ks if offline[k] is None]
        parts.append(f"offline-unlabelled given A {sum(1 for k in unl if given[k] == 'A')}, "
                     f"B {sum(1 for k in unl if given[k] == 'B')} (of {len(unl)})")
        out.append(f"  {label}: " + "; ".join(parts))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(len(a.runs)) as ex:
        for lines in ex.map(check, a.runs):
            print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
