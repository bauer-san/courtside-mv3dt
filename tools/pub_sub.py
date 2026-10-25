"""Write a hand-made MV3DT pub/sub (camera neighbour) config.

Same format as NVIDIA's utils/generate_pub_sub_configs.py output (pubBrokerTopicStr per camera, in the
auto-configurator's sorted video order = stream order; subPeerBrokerTopicStrs = topics each camera
subscribes to). Court coverage (a whole standing person in frame): NL-FL 14%, NR-FR 20%, FL-FR 4% (centre),
all other pairs 0%, so cross-half links only create wrong associations.

Topologies
  all            every camera subscribes to the other three (what the auto-configurator generates for 4 cams)
  pairs          NL<->FL (left half), NR<->FR (right half)
  pairs_centre   pairs + FL<->FR (the two end-wall cameras overlap around centre court)

  python -m tools.pub_sub pairs <out.yml>
"""

import argparse
from pathlib import Path

import yaml

from tools import CAMERAS

BROKER = "127.0.0.1:1883"
TOPIC = "/trck/cam%d"  # deployment_config.yml topic_template; camera ids 1..N in stream order
POS_ID = {pos: i + 1 for i, (_, pos, _) in enumerate(CAMERAS)}

TOPOLOGIES = {
    "all": {p: [q for q in POS_ID if q != p] for p in POS_ID},
    "pairs": {"NL": ["FL"], "FL": ["NL"], "NR": ["FR"], "FR": ["NR"]},
    "pairs_centre": {"NL": ["FL"], "FL": ["NL", "FR"], "NR": ["FR"], "FR": ["NR", "FL"]},
}


def config(topology: str) -> dict:
    subs = TOPOLOGIES[topology]
    order = sorted(POS_ID, key=POS_ID.get)
    return {
        "pubBrokerTopicStr": [f"{BROKER};{TOPIC % POS_ID[p]}" for p in order],
        "subPeerBrokerTopicStrs": [[f"{BROKER};{TOPIC % POS_ID[q]}" for q in subs[p]] for p in order],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("topology", choices=sorted(TOPOLOGIES))
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    args.out.write_text(yaml.dump(config(args.topology), default_flow_style=False))


if __name__ == "__main__":
    main()
