"""Hand-made pub/sub topologies match NVIDIA's generated format."""

import yaml

from tools import ROOT
from tools.pub_sub import config


def test_all_matches_generated_config():
    generated = ROOT / "experiments" / "c60_pnt" / "pub_sub_info_config_0.yml"
    ours = config("all")
    if generated.exists():
        g = yaml.safe_load(generated.read_text())
        assert g["pubBrokerTopicStr"] == ours["pubBrokerTopicStr"]
        assert [sorted(s) for s in g["subPeerBrokerTopicStrs"]] == [sorted(s) for s in ours["subPeerBrokerTopicStrs"]]


def test_pairs():
    c = config("pairs")
    # stream order NL(1), FL(2), NR(3), FR(4)
    assert c["subPeerBrokerTopicStrs"] == [["127.0.0.1:1883;/trck/cam2"], ["127.0.0.1:1883;/trck/cam1"],
                                           ["127.0.0.1:1883;/trck/cam4"], ["127.0.0.1:1883;/trck/cam3"]]


def test_pairs_centre_is_symmetric():
    c = config("pairs_centre")
    subs = {i + 1: {int(t.rsplit("cam", 1)[1]) for t in s} for i, s in enumerate(c["subPeerBrokerTopicStrs"])}
    assert all(a in subs[b] for a, s in subs.items() for b in s)
    assert subs[2] == {1, 4} and subs[4] == {3, 2}
