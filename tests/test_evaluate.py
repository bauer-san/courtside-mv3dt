"""Evaluator metrics on a synthetic run with known answers."""

import gzip
import json

import numpy as np
import pytest

from tools.evaluate import inside, metrics


def msg(frame, sensor, objs):
    return {"id": str(frame), "sensorId": sensor, "objects": [
        {"id": gid, "info": {"footLocation3D": f"{x},{y}"}} for gid, (x, y) in objs.items()]}


@pytest.fixture
def run(tmp_path):
    lines = []
    for f in range(1, 61):  # 2 s
        # ID 1: seen by NL and FL, 0.2 m apart, inside court; born near the baseline (not interior)
        lines.append(msg(f, "Camera1", {"1": (-10.0, 0.0)}))
        objs = {"1": (-10.0, 0.2)}
        # ID 2 appears at frame 31 mid-court (interior birth); ID 3 is a spectator outside the court
        if f > 30:
            objs["2"] = (0.0, 0.0)
        objs["3"] = (0.0, 9.0)
        lines.append(msg(f, "Camera2", objs))
        # ID 4 from FR sits 0.1 m from ID 2: a missed cross-camera merge
        if f > 30:
            lines.append(msg(f, "Camera4", {"4": (0.1, 0.0)}))
    with gzip.open(tmp_path / "kafka.jsonl.gz", "wt") as fh:
        fh.write("\n".join(json.dumps(m) for m in lines) + "\n")
    return tmp_path


def test_inside():
    xy = np.array([[0, 0], [0, 7.5], [11.5, 0]])
    assert inside(xy).tolist() == [True, False, False]
    assert inside(xy, 1.0).tolist() == [True, True, True]


def test_metrics(run):
    m = metrics(run)
    assert m["frames"] == 60
    assert m["ids"] == 4
    assert m["multi_cam_ids"] == 1
    assert m["interior_births_per_min"] == pytest.approx(2 / (2 / 60))  # IDs 2 and 4, over 2 s
    assert m["xcam_med_m"] == pytest.approx(0.2)
    assert m["xcam_NL-FL"] == pytest.approx(0.2)
    assert m["missed_merge_rate"] == pytest.approx(30 / 60)
    # points: 60 (ID1 NL) + 60 (ID1 FL) + 60 (ID3 out) + 30 (ID2) + 30 (ID4) = 240; ID3 outside
    assert m["inside_court"] == pytest.approx(180 / 240)
