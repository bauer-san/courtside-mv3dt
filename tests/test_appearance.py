"""Appearance cues: geometric height estimate and torso histogram."""

import numpy as np
import yaml

from tools import DATASETS
from tools.appearance import person_height, torso_hist
from tools.court import project


def test_person_height_recovers_known_height():
    P = np.array(yaml.safe_load((DATASETS / "courtside" / "camInfo" / "cam_03_FR.yml").read_text())
                 ["projectionMatrix_3x4_w2p"]).reshape(3, 4)
    for x, y, h in [(4.0, 1.0, 1.85), (7.0, -2.5, 1.70), (2.0, 3.0, 2.00)]:
        px, _ = project(P, np.array([[x, y, 0.0], [x, y, h]]))
        (uf, vf), (_, vt) = px
        est = person_height(P, uf - 20, vt, uf + 20, vf)
        assert abs(est - h) < 0.02


def test_torso_hist_normalised_and_colour_sensitive():
    red = np.zeros((200, 80, 3), np.uint8)
    red[:] = (30, 30, 200)
    blue = red[..., ::-1].copy()
    hr, hb = torso_hist(red, 0, 0, 80, 200), torso_hist(blue, 0, 0, 80, 200)
    assert abs(hr.sum() - 1) < 1e-6 and abs(hb.sum() - 1) < 1e-6
    assert np.minimum(hr, hb).sum() < 0.1          # red and blue torsos share almost no histogram mass
    assert torso_hist(red, 0, 0, 10, 40) is None   # too small
