"""Acceptance tests for the pipeline. Run against the reference implementation (snake/); overrides can be tested the same way.

Run:  pytest -q
"""
from datetime import datetime, timedelta

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from snake import preprocess, rotation, sites, synthetic, vessels
from snake.config import DEFAULT


def implemented(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


@pytest.fixture(scope="module")
def arm():
    return synthetic.make_arm(seed=0)


def test_enhance_keeps_shape_and_dtype(arm):
    out = implemented(preprocess.enhance, arm.image)
    assert out.shape == arm.image.shape and out.dtype == np.uint8


def test_enhance_increases_vein_contrast(arm):
    out = implemented(preprocess.enhance, arm.image)
    near = distance_transform_edt(~arm.vein_centerlines) <= 3
    far = distance_transform_edt(~arm.vein_centerlines) >= 12
    before = arm.image[far].mean() - arm.image[near].mean()
    after = out[far].mean() - out[near].mean()
    assert after > before * 0.9  # contrast should not get worse


def test_vesselness_range_and_vein_response(arm):
    enh = implemented(preprocess.enhance, arm.image)
    v = implemented(vessels.vesselness, enh)
    assert v.shape == arm.image.shape
    assert 0.0 <= float(v.min()) and float(v.max()) <= 1.0 + 1e-6
    on = v[arm.vein_centerlines].mean()
    off = v[distance_transform_edt(~arm.vein_centerlines) >= 15].mean()
    assert on > 3 * off


def test_mask_and_skeleton_follow_true_veins(arm):
    enh = implemented(preprocess.enhance, arm.image)
    v = implemented(vessels.vesselness, enh)
    mask = implemented(vessels.vein_mask, v)
    skel = implemented(vessels.skeleton, mask)
    assert skel.dtype == bool and skel.sum() > 300
    d = distance_transform_edt(~arm.vein_centerlines)
    assert (d[skel] <= 4).mean() > 0.85  # at least 85% of skeleton within 4 px of a true vein


def test_top_sites_are_on_veins_and_separated(arm):
    enh = implemented(preprocess.enhance, arm.image)
    v = implemented(vessels.vesselness, enh)
    skel = implemented(vessels.skeleton, implemented(vessels.vein_mask, v))
    cands = implemented(sites.score_candidates, skel, v)
    top = implemented(sites.pick_top, cands)
    assert 1 <= len(top) <= DEFAULT.n_sites
    d = distance_transform_edt(~arm.vein_centerlines)
    for s in top:
        assert d[int(round(s.y)), int(round(s.x))] <= 5
        assert 0.0 <= s.score <= 1.0
    for i, a in enumerate(top):
        for b in top[i + 1:]:
            assert np.hypot(a.x - b.x, a.y - b.y) >= DEFAULT.min_site_separation_px - 1e-6
    assert [s.score for s in top] == sorted((s.score for s in top), reverse=True)


def test_arm_coordinates():
    frame = rotation.ArmFrame(wrist=(40.0, 240.0), elbow=(600.0, 240.0))
    t0, s0 = implemented(rotation.to_arm_coords, 40.0, 240.0, frame)
    t1, s1 = implemented(rotation.to_arm_coords, 600.0, 240.0, frame)
    assert (round(t0, 6), round(s0, 6)) == (0.0, 0.0)
    assert (round(t1, 6), round(s1, 6)) == (1.0, 0.0)
    tm, sm = rotation.to_arm_coords(320.0, 240.0 + 56.0, frame)
    assert abs(tm - 0.5) < 1e-6 and abs(abs(sm) - 0.1) < 1e-6


def test_rotation_pushes_recent_site_down(tmp_path):
    frame = rotation.ArmFrame(wrist=(0.0, 0.0), elbow=(100.0, 0.0))
    a = sites.Site(x=30.0, y=0.0, score=0.9)   # best raw site
    b = sites.Site(x=70.0, y=0.0, score=0.8)
    log = rotation.SiteLog(tmp_path / "log.json")
    now = datetime(2026, 10, 25, 12, 0)
    log.add(0.30, 0.0, when=now - timedelta(days=1))  # site A was used yesterday
    out = implemented(rotation.rank_with_rotation, [a, b], log, frame, DEFAULT, now)
    assert out[0].x == 70.0                    # B now ranks first
    assert (a.score, b.score) == (0.9, 0.8)    # inputs not mutated
    assert out[1].note != ""                   # explains why A was pushed down


def test_learn_pretrain_beats_chance():
    import pytest
    from snake import learn
    from snake.config import DEFAULT as p
    try:
        Xtr, ytr = learn.make_training_set(15, p, seed0=0)
        Xte, yte = learn.make_training_set(5, p, seed0=100)
        m = learn.train(Xtr, ytr)
        pr = learn.predict(m, Xte)
    except NotImplementedError:
        pytest.skip("learn.py not implemented yet")
    import numpy as np
    assert pr.shape == (len(yte),) and ((pr >= 0) & (pr <= 1)).all()
    acc = ((pr > 0.5) == (yte == 1)).mean()
    base = max(yte.mean(), 1 - yte.mean())
    assert acc >= base  # at least as good as always guessing the majority class on held-out seeds
