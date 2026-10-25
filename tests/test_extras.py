"""Edge cases and end-to-end checks for the reference implementation."""
from datetime import datetime, timedelta

import numpy as np

from snake import learn, pipeline, rotation, sites, synthetic, vessels
from snake.config import DEFAULT


def test_empty_inputs_do_not_crash():
    blank = np.full((120, 160), 128, np.uint8)
    r = pipeline.run(blank)
    assert not r.skel.any() and r.candidates == [] and sites.pick_top(r.candidates) == []


def test_sites_respect_border_and_branch_limits():
    arm = synthetic.make_arm(seed=3)
    r = pipeline.run(arm.image)
    for s in r.candidates:
        assert s.border_px >= DEFAULT.min_border_px and s.branch_px >= DEFAULT.min_branch_px and s.straight_px >= DEFAULT.min_straight_px


def test_pipeline_is_deterministic():
    a = pipeline.run(synthetic.make_arm(seed=5).image)
    b = pipeline.run(synthetic.make_arm(seed=5).image)
    assert (a.skel == b.skel).all() and len(a.candidates) == len(b.candidates)


def test_accuracy_regression():
    from scipy.ndimage import distance_transform_edt as edt
    for seed in range(5):
        arm = synthetic.make_arm(seed=seed)
        r = pipeline.run(arm.image)
        assert (edt(~arm.vein_centerlines)[r.skel] <= 4).mean() > 0.95
        assert (edt(~r.skel)[arm.vein_centerlines] <= 4).mean() > 0.9


def test_rotation_empty_log_leaves_order(tmp_path):
    frame = rotation.ArmFrame((0, 0), (100, 0))
    ss = [sites.Site(10, 0, 0.9), sites.Site(50, 0, 0.5)]
    out = rotation.rank_with_rotation(ss, rotation.SiteLog(tmp_path / "l.json"), frame)
    assert [s.x for s in out] == [10, 50]


def test_rotation_use_fades_with_age(tmp_path):
    frame = rotation.ArmFrame((0, 0), (100, 0))
    now = datetime(2026, 10, 25, 12)
    log = rotation.SiteLog(tmp_path / "l.json")
    log.add(0.3, 0.0, when=now - timedelta(days=1))
    p_recent, _ = rotation.rotation_penalty(0.3, 0.0, log, DEFAULT, now)
    log2 = rotation.SiteLog(tmp_path / "l2.json")
    log2.add(0.3, 0.0, when=now - timedelta(days=6))
    p_old, _ = rotation.rotation_penalty(0.3, 0.0, log2, DEFAULT, now)
    assert p_recent > p_old > 0


def test_learning_handles_one_class_and_empty_feedback():
    X = np.random.default_rng(0).random((20, len(learn.FEATURES)))
    m = learn.train(X, np.ones(20, int))
    assert ((learn.predict(m, X) >= 0) & (learn.predict(m, X) <= 1)).all()
    assert learn.update_from_feedback(m, np.zeros((0, len(learn.FEATURES))), np.zeros(0, int)) is m


def test_feedback_moves_the_model():
    rng = np.random.default_rng(1)
    X = rng.random((200, len(learn.FEATURES)))
    y = (X[:, 2] > 0.5).astype(int)           # pretrain: straightness matters
    m = learn.train(X, y)
    probe = np.full((1, len(learn.FEATURES)), 0.5)
    probe[0, 2] = 0.9
    before = learn.predict(m, probe)[0]
    Xf = np.tile(probe, (10, 1))
    m2 = learn.update_from_feedback(m, Xf, np.zeros(10, int))   # user rejects such sites
    assert learn.predict(m2, probe)[0] < before
    assert m.n_seen == 0 and m2.n_seen == 10


def test_feedback_store_roundtrip(tmp_path):
    st = learn.FeedbackStore(tmp_path / "f.json")
    st.add(np.ones(len(learn.FEATURES)), True)
    X, y = learn.FeedbackStore(tmp_path / "f.json").arrays()
    assert X.shape == (1, len(learn.FEATURES)) and y[0] == 1
