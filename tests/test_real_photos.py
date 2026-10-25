"""Arm mask (real photos with a dark background) and the labeling store. No Kaggle images needed."""
import cv2
import numpy as np
from scipy import ndimage as ndi

from snake import labels, learn, pipeline, preprocess, rotation, synthetic, vessels
from snake.config import DEFAULT


def fake_photo() -> np.ndarray:
    """Bright vertical 'arm' on a black background with one dark vertical vein down its middle."""
    img = np.full((300, 400), 10, np.uint8)
    img[:, 120:280] = 160
    cv2.line(img, (200, 20), (200, 280), 140, 6)     # about as faint as real NIR veins
    rng = np.random.default_rng(0)
    return np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


def test_arm_mask_skipped_on_synthetic():
    for noise, contrast in [(5.0, 1.0), (14.0, 0.5)]:
        assert preprocess.arm_mask(synthetic.make_arm(seed=1, noise=noise, contrast=contrast).image) is None


def test_arm_mask_finds_the_arm():
    arm = preprocess.arm_mask(fake_photo())
    assert arm is not None
    assert arm[150, 200] and not arm[150, 50] and not arm[150, 350]


def test_sites_on_vein_not_on_arm_outline():
    r = pipeline.run(fake_photo())
    assert r.candidates, "the vein should produce candidates"
    d_arm = ndi.distance_transform_edt(r.arm)
    for s in r.candidates:
        assert abs(s.x - 200) <= 3                      # on the vein, not the outline at x=120 or x=280
        assert d_arm[int(s.y), int(s.x)] >= DEFAULT.min_border_px and s.border_px >= DEFAULT.min_border_px


def test_label_store_roundtrip_and_relabel(tmp_path):
    st = labels.LabelStore(tmp_path / "l.json")
    st.set("train", 4, 100, 50, True)
    st.set("train", 4, 101, 50, False)     # within 3 px of the first point: replaces it
    st.set("test", 61, 10, 10, True)
    again = labels.LabelStore(tmp_path / "l.json")
    assert again.get("train", 4, 100, 50) is False
    assert len(again.for_split("train")) == 1 and len(again.for_split("test")) == 1


def test_trace_path_follows_the_vein():
    r = pipeline.run(fake_photo())
    s = r.candidates[len(r.candidates) // 2]
    path = vessels.trace_path(r.skel, s.x, s.y, 40)
    ys, xs = np.nonzero(path)
    assert len(xs) > 40 and np.all(np.abs(xs - 200) <= 3)     # runs up and down the vein, stays on it
    assert not vessels.trace_path(r.skel, 10, 10).any()        # far from any vein: nothing traced


def test_frame_from_arm_puts_wrist_at_narrow_end():
    img = np.full((400, 300), 10, np.uint8)
    cv2.fillPoly(img, [np.array([[120, 20], [180, 20], [220, 380], [80, 380]])], 160)   # narrow at top
    f = rotation.frame_from_arm(preprocess.arm_mask(img))
    assert f.wrist[1] < 100 and f.elbow[1] > 300


def test_rotation_only_counts_the_same_arm(tmp_path):
    log = rotation.SiteLog(tmp_path / "l.json")
    log.add(0.5, 0.0, arm="Left")
    log.add(0.2, 0.0)                                           # old-style entry without an arm: counts for both
    assert len(log.recent(7, arm="Left")) == 2 and len(log.recent(7, arm="Right")) == 1
    pen_left, _ = rotation.rotation_penalty(0.5, 0.0, log, arm="Left")
    pen_right, _ = rotation.rotation_penalty(0.5, 0.0, log, arm="Right")
    assert pen_left > 0.9 and pen_right < 0.01


def test_features_for_labeled_points():
    arm = synthetic.make_arm(seed=2)
    r = pipeline.run(arm.image)
    s = r.candidates[len(r.candidates) // 2]
    X, found = labels.features_at(r, [(s.x, s.y), (0.0, 0.0)], DEFAULT)
    assert found == [True, False] and X.shape == (1, len(learn.FEATURES))


def test_shape_rejects_band_across_the_arm():
    """A dark band straight across the arm (like a wristband) must not produce sites; the vein along it still does."""
    img = fake_photo()
    cv2.line(img, (125, 150), (275, 150), 20, 6)            # very dark, across the arm
    r = pipeline.run(img)
    assert r.candidates and all(abs(s.x - 200) <= 3 for s in r.candidates)
    assert all(s.align >= DEFAULT.min_arm_alignment and s.contrast <= DEFAULT.max_contrast for s in r.candidates)


def test_memory_only_stores_write_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log, fb = rotation.SiteLog(None), learn.FeedbackStore(None)
    log.add(0.4, 0.0, arm="Left")
    fb.add(np.ones(len(learn.FEATURES)), True)
    assert len(log.recent(7)) == 1 and fb.arrays()[0].shape == (1, len(learn.FEATURES))
    log.clear(); fb.clear()
    assert not any(tmp_path.rglob("*"))                         # nothing written to disk


def test_clear_vein_check():
    from dataclasses import replace
    r = pipeline.run(synthetic.make_arm(seed=1).image)
    assert r.clear and r.candidates and r.vein_strength > DEFAULT.min_photo_vein_strength
    strict = pipeline.run(synthetic.make_arm(seed=1).image, replace(DEFAULT, min_photo_vein_strength=10.0))
    assert not strict.clear and strict.candidates == []       # too faint for the cutoff: no sites at all
    assert strict.skel.any()                                  # the vein picture is still there to show the user


def test_shape_features_measured():
    r = pipeline.run(fake_photo())
    mid = min(r.candidates, key=lambda s: abs(s.y - 150))
    assert mid.align > 0.95 and mid.piece_px > 100 and mid.arm_end < 0.2 and 0 < mid.contrast < 30
