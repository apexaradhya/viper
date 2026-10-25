"""Tests for AI-scaffolded parts. These should pass from the start."""
from datetime import datetime, timedelta

import numpy as np

from snake import preprocess, synthetic
from snake.rotation import SiteLog


def test_synthetic_arm_shape_and_truth():
    arm = synthetic.make_arm(seed=1)
    assert arm.image.dtype == np.uint8
    assert arm.image.shape == (480, 640)
    assert arm.vein_centerlines.sum() > 500
    # veins are darker than the surrounding skin on average
    assert arm.image[arm.vein_centerlines].mean() < arm.image.mean()


def test_resize_max_never_upscales():
    img = np.zeros((100, 200), np.uint8)
    assert preprocess.resize_max(img, 640).shape == (100, 200)
    big = np.zeros((1000, 2000), np.uint8)
    assert max(preprocess.resize_max(big, 640).shape) == 640


def test_site_log_roundtrip_and_recent(tmp_path):
    log = SiteLog(tmp_path / "log.json")
    now = datetime(2026, 10, 25, 12, 0)
    log.add(0.4, 0.05, when=now - timedelta(days=2))
    log.add(0.7, -0.1, when=now - timedelta(days=20))
    reloaded = SiteLog(tmp_path / "log.json")
    assert len(reloaded.entries) == 2
    recent = reloaded.recent(7, now=now)
    assert len(recent) == 1 and recent[0]["t"] == 0.4
