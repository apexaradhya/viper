"""Synthetic near-infrared arm images with KNOWN vein locations.

AI-scaffolded test data. It lets you (1) develop without a camera and (2) measure accuracy against ground truth.
It is a simplified model, not real anatomy. Say so in your video and README.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SyntheticArm:
    image: np.ndarray        # uint8 grayscale, veins darker than skin
    vein_centerlines: np.ndarray  # bool mask of true vein center pixels
    wrist: tuple[float, float]
    elbow: tuple[float, float]


def make_arm(h: int = 480, w: int = 640, seed: int = 0, noise: float = 5.0, contrast: float = 1.0) -> SyntheticArm:
    """noise = std of sensor noise; contrast scales how dark the veins are (1.0 = default, 0.5 = faint veins)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Skin: brighter in the middle of the arm, darker toward the edges, plus slow lighting variation.
    base = 175 - 55 * ((yy - h / 2) / (h / 2)) ** 2 - 25 * ((xx - w / 2) / (w / 2)) ** 2
    base += 8 * np.sin(xx / 90.0 + rng.uniform(0, 6))

    veins = np.zeros((h, w), np.float32)
    truth = np.zeros((h, w), np.uint8)
    specs = [  # (y offset from center, amplitude, wavelength, thickness, darkness)
        (-70, 18, 260, 6, 55),
        (10, 14, 300, 5, 45),
        (80, 20, 220, 4, 38),
    ]
    xs = np.arange(40, w - 40)
    for off, amp, wl, thick, dark in specs:
        phase = rng.uniform(0, 6.28)
        ys = h / 2 + off + amp * np.sin(xs / wl * 2 * np.pi + phase)
        pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
        layer = np.zeros((h, w), np.float32)
        cv2.polylines(layer, [pts], False, float(dark), thickness=thick, lineType=cv2.LINE_AA)
        veins = np.maximum(veins, layer)
        cv2.polylines(truth, [pts], False, 1, thickness=1)
    # A short branch joining the middle vein, so branch points exist in the data.
    bx = np.arange(300, 420)
    by = h / 2 + 10 + 14 * np.sin(bx / 300 * 2 * np.pi) - (bx - 300) * 0.45
    bpts = np.stack([bx, by], axis=1).astype(np.int32).reshape(-1, 1, 2)
    layer = np.zeros((h, w), np.float32)
    cv2.polylines(layer, [bpts], False, 35.0, thickness=3, lineType=cv2.LINE_AA)
    veins = np.maximum(veins, layer)
    cv2.polylines(truth, [bpts], False, 1, thickness=1)

    veins = cv2.GaussianBlur(veins, (0, 0), 1.6)
    img = base - contrast * veins + rng.normal(0, noise, (h, w))
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return SyntheticArm(img, truth.astype(bool), wrist=(40.0, h / 2), elbow=(w - 40.0, h / 2))
