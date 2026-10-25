"""Aradhya's own good/bad site labels on real Kaggle forearm photos, and turning them into training rows.

Labels are stored as (split, image number, x, y, good) in 640-px image coordinates, NOT as feature vectors. Why: if
detection changes later, training simply re-runs the pipeline and re-measures the features at each labeled point,
so his labeling work never goes stale.

The Kaggle images live only on this computer (data/kaggle_forearm_nir/, gitignored; license "Unknown", see
docs/DATA_SOURCES.md). The labels file holds only coordinates, so it can be committed.
Implemented by Claude Code.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from . import learn, pipeline, preprocess, sites
from .config import DEFAULT, Params

KAGGLE_DIR = Path("data/kaggle_forearm_nir")
LABELS_PATH = Path("data/labels/site_labels.json")
SPLITS = {"train": range(1, 61), "test": range(61, 71)}
SAME_POINT_PX = 3.0          # a label within this distance of another on the same image replaces it


def image_path(split: str, i: int) -> Path:
    return KAGGLE_DIR / split / "images" / f"{i:03d}.jpg"


def available(split: str) -> list[int]:
    return [i for i in SPLITS[split] if image_path(split, i).exists()]


def load_image(split: str, i: int, p: Params = DEFAULT) -> np.ndarray:
    """Grayscale, resized exactly like an uploaded image in the app (longest side p.max_side)."""
    return preprocess.resize_max(preprocess.load_gray(image_path(split, i).read_bytes()), p.max_side)


class LabelStore:
    """JSON list of {split, image, x, y, good, when}."""

    def __init__(self, path: str | Path = LABELS_PATH):
        self.path = Path(path)
        self.rows: list[dict] = json.loads(self.path.read_text()) if self.path.exists() else []

    def _find(self, split: str, image: int, x: float, y: float) -> int | None:
        for k, r in enumerate(self.rows):
            if r["split"] == split and r["image"] == image and np.hypot(r["x"] - x, r["y"] - y) <= SAME_POINT_PX:
                return k
        return None

    def get(self, split: str, image: int, x: float, y: float) -> bool | None:
        k = self._find(split, image, x, y)
        return None if k is None else bool(self.rows[k]["good"])

    def set(self, split: str, image: int, x: float, y: float, good: bool) -> None:
        row = {"split": split, "image": int(image), "x": round(float(x), 1), "y": round(float(y), 1),
               "good": bool(good), "when": datetime.now().isoformat(timespec="seconds")}
        k = self._find(split, image, x, y)
        if k is None:
            self.rows.append(row)
        else:
            self.rows[k] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.rows, indent=1))

    def for_split(self, split: str) -> list[dict]:
        return [r for r in self.rows if r["split"] == split]

    def images_done(self, split: str) -> set[int]:
        return {r["image"] for r in self.for_split(split)}


def sites_to_label(res: pipeline.Result, model: learn.SiteModel, p: Params = DEFAULT, k: int = 5) -> list[sites.Site]:
    """Up to 2k well-separated candidates: the top k by the hand-set rules plus the top k by the learned model.

    Why both: if only the rules' favourites were labeled, the test could only judge sites the rules already like,
    which would be unfair to the model. Sites are returned sorted top-to-bottom (not by score) and the page hides
    scores, so the labeler is not nudged by what VIPER thinks.
    """
    from dataclasses import replace
    many = replace(p, n_sites=k)
    pool = sites.pick_top(res.candidates, many) + sites.pick_top(learn.blend_scores(res.candidates, model, "Learned", p), many)
    out: list[sites.Site] = []
    for s in pool:
        if all(np.hypot(s.x - o.x, s.y - o.y) >= p.min_site_separation_px for o in out):
            out.append(s)
    return sorted(out, key=lambda s: (s.y, s.x))


def features_at(res: pipeline.Result, points: list[tuple[float, float]], p: Params = DEFAULT, tol: float = SAME_POINT_PX,
                with_sites: bool = False):
    """Feature rows for the candidates nearest each labeled point (within tol px). Returns (X, found-per-point),
    plus the matched Site objects if with_sites."""
    matched = []
    if not res.candidates:
        out = (np.zeros((0, len(learn.FEATURES))), [False] * len(points))
        return out + (matched,) if with_sites else out
    xy = np.array([(s.x, s.y) for s in res.candidates])
    rows, found = [], []
    for x, y in points:
        d = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
        j = int(np.argmin(d))
        found.append(bool(d[j] <= tol))
        if found[-1]:
            rows.append(learn.featurize(res.candidates[j], p))
            matched.append(res.candidates[j])
    out = (np.array(rows).reshape(-1, len(learn.FEATURES)), found)
    return out + (matched,) if with_sites else out


def training_rows(store: LabelStore, split: str, p: Params = DEFAULT, with_passed: bool = False):
    """(X, y, image ids, number of labels that no longer match a candidate) for every labeled image in a split.
    With with_passed (use p.apply_shape_rejects=False), also a boolean array: did each site pass the shape checks."""
    Xs, ys, ids, missing, passed = [], [], [], 0, []
    by_image: dict[int, list[dict]] = {}
    for r in store.for_split(split):
        by_image.setdefault(r["image"], []).append(r)
    for i, rows in sorted(by_image.items()):
        if not image_path(split, i).exists():
            missing += len(rows)
            continue
        res = pipeline.run(load_image(split, i, p), p)
        X, found, matched = features_at(res, [(r["x"], r["y"]) for r in rows], p, with_sites=True)
        kept = [r for r, f in zip(rows, found) if f]
        missing += len(rows) - len(kept)
        Xs.append(X)
        ys += [int(r["good"]) for r in kept]
        ids += [i] * len(kept)
        passed += [pipeline.passes_shape_checks(s, p) for s in matched]
    X = np.vstack(Xs) if Xs else np.zeros((0, len(learn.FEATURES)))
    out = (X, np.array(ys, int), np.array(ids, int), missing)
    return out + (np.array(passed, bool),) if with_passed else out
