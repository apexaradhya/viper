"""VIPER learning layer: learns which candidate sites are good and adapts to a user's accept/reject choices.

1. Pretrain on unlimited SYNTHETIC arms. Labels come from ground truth that is independent of the detector:
   a candidate is "good" if it is on a true vein, the TRUE vein is locally straight there, and it is away from the TRUE junction.
2. Personalize: user feedback rows are added (weighted higher) and the model is refit. Logistic regression is used
   because it is small, fast and explainable (coefficients can be shown next to the hand-set rule weights).
Implemented by Claude Code (reference implementation).
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import pipeline, synthetic
from .config import DEFAULT, Params
from .sites import Site, local_straightness, rule_score, site_terms
from .vessels import junctions

FEATURES = ["vesselness", "width", "straight", "branch", "border", "align", "piece", "arm_end"]
RULE_FEATURES = FEATURES[:5]   # the five the hand-set rule score uses
FEEDBACK_WEIGHT = 5.0


@dataclass
class SiteModel:
    clf: LogisticRegression | None = None
    scaler: StandardScaler | None = None
    const: float | None = None            # used when training data has a single class
    X_pre: np.ndarray = field(default_factory=lambda: np.zeros((0, len(FEATURES))))
    y_pre: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    X_fb: np.ndarray = field(default_factory=lambda: np.zeros((0, len(FEATURES))))
    y_fb: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    n_seen: int = 0                       # number of user feedback rows used


def featurize(site: Site, p: Params = DEFAULT) -> np.ndarray:
    t = site_terms(site, p)
    return np.array([t[k] for k in FEATURES], dtype=float)


def featurize_many(sites: list[Site], p: Params = DEFAULT) -> np.ndarray:
    return np.array([featurize(s, p) for s in sites]).reshape(len(sites), len(FEATURES))


CONDITIONS = [(5.0, 1.0), (10.0, 0.7), (14.0, 0.5)]  # (noise std, vein contrast): clean, noisy, faint+noisy


def make_training_set(n_arms: int, p: Params = DEFAULT, seed0: int = 0, stride: int = 3):
    """Features X and labels y from synthetic arms seed0 .. seed0+n_arms-1.

    Image difficulty cycles through CONDITIONS so the model sees clean and hard cases. Every `stride`-th
    candidate is kept to keep training fast."""
    Xs, ys = [], []
    for seed in range(seed0, seed0 + n_arms):
        noise, contrast = CONDITIONS[seed % len(CONDITIONS)]
        arm = synthetic.make_arm(seed=seed, noise=noise, contrast=contrast)
        res = pipeline.run(arm.image, p)
        if not res.candidates:
            continue
        truth = skeletonize(arm.vein_centerlines)
        d_true = ndi.distance_transform_edt(~truth)
        _, true_straight = local_straightness(truth, p.straight_window_px)
        tj = junctions(truth)
        d_junc = ndi.distance_transform_edt(~tj) if tj.any() else np.full(truth.shape, 1e3)
        for s in res.candidates[::stride]:
            xi, yi = int(s.x), int(s.y)
            good = d_true[yi, xi] <= 3 and true_straight[yi, xi] >= 20 and d_junc[yi, xi] >= 10
            Xs.append(featurize(s, p))
            ys.append(int(good))
    return np.array(Xs).reshape(-1, len(FEATURES)), np.array(ys, dtype=int)


def _fit(model: SiteModel) -> SiteModel:
    X = np.vstack([model.X_pre, model.X_fb]) if len(model.X_fb) else model.X_pre
    y = np.concatenate([model.y_pre, model.y_fb]) if len(model.y_fb) else model.y_pre
    w = np.concatenate([np.ones(len(model.y_pre)), np.full(len(model.y_fb), FEEDBACK_WEIGHT)])
    if len(np.unique(y)) < 2:
        model.clf, model.scaler = None, None
        model.const = float(y.mean()) if len(y) else 0.5
        return model
    model.scaler = StandardScaler().fit(X)
    model.clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(model.scaler.transform(X), y, sample_weight=w)
    model.const = None
    return model


def train(X: np.ndarray, y: np.ndarray) -> SiteModel:
    return _fit(SiteModel(X_pre=np.asarray(X, float), y_pre=np.asarray(y, int)))


def predict(model: SiteModel, X: np.ndarray) -> np.ndarray:
    """Probability each row is a good site, in 0..1."""
    X = np.atleast_2d(np.asarray(X, float))
    if model.clf is None:
        return np.full(len(X), 0.5 if model.const is None else model.const)
    return model.clf.predict_proba(model.scaler.transform(X))[:, 1]


def update_from_feedback(model: SiteModel, X: np.ndarray, accepted: np.ndarray) -> SiteModel:
    """Add the user's accept(1)/reject(0) rows (weighted x5) and refit. Safe with 0 rows or one class."""
    X = np.asarray(X, float).reshape(-1, len(FEATURES))
    accepted = np.asarray(accepted, int)
    if len(X) == 0:
        return model
    new = replace(model, X_fb=np.vstack([model.X_fb, X]), y_fb=np.concatenate([model.y_fb, accepted]))
    new.n_seen = model.n_seen + len(X)
    return _fit(new)


def blend_scores(sites: list[Site], model: SiteModel, mode: str, p: Params = DEFAULT) -> list[Site]:
    """Return copies of sites with score from 'Rules' (unchanged), 'Learned' (model) or 'Blend' (mean of both)."""
    if mode == "Rules" or not sites:
        return sites
    prob = predict(model, featurize_many(sites, p))
    out = []
    for s, pr in zip(sites, prob):
        score = float(pr) if mode == "Learned" else 0.5 * rule_score(s, p) + 0.5 * float(pr)
        out.append(replace(s, score=score))
    return out


MODEL_PATH = Path("models/site_model.pkl")


def save_model(model: SiteModel, path: str | Path = MODEL_PATH) -> None:
    """Pickle the trained model. It holds only numbers (coefficients and feature rows), never images."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(model))


def load_model(path: str | Path = MODEL_PATH) -> SiteModel | None:
    """The saved model, or None if there is none. Only load files this project wrote (pickle can run code)."""
    path = Path(path)
    return pickle.loads(path.read_bytes()) if path.exists() else None


def coefficients(model: SiteModel) -> dict:
    if model.clf is None:
        return {}
    return dict(zip(FEATURES, model.clf.coef_[0].round(3).tolist()))


class FeedbackStore:
    """Persists the user's accept/reject rows in JSON so learning survives restarts (memory only if path is None)."""

    def __init__(self, path: str | Path | None = "data/feedback.json"):
        self.path = Path(path) if path is not None else None
        self.rows: list[dict] = json.loads(self.path.read_text()) if self.path is not None and self.path.exists() else []

    def add(self, x: np.ndarray, accepted: bool) -> None:
        self.rows.append({"x": [float(v) for v in x], "y": int(accepted)})
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.rows))

    def arrays(self):
        rows = [r for r in self.rows if len(r["x"]) == len(FEATURES)]   # skip rows saved before features were added
        if not rows:
            return np.zeros((0, len(FEATURES))), np.zeros(0, int)
        return np.array([r["x"] for r in rows]), np.array([r["y"] for r in rows])

    def clear(self) -> None:
        self.rows = []
        if self.path is not None and self.path.exists():
            self.path.unlink()
