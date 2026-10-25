"""Step 4: remember which sites were used and rank new sites away from recent ones. Reference implementation by Claude Code.

Why arm coordinates? A site's pixel position changes if the photo changes. We define the arm axis from two
landmarks (wrist and elbow) so a site is stored as (t, s):
  t = how far along the arm from wrist (0) to elbow (1)
  s = sideways offset from the arm axis, divided by the wrist-to-elbow distance
"""
import json
import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .config import DEFAULT, Params
from .sites import Site

ARMS = ("Left", "Right")


@dataclass
class ArmFrame:
    wrist: tuple[float, float]   # (x, y) pixels
    elbow: tuple[float, float]   # (x, y) pixels


def frame_from_arm(arm: np.ndarray) -> ArmFrame:
    """Guess wrist and elbow from the arm mask of a real photo, so the user does not have to type coordinates.

    The arm's long direction is the main axis of its pixels (principal component). Its two ends are taken at the
    3rd and 97th percentile along that axis, so a few stray pixels don't set them. The forearm narrows toward the
    wrist, so the end whose band (10-25% in from the end) is narrower is called the wrist. A hand in the photo can
    fool this, which is why the app has a "swap wrist and elbow" button.
    """
    ys, xs = np.nonzero(arm)
    cx, cy = xs.mean(), ys.mean()
    _, vecs = np.linalg.eigh(np.cov(np.vstack([xs - cx, ys - cy])))
    ux, uy = vecs[:, 1]
    along = (xs - cx) * ux + (ys - cy) * uy
    across = (xs - cx) * -uy + (ys - cy) * ux
    lo, hi = np.percentile(along, [3, 97])

    def width(a: float, b: float) -> float:
        sel = (along >= min(a, b)) & (along <= max(a, b))
        return float(np.ptp(across[sel])) if sel.any() else 0.0

    span = hi - lo
    w_lo, w_hi = width(lo + 0.10 * span, lo + 0.25 * span), width(hi - 0.25 * span, hi - 0.10 * span)
    end_lo, end_hi = (cx + lo * ux, cy + lo * uy), (cx + hi * ux, cy + hi * uy)
    return ArmFrame(end_lo, end_hi) if w_lo <= w_hi else ArmFrame(end_hi, end_lo)


def to_arm_coords(x: float, y: float, frame: ArmFrame) -> tuple[float, float]:
    """Pixel (x, y) -> (t, s). Dot product with the unit arm axis gives distance along the arm; the 2D cross
    product gives the signed sideways offset (positive = clockwise side in image coordinates). Dividing by the
    arm length makes values comparable between photos of different size."""
    ax, ay = frame.elbow[0] - frame.wrist[0], frame.elbow[1] - frame.wrist[1]
    length = math.hypot(ax, ay)
    if length == 0:
        raise ValueError("Wrist and elbow must be different points.")
    ux, uy = ax / length, ay / length
    vx, vy = x - frame.wrist[0], y - frame.wrist[1]
    return (vx * ux + vy * uy) / length, (ux * vy - uy * vx) / length


class SiteLog:
    """Stores used sites as JSON on disk, or only in memory when path is None (one private log per app visitor)."""

    def __init__(self, path: str | Path | None = "data/site_log.json"):
        self.path = Path(path) if path is not None else None
        self.entries: list[dict] = []
        if self.path is not None and self.path.exists():
            self.entries = json.loads(self.path.read_text())

    def add(self, t: float, s: float, when: datetime | None = None, note: str = "", arm: str = "") -> None:
        when = when or datetime.now()
        self.entries.append({"t": t, "s": s, "when": when.isoformat(), "note": note, "arm": arm})
        self._save()

    def recent(self, days: int, now: datetime | None = None, arm: str = "") -> list[dict]:
        """Uses within the last `days`. With an arm given, only that arm's uses count: a site used on the left arm
        says nothing about the right arm. Old entries saved before arms were recorded count for both arms."""
        now = now or datetime.now()
        cutoff = now - timedelta(days=days)
        return [e for e in self.entries if datetime.fromisoformat(e["when"]) >= cutoff
                and (not arm or not e.get("arm") or e["arm"] == arm)]

    def clear(self) -> None:
        self.entries = []
        self._save()

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2))


def rotation_penalty(t: float, s: float, log: SiteLog, p: Params = DEFAULT, now: datetime | None = None, arm: str = ""):
    """Penalty in 0..1 and a note. Per recent use: exp(-(d/r)^2) * (1 - age/window); we keep the worst one.

    The Gaussian falloff is smooth (no cliff at the radius) and never exactly zero, and the recency factor
    fades a use out over rotation_days. Returns (penalty, note).
    """
    now = now or datetime.now()
    worst, note = 0.0, ""
    for e in log.recent(p.rotation_days, now, arm):
        age = (now - datetime.fromisoformat(e["when"])).total_seconds() / 86400
        recency = max(0.0, 1.0 - age / p.rotation_days)
        d = math.hypot(t - e["t"], s - e["s"])
        pen = math.exp(-((d / p.rotation_radius) ** 2)) * recency
        if pen > worst:
            worst = pen
            note = "near a site used today" if age < 1 else f"near a site used {int(age)} day(s) ago"
    return worst, (note if worst > 0.05 else "")


def rank_with_rotation(
    sites: list[Site], log: SiteLog, frame: ArmFrame, p: Params = DEFAULT, now: datetime | None = None, arm: str = ""
) -> list[Site]:
    """New list, best first, with scores reduced near recently used sites. Inputs are not modified."""
    out = []
    for s in sites:
        t, side = to_arm_coords(s.x, s.y, frame)
        pen, note = rotation_penalty(t, side, log, p, now, arm)
        out.append(replace(s, score=s.score * (1 - p.rotation_strength * pen), note=(note + "; " if note else "") + s.note))
    return sorted(out, key=lambda s: s.score, reverse=True)
