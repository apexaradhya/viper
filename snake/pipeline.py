"""One place that runs the whole pipeline, so the app, the learning layer and the sweep script stay consistent.
"""
from dataclasses import dataclass, replace
import importlib

import numpy as np
from scipy import ndimage as ndi

from . import preprocess, sites as _sites, vessels
from .config import DEFAULT, Params

_REF = {
    "enhance": preprocess.enhance,
    "vesselness": vessels.vesselness,
    "vein_mask": vessels.vein_mask,
    "skeleton": vessels.skeleton,
    "score_candidates": _sites.score_candidates,
    "pick_top": _sites.pick_top,
}


def student_overrides() -> list[str]:
    """Names of functions currently provided by the student/ package."""
    try:
        mod = importlib.import_module("student")
        importlib.reload(mod)
    except ImportError:
        return []
    return [n for n in _REF if callable(getattr(mod, n, None))]


def _fn(name: str, use_student: bool):
    if use_student:
        try:
            mod = importlib.import_module("student")
            f = getattr(mod, name, None)
            if callable(f):
                return f
        except ImportError:
            pass
    return _REF[name]


@dataclass
class Result:
    enhanced: np.ndarray
    vess: np.ndarray
    mask: np.ndarray
    skel: np.ndarray
    candidates: list
    arm: np.ndarray | None = None   # arm mask for real photos; None when the image has no dark background
    vein_strength: float = 1.0      # how strong the photo's veins are overall (see vein_strength())
    clear: bool = True              # False = veins too faint to suggest sites; the app asks for a clearer scan


def vein_strength(raw: np.ndarray, arm: np.ndarray | None) -> float:
    """99th percentile of the raw ridge response inside the arm (whole image if no arm mask).

    Why: vesselness is rescaled per photo so the strongest 1% of ridges count as fully vein-like. In a photo with
    no clear veins those strongest ridges are only skin texture, and rescaling would promote them. Measuring the
    strength BEFORE rescaling tells us whether the photo has clear veins at all. On training labels,
    the photos below 0.095 had 12% good sites vs 56% for the rest.
    """
    region = arm if arm is not None else np.ones(raw.shape, bool)
    return float(np.percentile(raw[region], 99)) if region.any() else 0.0


def run(gray: np.ndarray, p: Params = DEFAULT, use_student: bool = False) -> Result:
    arm = preprocess.arm_mask(gray, p)
    enhanced = _fn("enhance", use_student)(gray, p)
    raw = vessels.raw_vesselness(enhanced, p)
    strength = vein_strength(raw, arm)
    student_vess = use_student and _fn("vesselness", True) is not vessels.vesselness
    vess = _fn("vesselness", True)(enhanced, p) if student_vess else vessels.vesselness(enhanced, p, raw=raw)
    if arm is not None:
        vess = _vesselness_in_arm(vess, arm, p)
    mask = _fn("vein_mask", use_student)(vess, p)
    skel = _fn("skeleton", use_student)(mask) if _takes_one(_fn("skeleton", use_student)) else _fn("skeleton", use_student)(mask, p)
    cands = _fn("score_candidates", use_student)(skel, vess, p)
    if arm is not None:
        cands = _keep_off_arm_edge(cands, arm, p)
    cands = _add_shape_features(cands, gray, skel, arm, p)
    if p.apply_shape_rejects:
        cands = [s for s in cands if passes_shape_checks(s, p)]
    clear = strength >= p.min_photo_vein_strength
    return Result(enhanced, vess, mask, skel, cands if clear else [], arm, strength, clear)


def passes_shape_checks(s, p: Params = DEFAULT) -> bool:
    """Hard rejects for things that look like veins but are not: a band, cuff or skin fold runs ACROSS the forearm,
    while forearm veins run ALONG it; and jewelry or straps are far darker than any vein. On 122 training
    labels these removed 14 + 5 bad sites and no good ones.
    Clutter: ribbed or patterned fabric shows many parallel lines a few pixels apart, a vein is a single line. A site
    with 2+ other ridge pieces within 12 px was 98% of sites on a ribbed sleeve in photo, and 1 of 57
    good training sites."""
    return s.align >= p.min_arm_alignment and s.contrast <= p.max_contrast and s.nearby_ridges <= p.max_nearby_ridges


def _arm_axis(region: np.ndarray):
    """Centre, unit long direction and the min/max position along it, from the pixels of the arm region."""
    ys, xs = np.nonzero(region)
    c = np.array([xs.mean(), ys.mean()])
    _, vecs = np.linalg.eigh(np.cov(np.vstack([xs - c[0], ys - c[1]])))
    u = vecs[:, 1]
    along = (xs - c[0]) * u[0] + (ys - c[1]) * u[1]
    return c, u, float(along.min()), float(along.max())


def _add_shape_features(cands: list, gray: np.ndarray, skel: np.ndarray, arm: np.ndarray | None, p: Params) -> list:
    """Measure align, piece_px, arm_end and contrast for every candidate (copies; inputs unchanged).

    With no arm mask (synthetic arms, or photos without a dark background) the whole image is treated as the arm,
    so its long side is the arm direction.
    - align: |cos| of the angle between the vein's local direction (main axis of the skeleton within
      align_path_px steps of the site) and the arm's long direction.
    - piece_px: pixel count of the connected skeleton piece the site sits on.
    - arm_end: position along the arm, 0 at the middle of its length, 1 at either end.
    - contrast: mean brightness of a skin ring around the site (inside the arm) minus the site's brightness,
      after a light blur; computed for all pixels at once with ring-shaped filters.
    """
    if not cands:
        return cands
    region = arm if arm is not None else np.ones(gray.shape, bool)
    c, u, lo, hi = _arm_axis(region)
    pieces, _ = ndi.label(skel, structure=np.ones((3, 3), int))
    piece_len = np.bincount(pieces.ravel())
    r_in, r_out = p.contrast_ring_px
    yy, xx = np.mgrid[-r_out:r_out + 1, -r_out:r_out + 1]
    ring = ((np.hypot(xx, yy) >= r_in) & (np.hypot(xx, yy) <= r_out)).astype(np.float32)
    g = ndi.gaussian_filter(gray.astype(np.float32), 1.5)
    inside = region.astype(np.float32)
    ring_mean = ndi.convolve(g * inside, ring, mode="constant") / np.maximum(ndi.convolve(inside, ring, mode="constant"), 1e-6)
    out = []
    for s in cands:
        xi, yi = int(s.x), int(s.y)
        k = p.align_path_px          # the walk can't go further than k px, so a small crop is enough (much faster)
        y0, x0 = max(0, yi - k), max(0, xi - k)
        ys, xs = np.nonzero(vessels.trace_path(skel[y0:yi + k + 1, x0:xi + k + 1], xi - x0, yi - y0, k))
        if len(xs) >= 4:
            _, vecs = np.linalg.eigh(np.cov(np.vstack([xs - xs.mean(), ys - ys.mean()])))
            align = abs(float(vecs[:, 1] @ u))
        else:
            align = 1.0                       # too short to judge direction: do not reject on it
        t = ((xi - c[0]) * u[0] + (yi - c[1]) * u[1] - lo) / max(hi - lo, 1.0)
        r = p.clutter_radius_px
        win = pieces[max(0, yi - r): yi + r + 1, max(0, xi - r): xi + r + 1]
        nearby = len(set(np.unique(win).tolist()) - {0, int(pieces[yi, xi])})
        out.append(replace(s, align=align, piece_px=float(piece_len[pieces[yi, xi]]) if pieces[yi, xi] else 0.0,
                           arm_end=float(min(1.0, abs(t - 0.5) * 2)), contrast=float(ring_mean[yi, xi] - g[yi, xi]),
                           nearby_ridges=nearby))
    return out


def _vesselness_in_arm(vess: np.ndarray, arm: np.ndarray, p: Params) -> np.ndarray:
    """Zero vesselness outside the arm and in a margin along its outline, then rescale inside the arm.

    Dividing by the 99th percentile instead of the max: a handful of extreme pixels (bracelet, sleeve edge) would
    otherwise set the scale and make every real vein look faint. The top 1% of pixels inside the arm is a steadier
    reference for "this is as vein-like as it gets in this photo". Values above it are clipped to 1.
    """
    inner = ndi.binary_erosion(arm, iterations=p.arm_margin_px) if p.arm_margin_px else arm
    v = vess * inner
    ref = float(np.percentile(v[inner], p.arm_norm_percentile)) if inner.any() else 0.0
    return np.clip(v / ref, 0, 1).astype(np.float32) if ref > 0 else np.zeros_like(vess, dtype=np.float32)


def _keep_off_arm_edge(cands: list, arm: np.ndarray, p: Params) -> list:
    """Border distance becomes the distance to the nearer of the image edge and the arm's edge; rescore and re-apply
    the min_border_px hard limit, so no site is suggested at the side of the arm."""
    d_arm = ndi.distance_transform_edt(arm)
    out = []
    for s in cands:
        border = min(s.border_px, float(d_arm[int(s.y), int(s.x)]))
        if border >= p.min_border_px:
            s = replace(s, border_px=border)
            s.score = _sites.rule_score(s, p)
            out.append(s)
    return out


def _takes_one(f) -> bool:
    import inspect
    return len(inspect.signature(f).parameters) == 1
