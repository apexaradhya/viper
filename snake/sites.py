"""Step 3: score candidate access sites along the vein skeleton and pick the best few.

A good site is on a vein that is (1) clearly detected, (2) wide, (3) straight for a stretch, and (4) away from
branch points and the image border. Each property is turned into a 0..1 term; the score is their weighted sum
(weights in config.py). Hard limits reject sites that are unsafe regardless of the other terms.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .config import DEFAULT, Params
from .vessels import junctions


@dataclass
class Site:
    x: float                  # pixel column
    y: float                  # pixel row
    score: float              # higher is better, 0..1
    width_px: float = 0.0     # estimated vein width at this point
    straight_px: float = 0.0  # length of locally straight vein around this point
    note: str = ""            # short human-readable reason, shown in the app
    vess: float = 0.0         # local vesselness 0..1
    branch_px: float = 0.0    # distance to nearest junction
    border_px: float = 0.0    # distance to nearest image edge
    align: float = 1.0        # 1 = vein runs along the arm, 0 = straight across it
    piece_px: float = 0.0     # length of the connected piece of vein the site is on
    arm_end: float = 0.0      # 0 = middle of the arm's length, 1 = at the wrist or elbow end
    contrast: float = 0.0     # how much darker the site is than the skin ring around it (0-255 levels)
    nearby_ridges: int = 0    # other separate ridge pieces close to the site (many = fabric ribbing or hair)


def local_straightness(binary: np.ndarray, window: int):
    """For every pixel, how straight is the binary curve inside a window x window box around it?

    Uses local principal-axis analysis computed with box filters (vectorized, no per-pixel loops): from the moments
    of the curve pixels in the box we get the 2x2 covariance, whose eigenvalues l1 >= l2 measure spread along and
    across the dominant direction. shape = 1 - l2/l1 is 1 for a perfect line and 0 for a blob/cross.
    length = sqrt(12*l1) is the extent along the line (a uniform line of length L has variance L^2/12).
    Returns (shape, straight_px) arrays.
    """
    b = binary.astype(np.float64)
    h, w = b.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    uf = lambda a: ndi.uniform_filter(a, size=window, mode="constant")
    n = uf(b)
    safe = np.where(n > 1e-9, n, 1.0)
    mx, my = uf(b * xx) / safe, uf(b * yy) / safe
    cxx = np.maximum(uf(b * xx * xx) / safe - mx * mx, 0)
    cyy = np.maximum(uf(b * yy * yy) / safe - my * my, 0)
    cxy = uf(b * xx * yy) / safe - mx * my
    tr, det = cxx + cyy, cxx * cyy - cxy * cxy
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 0)
    shape = np.where(l1 > 1e-9, 1 - l2 / np.where(l1 > 1e-9, l1, 1), 0.0)
    shape = np.clip(shape, 0, 1) * (n * window * window >= 4)
    length = np.sqrt(12 * l1) * shape
    return shape, length


def site_terms(site: Site, p: Params = DEFAULT) -> dict:
    """0..1 terms for a site. The first five make the rule-based score; all eight are ML features.

    The last three were added after studying labels on real photos: good sites lay on veins running
    along the arm, on longer continuous vein pieces, and away from the wrist and elbow ends."""
    clip = lambda v: float(min(1.0, max(0.0, v)))
    return {
        "vesselness": clip(site.vess),
        "width": clip(site.width_px / p.sat_width_px),
        "straight": clip(site.straight_px / p.sat_straight_px),
        "branch": clip(site.branch_px / p.sat_branch_px),
        "border": clip(site.border_px / p.sat_border_px),
        "align": clip(site.align),
        "piece": clip(site.piece_px / p.sat_piece_px),
        "arm_end": clip(site.arm_end),
    }


def rule_score(site: Site, p: Params = DEFAULT) -> float:
    t = site_terms(site, p)
    return p.w_vess * t["vesselness"] + p.w_width * t["width"] + p.w_straight * t["straight"] + p.w_branch * t["branch"] + p.w_border * t["border"]


def score_candidates(skel: np.ndarray, vess: np.ndarray, p: Params = DEFAULT) -> list[Site]:
    """A scored Site for every skeleton pixel that passes the hard limits."""
    if not skel.any():
        return []
    h, w = skel.shape
    mask = vess >= p.vesselness_thresh
    width = 2.0 * ndi.distance_transform_edt(mask)           # local vein width in px
    vloc = ndi.uniform_filter(vess, size=5)                    # steadier than the raw pixel value
    _, straight = local_straightness(skel, p.straight_window_px)
    j = junctions(skel)
    branch = ndi.distance_transform_edt(~j) if j.any() else np.full(skel.shape, 1e3)
    yy, xx = np.mgrid[0:h, 0:w]
    border = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy]).astype(float)

    ok = skel & (branch >= p.min_branch_px) & (border >= p.min_border_px) & (straight >= p.min_straight_px)
    out = []
    for y, x in zip(*np.nonzero(ok)):
        s = Site(x=float(x), y=float(y), score=0.0, width_px=float(width[y, x]), straight_px=float(straight[y, x]),
                 vess=float(vloc[y, x]), branch_px=float(min(branch[y, x], 999.0)), border_px=float(border[y, x]))
        s.score = rule_score(s, p)
        s.note = explain(s)
        out.append(s)
    return out


def explain(s: Site) -> str:
    return f"width {s.width_px:.0f}px, straight {s.straight_px:.0f}px, {min(s.branch_px, 99):.0f}px from branch"


def pick_top(sites: list[Site], p: Params = DEFAULT) -> list[Site]:
    """Greedy non-maximum suppression: best first, skip anything closer than min_site_separation_px to a kept site."""
    kept: list[Site] = []
    for s in sorted(sites, key=lambda s: s.score, reverse=True):
        if all(np.hypot(s.x - k.x, s.y - k.y) >= p.min_site_separation_px for k in kept):
            kept.append(s)
            if len(kept) == p.n_sites:
                break
    return kept
