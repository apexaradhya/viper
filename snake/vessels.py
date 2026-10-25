"""Step 2: enhanced image -> vesselness -> vein mask -> skeleton (center lines). Reference implementation by Claude Code."""
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import frangi
from skimage.morphology import skeletonize

from .config import DEFAULT, Params

_K8 = np.ones((3, 3), int)


def raw_vesselness(enhanced: np.ndarray, p: Params = DEFAULT) -> np.ndarray:
    """Frangi ridge response before any rescaling (absolute units, comparable between photos)."""
    return frangi(enhanced.astype(np.float32) / 255.0, sigmas=p.frangi_sigmas, black_ridges=p.veins_are_dark)


def vesselness(enhanced: np.ndarray, p: Params = DEFAULT, raw: np.ndarray | None = None) -> np.ndarray:
    """Float array in [0, 1]; high = looks like a vein ridge.

    Frangi looks at the Hessian (second derivatives) at several scales. A tube has one strongly curved direction
    across it and almost none along it; blobs and flat areas do not. Taking the max over scales lets thin and
    thick veins both respond. black_ridges=True because veins are darker than skin in near-infrared.
    """
    v = raw_vesselness(enhanced, p) if raw is None else raw
    m = float(v.max())
    return (v / m).astype(np.float32) if m > 0 else np.zeros_like(v, dtype=np.float32)


def vein_mask(vess: np.ndarray, p: Params = DEFAULT) -> np.ndarray:
    """Boolean vein mask: threshold, then drop specks smaller than min_component_px."""
    mask = vess >= p.vesselness_thresh
    if not mask.any():
        return mask
    labels, n = ndi.label(mask, structure=_K8)
    sizes = ndi.sum(mask, labels, np.arange(1, n + 1))
    keep = np.concatenate([[False], sizes >= p.min_component_px])
    return keep[labels]


def neighbor_count(skel: np.ndarray) -> np.ndarray:
    """Number of 8-neighbours that are also skeleton, evaluated on skeleton pixels (0 elsewhere)."""
    return (ndi.convolve(skel.astype(int), _K8, mode="constant") - skel.astype(int)) * skel


def junctions(skel: np.ndarray) -> np.ndarray:
    """Boolean map of branch points (3+ skeleton neighbours)."""
    return neighbor_count(skel) >= 3


def prune_spurs(skel: np.ndarray, min_len: int) -> np.ndarray:
    """Remove short dead-end branches. They are skeletonization artefacts that would create fake junctions.

    Method: cut the skeleton at junctions, label the pieces, and delete pieces that (a) contain an endpoint,
    (b) are shorter than min_len, and (c) touch a junction. Pieces with no junction are whole isolated veins and stay.
    """
    if min_len <= 0 or not skel.any():
        return skel
    nb = neighbor_count(skel)
    junc = ndi.binary_dilation(nb >= 3, structure=_K8)
    pieces, n = ndi.label(skel & ~junc, structure=_K8)
    if n == 0:
        return skel
    idx = np.arange(1, n + 1)
    size = ndi.sum(np.ones_like(pieces), pieces, idx)
    has_end = ndi.maximum((nb == 1).astype(int), pieces, idx) > 0
    touches = ndi.maximum(ndi.binary_dilation(junc, structure=_K8).astype(int), pieces, idx) > 0
    kill = idx[(size < min_len) & has_end & touches]
    out = skel & ~np.isin(pieces, kill)
    return skeletonize(out)


def trace_path(skel: np.ndarray, x: float, y: float, max_len: int = 60) -> np.ndarray:
    """The stretch of vein center line through a site: skeleton pixels reachable from (x, y) within max_len steps.

    Why a walk along the skeleton instead of a circle around the site: it follows the vein itself, so the drawing
    shows which vein the site is on and which way it runs, and stops at gaps instead of jumping to a neighbour vein.
    Breadth-first search over 8-connected skeleton pixels. Returns a boolean mask (empty if no skeleton nearby).
    """
    out = np.zeros_like(skel, dtype=bool)
    ys, xs = np.nonzero(skel)
    if len(xs) == 0:
        return out
    k = int(np.argmin(np.hypot(xs - x, ys - y)))
    if np.hypot(xs[k] - x, ys[k] - y) > 3:
        return out
    h, w = skel.shape
    frontier = [(int(ys[k]), int(xs[k]))]
    out[frontier[0]] = True
    for _ in range(max_len):
        nxt = []
        for cy, cx in frontier:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] and not out[ny, nx]:
                        out[ny, nx] = True
                        nxt.append((ny, nx))
        if not nxt:
            break
        frontier = nxt
    return out


def skeleton(mask: np.ndarray, p: Params = DEFAULT) -> np.ndarray:
    """Boolean 1-pixel center line of the mask, with short spurs pruned."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return prune_spurs(skeletonize(mask), p.spur_len_px)
