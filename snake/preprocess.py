"""Step 1: load an image and make veins easier to see. Implemented by Claude Code (reference implementation)."""
import cv2
import numpy as np
from scipy import ndimage as ndi

from .config import DEFAULT, Params


def load_gray(data: bytes) -> np.ndarray:
    """Decode uploaded image bytes into a uint8 grayscale array."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image. Use PNG or JPG.")
    return img


def resize_max(img: np.ndarray, max_side: int = DEFAULT.max_side) -> np.ndarray:
    """Downscale so the longest side is at most max_side (never upscales)."""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def arm_mask(gray: np.ndarray, p: Params = DEFAULT) -> np.ndarray | None:
    """Boolean mask of the arm in a photo with a dark background, or None if there is no background.

    Why: real NIR photos show a bright arm on a near-black background. The ridge filter sees the arm's outline as a
    huge dark ridge, and because vesselness was scaled by its single strongest pixel, that outline made the real
    veins look faint. Finding the arm lets the pipeline ignore the background and the outline.
    How: blur heavily (so veins and skin texture don't cut holes), call everything brighter than arm_bg_thresh
    "arm", keep the largest connected region, and fill holes. Synthetic arms have no background, so they get None
    and the pipeline behaves exactly as before.
    """
    blur = cv2.GaussianBlur(gray, (0, 0), p.arm_blur_sigma)
    fg = blur > p.arm_bg_thresh
    if fg.mean() > p.arm_max_fg_frac or not fg.any():
        return None
    labels, n = ndi.label(fg)
    largest = 1 + int(np.argmax(ndi.sum(fg, labels, np.arange(1, n + 1))))
    return ndi.binary_fill_holes(labels == largest)


def enhance(gray: np.ndarray, p: Params = DEFAULT) -> np.ndarray:
    """Contrast-enhance then lightly denoise.

    Why CLAHE: plain histogram equalization stretches contrast over the whole image, so a bright patch of skin
    drowns the faint veins. CLAHE equalizes in small tiles (with a clip limit so noise is not exploded), which
    brings out local contrast such as a vein against its neighbouring skin. The small Gaussian blur afterwards
    suppresses the speckle that CLAHE amplifies, so the ridge filter does not respond to noise.
    """
    clahe = cv2.createCLAHE(clipLimit=float(p.clahe_clip), tileGridSize=(int(p.clahe_tile), int(p.clahe_tile)))
    out = clahe.apply(gray.astype(np.uint8))
    if p.blur_sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=float(p.blur_sigma))
    return out
