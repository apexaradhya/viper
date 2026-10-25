"""Drawing helpers for the app. AI-scaffolded."""
import cv2
import numpy as np

from .sites import Site


def to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def overlay_mask(gray: np.ndarray, mask: np.ndarray, color=(0, 200, 255), alpha: float = 0.5) -> np.ndarray:
    """Tint mask pixels on top of the grayscale image. Returns RGB for display."""
    bgr = to_bgr(gray)
    tint = bgr.copy()
    tint[mask.astype(bool)] = color
    out = cv2.addWeighted(tint, alpha, bgr, 1 - alpha, 0)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def draw_sites(gray: np.ndarray, sites: list[Site], skel: np.ndarray | None = None) -> np.ndarray:
    """Draw ranked sites as numbered circles. Returns RGB for display."""
    bgr = to_bgr(gray)
    if skel is not None:
        bgr[skel.astype(bool)] = (0, 200, 255)
    for i, s in enumerate(sites, start=1):
        c = (int(round(s.x)), int(round(s.y)))
        cv2.circle(bgr, c, 12, (0, 255, 0), 2)
        cv2.putText(bgr, str(i), (c[0] + 14, c[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


PATH_COLOR = (255, 175, 95)    # BGR light blue: the traced vein through a suggested site
SITE_COLOR = (87, 74, 196)     # BGR logo red: site ring and number badge (always numbered, never color-only meaning)


def draw_sites_with_paths(gray: np.ndarray, sites: list[Site], skel: np.ndarray, arm: np.ndarray | None = None,
                          path_len: int = 60) -> np.ndarray:
    """Numbered sites, each with the vein it sits on traced along the skeleton; faint skeleton elsewhere; thin arm
    outline for real photos. Returns RGB for display."""
    from .vessels import trace_path
    bgr = to_bgr(gray)
    bgr[skel.astype(bool)] = (bgr[skel.astype(bool)] * 0.4 + np.array([0, 120, 160]) * 0.6).astype(np.uint8)
    if arm is not None:
        contours, _ = cv2.findContours(arm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(bgr, contours, -1, (160, 160, 160), 1)
    for s in sites:
        path = cv2.dilate(trace_path(skel, s.x, s.y, path_len).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        bgr[path] = PATH_COLOR
    white = (255, 255, 255)
    for i, s in enumerate(sites, start=1):
        c = (int(round(s.x)), int(round(s.y)))
        cv2.circle(bgr, c, 13, white, 4, cv2.LINE_AA)            # white halo keeps the ring visible on any skin
        cv2.circle(bgr, c, 12, SITE_COLOR, 2, cv2.LINE_AA)
        b = (c[0] + 16, c[1] - 14)                               # numbered badge, up and to the right
        cv2.circle(bgr, b, 10, white, -1, cv2.LINE_AA)
        cv2.circle(bgr, b, 9, SITE_COLOR, -1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(str(i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.putText(bgr, str(i), (b[0] - tw // 2, b[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 2, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_arm_frame(rgb: np.ndarray, wrist: tuple[float, float], elbow: tuple[float, float]) -> np.ndarray:
    """Draw the wrist-elbow axis on an RGB image: blue W at the wrist, orange E at the elbow (letters, so meaning
    never rests on color alone)."""
    out = np.ascontiguousarray(rgb).copy()
    w = (int(wrist[0]), int(wrist[1]))
    e = (int(elbow[0]), int(elbow[1]))
    blue, orange = (0, 170, 255), (255, 160, 0)
    cv2.line(out, w, e, (200, 200, 200), 2)
    for pt, color, tag in ((w, blue, "W"), (e, orange, "E")):
        cv2.circle(out, pt, 8, color, -1)
        cv2.putText(out, tag, (pt[0] + 10, pt[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out
