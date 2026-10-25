"""Tunable parameters in one place. Change values here (or via the app sidebar), not inside algorithms."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # --- preprocessing ---
    max_side: int = 640            # downscale large images so everything runs fast
    clahe_clip: float = 3.0        # CLAHE contrast limit
    clahe_tile: int = 8            # CLAHE tile grid size
    blur_sigma: float = 1.0        # light denoise before ridge filtering

    # --- vessel detection ---
    veins_are_dark: bool = True    # in near-infrared images veins look darker than tissue
    frangi_sigmas: tuple = (1.5, 2.0, 3.0, 4.0)   # expected vein half-widths in pixels
    vesselness_thresh: float = 0.15               # fraction of max vesselness kept as "vein"
    min_component_px: int = 60                    # drop tiny specks

    # --- arm mask (real photos with a dark background; approved by Aradhya 2026-10-24, see DECISIONS.md) ---
    arm_bg_thresh: int = 40          # after blurring, pixels at or below this (0-255) are background, not arm
    arm_blur_sigma: float = 5.0      # blur before the background cutoff so skin texture and veins don't punch holes
    arm_max_fg_frac: float = 0.95    # if more of the image than this is "arm", there is no background: skip the mask
    arm_margin_px: int = 12          # ignore this band just inside the arm's outline (the outline looks like a ridge)
    arm_norm_percentile: float = 99.0  # vesselness is divided by this percentile inside the arm, not the image max

    # --- site selection ---
    n_sites: int = 3               # how many candidate sites to show
    min_site_separation_px: int = 25
    straight_window_px: int = 30   # length of vein segment used to judge straightness
    spur_len_px: int = 8           # skeleton branches shorter than this are noise spurs and get pruned

    # scoring weights (sum to 1) and the value at which each term counts as "perfect" (saturation)
    w_vess: float = 0.30
    w_width: float = 0.25
    w_straight: float = 0.25
    w_branch: float = 0.10
    w_border: float = 0.10
    sat_width_px: float = 8.0      # vein width (px) that earns the full width score
    sat_straight_px: float = 30.0  # straight stretch (px) that earns the full straightness score
    sat_branch_px: float = 30.0    # distance from nearest junction (px) that earns the full score
    sat_border_px: float = 40.0    # distance from image edge (px) that earns the full score
    min_branch_px: float = 8.0     # hard reject: closer than this to a junction
    min_border_px: float = 15.0    # hard reject: closer than this to the image edge
    min_straight_px: float = 12.0  # hard reject: straight stretch shorter than this

    # --- shape checks and extra model features (approved by Aradhya 2026-10-24, see DECISIONS.md) ---
    min_arm_alignment: float = 0.5   # hard reject: vein runs more than 60 deg across the arm (bands, cuffs, skin folds)
    max_contrast: float = 30.0       # hard reject: far darker than any vein seen (veins 3-8; bands and jewelry 70-80)
    align_path_px: int = 15          # how far along the vein to look when measuring its direction
    contrast_ring_px: tuple = (6, 12)  # skin ring (inner, outer radius) compared with the site's own brightness
    sat_piece_px: float = 150.0      # vein-piece length that counts as "long" for the model's 0..1 feature
    max_nearby_ridges: int = 1       # hard reject: 2+ OTHER ridge pieces within clutter_radius_px (ribbed fabric, hair)
    clutter_radius_px: int = 12
    apply_shape_rejects: bool = True  # evaluation turns this off to see the rejected sites too
    # below this the photo's veins are too faint: suggest no sites (0 = off). Off since 2026-10-24: 0.095 failed its
    # test check (blocked test photos had MORE good sites than shown ones). Kept for a better clarity measure later.
    min_photo_vein_strength: float = 0.0

    # --- rotation ---
    rotation_radius: float = 0.08  # in normalized arm units; sites closer than this to a used site are penalized
    rotation_days: int = 7         # how long a used site counts as "recently used"
    rotation_strength: float = 0.85  # max fraction of a score removed by a very recent, very close use


DEFAULT = Params()
