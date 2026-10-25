# VIPER: Specification (v0.1, refine together)

## 1. One-sentence purpose (use this in the video)
VIPER maps veins in near-infrared arm images, ranks the best access sites, and tracks which sites were used so they can be rotated.

## 2. Target audience
- Nursing and phlebotomy students learning vein access (education use).
- Clinicians and care teams as a *concept demo* for patients who need repeated IV sticks (for example chemotherapy or dialysis), where reusing the same spot causes damage and hard-to-find veins cause repeated attempts.
- Judges and reviewers: this is a research prototype, not a medical device.

## 3. Scope
**In scope (must ship by Sunday night):**
- F1 Image input: upload a PNG/JPG or use a synthetic sample.
- F2 Enhancement: contrast boost and denoise.
- F3 Vein detection: multi-scale ridge filter, mask, skeleton.
- F4 Site ranking: top N sites with a score and a plain-English reason.
- F5 Rotation tracking: log used sites in arm coordinates; re-rank so recently used sites drop.
- F6 Evidence: on synthetic arms, report how close detected veins are to the known truth.

**Out of scope (say so in the video):** live camera, real-time tracking, the robot arm, patient data, clinical validation, depth/diameter in millimeters.

**Stretch (only if everything above is done and tested):** user-picked landmarks by clicking on the image, export the site log as CSV, side-by-side before/after enhancement, a second synthetic arm with different lighting.

## 4. User workflow
1. Open the app. Choose **Synthetic sample** (or upload an image).
2. **Enhance** tab: see original vs enhanced.
3. **Veins** tab: see vesselness and the detected veins; (synthetic) see the accuracy metric.
4. **Sites** tab: see numbered recommended sites on the image with scores and reasons.
5. **Rotation** tab: set wrist and elbow, click **Log used site**, then see the rotation-aware ranking change.

## 5. Architecture
```
 image bytes / synthetic arm
        |
  preprocess.enhance        (Step 1)
        |
  vessels.vesselness  -> vein_mask -> skeleton     (Step 2)
        |
  sites.score_candidates -> pick_top               (Step 3)
        |
  rotation.to_arm_coords + SiteLog + rank_with_rotation   (Step 4)
        |
  app.py (Streamlit) + overlay.py (drawing)
```
Each box is a small pure function with a test. Parameters live in `snake/config.py`.

## 6. Data model
- `Site`: `x, y` (pixels), `score` (0..1), `width_px`, `straight_px`, `note`.
- `ArmFrame`: `wrist`, `elbow` pixel points. Site arm coordinates `(t, s)`: `t` along the arm (wrist 0, elbow 1), `s` sideways offset divided by wrist-elbow distance.
- Site log entry (JSON in `data/site_log.json`): `{t, s, when (ISO time), note}`.

## 7. Algorithm notes (decide and justify; this feeds the video and Q4)
- **Enhance:** CLAHE then light Gaussian blur. Why CLAHE: it boosts local contrast without blowing out bright regions.
- **Vesselness:** Frangi filter across several scales because veins differ in width. Veins are dark in NIR, so look for dark ridges.
- **Mask and skeleton:** threshold as a fraction of the max response, remove small specks, skeletonize to center lines.
- **Site score (suggested):** `score = w1*width + w2*straightness + w3*branch_clearance + w4*border_clearance`, normalized to 0..1. Put weights in `config.py`.
- **Rotation penalty:** if a candidate is within `rotation_radius` of a site used in the last `rotation_days`, multiply its score by `distance / radius`.

## 8. Acceptance criteria (the tests encode these)
- Enhancement keeps shape and dtype and does not reduce vein contrast.
- Vesselness is in [0, 1] and at least 3x higher on true veins than away from them (synthetic).
- At least 85% of skeleton pixels lie within 4 px of a true vein (synthetic).
- Top sites lie within 5 px of true veins, are separated, sorted by score.
- Arm coordinates: wrist maps to (0, 0), elbow to (1, 0).
- A recently used site drops below an unused one, and the reason is shown.

## 9. Ethics, safety, privacy
- Visible disclaimer in UI and README: educational research prototype, not medical advice.
- No patient data. Images are processed locally; nothing is uploaded.
- Synthetic data is a simplified model; real images vary. State this honestly.
- Public datasets: use only with proper license, and credit them in `docs/DATA_SOURCES.md`.

## 10. Risks and fallbacks
| Risk | Fallback |
| --- | --- |
| Frangi gives noisy detections on a real image | Tune sigmas and threshold; demo mainly on synthetic with the accuracy metric, show one real image honestly |
| Rotation math takes too long | Ship F1-F4 plus a simpler rotation (pixel distance), say so in the video |
| Streamlit layout fights | Keep the four tabs; skip polish |
| Not submitted in time | Submit by Sunday 8 pm; Monday morning is only a buffer |

## 11. Definition of done
- `pytest -q` passes with no skips.
- `streamlit run app.py` runs the whole flow on the synthetic sample.
- README, AI disclosure and AI log are accurate.
- Demo video is public on YouTube or Vimeo, 1-3 minutes.
- Submission form completed and confirmed before the deadline.

## Open questions to refine together
1. Which scoring weights feel right, and how will you explain them?
2. Do you want click-to-set landmarks or number inputs for wrist and elbow?
3. Do we include one real public image, and which license is acceptable?
4. App name: keep "VIPER" or "VeinAssist"?
