# VIPER master brief (for Claude Code; read with CLAUDE.md first)

VIPER = **V**ein **I**maging and **P**uncture-site **E**valuation with **R**otation. Congressional App Challenge 2026 entry by Aradhya Pasikanti. Deadline Mon Oct 26 2026 12:00 pm EDT; internal submit target Sun Oct 25 8 pm.

## 0. Roles and rules (read docs/ROLES_AND_ATTRIBUTION.md)
1. Aradhya owns requirements, logic, scoring decisions and testing. Claude Code writes the code and tests (builder mode). Explain choices in plain language so he can defend them.
2. Every session ends with an honest entry in `docs/AI_LOG.md`. Never describe AI-written code as his. Keep git Co-Authored-By trailers.
3. No clinical claims. VIPER is an educational research prototype. Report only measured numbers on named data.
4. Do not download or train on any image whose license is unrecorded in `docs/DATA_SOURCES.md`.
5. For the Congressional App Challenge the student must do the coding. Track B (`student/`) exists for that; see roles doc.

## 1. Why it exists
Aradhya lost his little brother, who had hemophilia, to an internal brain bleed. People with hemophilia may need frequent IV infusions, so reliable venous access matters and repeated sticks in one place hurt. VIPER is the software core of his larger S.N.A.K.E. platform. The UI must be honest about being a prototype, and the tone of the README and video should be sincere, never exploitative. He writes Question 3 himself.

## 2. Product in one sentence
Given a near-infrared (NIR) arm image, VIPER shows where veins are, ranks the best puncture-site candidates with a plain-language reason for each, remembers which sites were used, ranks future sites to rotate away from them, and learns from the user's accept/reject choices.

## 3. Users and workflow
Audience: nursing/phlebotomy students and, as a concept, clinicians treating patients needing repeated access.
Flow: (1) load image (upload or synthetic) -> (2) enhance -> (3) veins + skeleton -> (4) mark wrist and elbow -> (5) ranked sites with reasons -> (6) user accepts, rejects, or logs "used" -> (7) next ranking rotates away and personalizes. Every tab explains itself in one sentence.

## 4. Architecture (already scaffolded in the repo)
`config.py` (all numbers) -> `preprocess.py` -> `vessels.py` -> `sites.py` -> `rotation.py` -> `learn.py` -> `app.py` (Streamlit). Helpers: `overlay.py`, `synthetic.py`. Tests in `tests/`. Pure functions, no globals, parameters through `Params`.

## 5. Algorithms (the logic, after a careful consistency pass)
**Enhance**: CLAHE (clip 3, tile 8) then Gaussian blur sigma 1. Output uint8, same shape.
**Vesselness**: multi-scale Frangi (sigmas 1.5-4), `black_ridges=True`, normalize by max to 0..1. Multi-scale is needed because veins vary in width; use the max over scales.
**Mask**: threshold at `vesselness_thresh * max`, remove components under `min_component_px`.
**Skeleton**: skeletonize the mask. Add: prune spurs shorter than ~8 px, because spurs create fake branch points that wrongly penalize good straight segments.
**Candidate features per skeleton pixel**:
- vesselness (mean in a 5x5 window)
- width_px = 2 x distance transform of the mask at that pixel
- straight_px = length of skeleton path through the pixel within the window that stays within ~3 px of a straight line (or the PCA ratio of neighbors: ratio of 2nd to 1st eigenvalue near 0 = straight)
- branch_clearance_px = distance to the nearest skeleton pixel with 3+ neighbors
- border_clearance_px = distance to the image edge
**Hard rejects** (score 0): branch clearance < 8 px, border clearance < 15 px, straight_px < 12 px.
**Score** (start point, then tune on synthetic seeds): `score = 0.30*vess + 0.25*width_n + 0.25*straight_n + 0.10*branch_n + 0.10*border_n`, each term clipped to 0..1 by dividing by a saturation value (width 12 px, straight 30 px, branch 30 px, border 40 px). Rationale: a good site is on a clearly detected, thick, straight vein away from junctions and edges. Width and straightness dominate because needle insertion needs a long, straight, large target. Vesselness guards against noise.
**Pick top N**: greedy non-max suppression, sort by score, accept a site only if at least `min_site_separation_px` from all accepted. Complexity O(N*k) is fine.
**Arm coordinates**: `axis = elbow - wrist; L = |axis|; u = axis/L; v = point - wrist; t = dot(v,u)/L; s = cross(u,v)/L` (sign convention documented). Makes sites comparable across images of different scale.
**Rotation**: for each candidate, `penalty = max over recent uses of exp(-(d/r)^2) * recency`, `d` = Euclidean distance in (t,s), `r = rotation_radius`, `recency = 1 - age_days/rotation_days` (0 beyond the window). Final `adjusted = score * (1 - 0.85*penalty)`. Gaussian falloff is smoother than a linear distance penalty and never fully zeros a site, so the system still returns options when the arm is crowded. Note text: "near site used 2 days ago". Never mutate the input list; return a new sorted list.

## 6. Implementation ownership
**STATUS (Oct 24, 12:58 pm): reference pipeline, rotation, learning layer, app, evaluation script and 20 tests are DONE and passing. Next work is UI polish, tuning with Aradhya's decisions, deployment, and his `student/` track.** Student overrides: any function defined in `student/__init__.py` with the same name replaces the reference one when the sidebar toggle is on (see `snake/pipeline.py`).

Claude Code implements: `enhance`, `vesselness`, `vein_mask`, `skeleton`, `score_candidates`, `pick_top`, `to_arm_coords`, `rank_with_rotation`, and `learn.py` (`featurize`, `make_training_set`, `train`, `predict`, `update_from_feedback`), plus `scripts/sweep.py`. Remove the NotImplementedError stubs as each lands; the skipped tests then run automatically. Create `student/` with the same function signatures for Aradhya's own versions, and an app sidebar toggle to switch between `snake/` (reference) and `student/`.

## 7. Efficiency
Images resized to max side 640 (already done). Frangi at four sigmas on 640x480 runs in about a second; cache results in Streamlit with `st.cache_data` keyed on image bytes and params. Candidate scoring must be vectorized with NumPy over skeleton pixels (not a Python loop per pixel). Target under 3 s end to end on a laptop.

## 8. Accuracy plan (honest)
- Segmentation: fraction of skeleton pixels within 4 px of a true vein centerline >= 0.85, and recall of true centerline pixels within 4 px of the skeleton >= 0.70, mean and minimum over 20 seeds. Report both.
- Sites: top-3 sites lie within 5 px of a true vein in >= 90% of seeds.
- Sweep script `scripts/sweep.py` (write it; a plain grid search over threshold and weights on seeds 0-19, evaluate on seeds 100-119 to avoid tuning on the test set).
- Real images: qualitative only, labelled as such.

## 9. Learning system ("AI/ML") — what is realistic in 36 hours
**Data reality (checked Oct 24):** the public NIR vein sets I found are gated or restricted, and none ships vein masks: VERA PalmVein is under a research-only EULA and requires an institutional signatory; NTUST IB811 wrist veins requires an IEEE DataPort subscription; others are palm/finger, not forearm. The sandbox cannot reach Zenodo or Kaggle. So "pull as many public images as possible and train on them" is neither license-safe nor labelled. Do not scrape.
**Therefore the learning layer is:**
1. **Pretrain on synthetic arms.** `make_training_set` generates hundreds of arms with known veins. Label a candidate 1 if within 3 px of a true centerline and on a straight stretch, else 0. Unlimited, license-clean, labelled.
2. **Model**: logistic regression first (explainable: show learned weights next to the hand-tuned ones), shallow decision tree as a comparison. Features: the five in `learn.FEATURES` (rotation is applied separately, after ranking). Evaluate on held-out seeds; must beat the majority-class baseline and report it against the hand-tuned score (AUC or top-3 hit rate).
3. **Personalization**: accept/reject clicks in the Sites tab are stored in `data/feedback.json`; `update_from_feedback` updates the model (SGD `partial_fit`, or refit with user rows weighted 5x). Show "VIPER has learned from N of your choices" and a before/after ranking.
4. **"Plan new sites"**: final score = `0.5*hand_tuned + 0.5*model_prob`, then rotation penalty. The user can switch between Rules, Learned, and Blend to compare, which also makes a great demo moment.
5. **Optional only if everything else is done:** if Aradhya obtains permission/real access to a dataset, run it as an evaluation-only qualitative demo, never as training data without a recorded license.
Avoid deep learning (a U-Net) for this entry: no labelled real data, no time, and it would be hard to show he wrote it.

## 10. UI/UX spec
Wide layout, one-sentence explainer under each tab title, sidebar for image source and the few sliders that matter. Sites tab: numbered markers, a ranked table with score and "why" text, accept/reject buttons, mode switch (Rules / Learned / Blend). Rotation tab: wrist/elbow inputs (stretch: click to place), log-used button, history table with ages, clear history. Persistent banner: "Educational research prototype. Not a medical device." Download button for the annotated image. Friendly errors for bad uploads. Colors: calm clinical palette, high contrast, no red/green-only encodings. Include a short "About VIPER" expander with the mission sentence (written by Aradhya).

## 11. Tests (exist or to add)
Scaffold tests pass now. Student tests unlock as functions land. Add after implementing: edge cases (empty mask -> no sites, no crash; sites near border; empty log; one-class feedback); a determinism test (same seed same result); a regression test pinning the accuracy numbers with a tolerance.

## 12. Definition of done for Sunday 8 pm
Pipeline works on synthetic + image upload; all non-skipped tests pass; learning layer trains and shows a measured comparison; README has real numbers; AI disclosure matches the log; video recorded and public; form submitted. Cut order if late: real-image demo, UI polish, blend mode, learning layer (keep Rules + rotation). Never cut tests, disclosure, video, submission.
