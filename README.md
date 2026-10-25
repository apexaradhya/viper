# VIPER

Research prototype that maps veins in near-infrared arm images, ranks candidate IV access sites, and tracks site rotation so the same spot is not stuck repeatedly.

**Status:** Congressional App Challenge 2026 entry (work in progress).
**Disclaimer:** Educational research prototype. Not a medical device. Not for clinical use or medical decisions.

## What it does
1. **Enhance** the image (contrast and denoise).
2. **Detect veins** with a multi-scale ridge filter, then skeletonize to center lines.
3. **Rank sites** by vein width, local straightness and distance from branch points.
4. **Rotate** by logging used sites in arm coordinates and pushing nearby sites down in the ranking.

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                 # scaffold tests pass; your tests skip until implemented
streamlit run app.py
```

## Layout
```
app.py                  Streamlit UI (AI-scaffolded layout)
snake/config.py         All tunable parameters
snake/preprocess.py     Step 1: enhance()            (student)
snake/vessels.py        Step 2: vesselness, vein_mask, skeleton   (student)
snake/sites.py          Step 3: score_candidates, pick_top        (student)
snake/rotation.py       Step 4: to_arm_coords, rank_with_rotation (student), SiteLog (scaffold)
snake/overlay.py        Drawing helpers (scaffold)
snake/synthetic.py      Synthetic arm images with ground-truth veins (scaffold)
tests/                  Scaffold tests and student-step tests
docs/                   Spec, workflow, checklist, AI-use disclosure, video script, answers
prompts/                Claude Code prompt pack
```

## Data
- Synthetic arms with known vein positions (generated in code).
- Optional: public near-infrared vein images used under their licenses, with attribution in `docs/DATA_SOURCES.md`.
- No patient data. No images are uploaded to any server by this app.

## AI use
See `docs/AI_USE_DISCLOSURE.md` and `docs/AI_LOG.md`.


## Status and measured results
Reference pipeline, rotation tracking, learning layer, app and 20 tests are implemented (by Claude Code, designed and directed by Aradhya Pasikanti; see CONTRIBUTIONS.md). Measured on synthetic arms with known veins (`python scripts/evaluate.py`, results in docs/EVAL_RESULTS.txt): on clean images the skeleton is on a true vein 100% of the time and finds 99.9% of true vein pixels; on very noisy, faint images precision drops to 65% because noise creates false veins. A learned site ranker reaches AUC 0.977 vs 0.958 for the hand-set rules on held-out synthetic arms. Synthetic data is a simplified model; this is not clinical accuracy and real near-infrared images are future work.
