# VIPER: Vein Imaging and Puncture-site Evaluation with Rotation
designed by Aradhya Pasikanti

Research prototype that maps veins in near-infrared arm images, ranks candidate IV access sites, and tracks site rotation so the same spot is not stuck repeatedly.

**Status:** work in progress, future implementation with SNAKE (Subcutaneous Navigation And Kinetic Engagement Platform)
**Disclaimer:** Currently an ERP, will be going under regulatory/clinical approval in the near future to be licensed

## What it does
1. **Enhance** the image (contrast and denoise).
2. **Detect veins** with a multi scale ridge filter, then skeletonize to center lines.
3. **Rank sites** by vein width, local straightness and distance from branch points.
4. **Rotate** by logging used sites in arm coordinates and pushing nearby sites down in the ranking.

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                 
streamlit run app.py
```

## File Workflow
```
snake/preprocess.py     Step 1: enhance()            
snake/vessels.py        Step 2: vesselness, vein_mask, skeleton   
snake/sites.py          Step 3: score_candidates, pick_top        
snake/rotation.py       Step 4: to_arm_coords, rank_with_rotation, SiteLog
snake/overlay.py        Drawing helpers 
snake/synthetic.py      Synthetic arm images with ground truth veins
docs/                   Spec, workflow, checklist, video script, answers
```

## Data
- Synthetic arms with known vein positions (generated in code).
- Public near-infrared vein images used under their licenses (Machine Learning based on 99 images on "forearm veins (NIR)"
- No patient data shared. No images are uploaded to any server by this app and instead logged directly on patient's devices


## Measured results
Reference pipeline, rotation tracking, learning layer, app and 20 tests are implemented. Measured on synthetic arms with known veins (`python scripts/evaluate.py`, results in docs/EVAL_RESULTS.txt): on clean images the skeleton is on a true vein 100% of the time and finds 99.9% of true vein pixels. 

On noisy images the accuracy drops to ~65% because noise creates false veins. A learned site ranker reaches AUC 0.977 vs 0.958 for the hand set rules on synthetic arms. Synthetic data is a simplified model to test proof of concept and for training. Real near-infrared images are future work that is expected to be out in the near future.
