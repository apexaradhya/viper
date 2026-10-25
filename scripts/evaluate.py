"""Measure VIPER on synthetic arms with known veins. Run:  python scripts/evaluate.py [--arms 20]

Reports, per image condition: skeleton precision (skeleton px within 4 px of a true vein), recall (true vein px within
4 px of the skeleton), top-3 hit rate (top sites within 5 px of a true vein), and ranking quality (ROC AUC) of the
hand-set rule score vs the learned model on HELD-OUT seeds (100+). Synthetic data is a simplified model: these numbers
are not clinical accuracy.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake import learn, pipeline, sites, synthetic  # noqa: E402
from snake.config import DEFAULT  # noqa: E402


def segmentation(arms, p):
    prec, rec, hit = [], [], []
    for a in arms:
        r = pipeline.run(a.image, p)
        d_true, d_skel = ndi.distance_transform_edt(~a.vein_centerlines), ndi.distance_transform_edt(~r.skel)
        prec.append((d_true[r.skel] <= 4).mean() if r.skel.any() else 0.0)
        rec.append((d_skel[a.vein_centerlines] <= 4).mean())
        top = sites.pick_top(r.candidates, p)
        hit.append(float(bool(top) and all(d_true[int(s.y), int(s.x)] <= 5 for s in top)))
    return np.mean(prec), np.min(prec), np.mean(rec), np.min(rec), np.mean(hit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=int, default=20)
    n = ap.parse_args().arms
    p = DEFAULT
    print(f"Held-out seeds 100..{100 + n - 1}; parameters = config.DEFAULT\n")
    for noise, contrast in learn.CONDITIONS:
        arms = [synthetic.make_arm(seed=100 + i, noise=noise, contrast=contrast) for i in range(n)]
        pm, pmin, rm, rmin, hit = segmentation(arms, p)
        print(f"noise={noise:>4} contrast={contrast}: precision mean {pm:.3f} (min {pmin:.3f}) | recall mean {rm:.3f} (min {rmin:.3f}) | top-3 hit rate {hit:.2f}")
    X, y = learn.make_training_set(30, p, seed0=0)
    model = learn.train(X, y)
    Xt, yt = learn.make_training_set(max(n, 12), p, seed0=100)
    # rule score for the same rows: weights applied to the same normalised features
    w = np.array([p.w_vess, p.w_width, p.w_straight, p.w_branch, p.w_border])
    rule = Xt[:, : len(w)] @ w
    pr = learn.predict(model, Xt)
    print(f"\nRanking quality on held-out candidates (AUC, 1.0 = perfect, 0.5 = chance): rules {roc_auc_score(yt, rule):.3f} | learned {roc_auc_score(yt, pr):.3f}")
    print("learned coefficients:", learn.coefficients(model))


if __name__ == "__main__":
    main()
