"""Train the site model on synthetic arms + Aradhya's labels (Kaggle 1-60), test on his labels for 61-70. Run:
    python scripts/train_real.py --model-out PATH     (also writes --report-out)

Training: the synthetic-pretrained model is refit with his labeled TRAIN rows weighted 5x, the existing rule for a
user's own choices (learn.FEATURES has 8 features since 2026-10-24). Only labeled sites that pass the shape checks
are used, because the model only ever ranks sites that passed them.

Test: ALL his labeled TEST sites are ranked by each method. Sites a method's shape checks reject are put last, so
wrongly rejecting a good site counts against it. Methods:
  "before" = what the app did until today: 5 features, no shape checks.
  "after"  = approved changes: shape checks + 8 features, synthetic + his train labels.
Numbers:
  AUC: chance a random good site ranks above a random bad one (0.5 = chance).
  top-3 good rate: on each test image, of the top 3 ranked labeled sites, the share he marked good.
  95% range: bootstrap over whole test images (2000 resamples), since sites on one arm are not independent.
  p (vs random): permutation test; shuffle his labels within each image 5000 times and count how often random
  top-3 picks do at least as well. Small p (< 0.05) = unlikely to be luck.
One labeler, 10 test images: small-sample numbers, not clinical accuracy.
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake import labels, learn  # noqa: E402
from snake.config import DEFAULT  # noqa: E402

RULE_W = np.array([DEFAULT.w_vess, DEFAULT.w_width, DEFAULT.w_straight, DEFAULT.w_branch, DEFAULT.w_border])
N5 = len(RULE_W)


def top3_good_rate(score: np.ndarray, y: np.ndarray, ids: np.ndarray) -> float:
    hits = []
    for i in np.unique(ids):
        k = np.nonzero(ids == i)[0]
        hits += list(y[k[np.argsort(-score[k], kind="stable")[:3]]])
    return float(np.mean(hits)) if hits else float("nan")


def bootstrap(score, y, ids, rng, n=2000):
    imgs, aucs = np.unique(ids), []
    for _ in range(n):
        k = np.concatenate([np.nonzero(ids == i)[0] for i in rng.choice(imgs, len(imgs))])
        if len(np.unique(y[k])) == 2:
            aucs.append(roc_auc_score(y[k], score[k]))
    return np.percentile(aucs, [2.5, 97.5])


def permutation_p(score, y, ids, rng, n=5000):
    obs, imgs, count = top3_good_rate(score, y, ids), np.unique(ids), 0
    for _ in range(n):
        yy = y.copy()
        for i in imgs:
            k = np.nonzero(ids == i)[0]
            yy[k] = rng.permutation(yy[k])
        count += top3_good_rate(score, yy, ids) >= obs
    return count / n


def report(name, score, y, ids, rng) -> str:
    lo, hi = bootstrap(score, y, ids, rng)
    return (f"{name:44s} AUC {roc_auc_score(y, score):.3f} (95% range {lo:.2f}-{hi:.2f}) | "
            f"top-3 good rate {top3_good_rate(score, y, ids):.2f}, p vs random {permutation_p(score, y, ids, rng):.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(labels.LABELS_PATH))
    ap.add_argument("--model-out", default=str(learn.MODEL_PATH))
    ap.add_argument("--report-out", default="docs/EVAL_REAL.txt")
    args = ap.parse_args()
    store = labels.LabelStore(args.labels)
    p_all = replace(DEFAULT, apply_shape_rejects=False, min_photo_vein_strength=0.0)   # see every labeled site
    rng = np.random.default_rng(0)

    Xtr, ytr, idtr, miss_tr, ok_tr = labels.training_rows(store, "train", p_all, with_passed=True)
    Xte, yte, idte, miss_te, ok_te = labels.training_rows(store, "test", p_all, with_passed=True)
    lines = [f"Labels: {args.labels}",
             f"train (Kaggle 1-60): {len(ytr)} labeled sites on {len(np.unique(idtr))} images, {int(ytr.sum())} good; "
             f"{miss_tr} unmatched; shape checks reject {int((~ok_tr & (ytr == 0)).sum())} bad and {int((~ok_tr & (ytr == 1)).sum())} good",
             f"test (Kaggle 61-70): {len(yte)} labeled sites on {len(np.unique(idte))} images, {int(yte.sum())} good; "
             f"{miss_te} unmatched; shape checks reject {int((~ok_te & (yte == 0)).sum())} bad and {int((~ok_te & (yte == 1)).sum())} good"]
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        print("\n".join(lines + ["Need both good and bad labels in train and in test. Label more images first."]))
        return 1

    old_synth = learn.train(*[a[:, :N5] if a.ndim == 2 else a for a in learn.make_training_set(12, p_all, seed0=0)])
    new_synth = learn.train(*learn.make_training_set(12, DEFAULT, seed0=0))
    after = learn.update_from_feedback(new_synth, Xtr[ok_tr], ytr[ok_tr])
    last = lambda s: np.where(ok_te, s, -1.0)                       # rejected sites rank last
    rules = Xte[:, :N5] @ RULE_W
    lines += ["", "Ranking ALL labeled test sites (rejected sites ranked last for methods with shape checks):",
              report("BEFORE rules (5 features, no checks)", rules, yte, idte, rng),
              report("BEFORE app model (synthetic, 5 feat, no checks)", learn.predict(old_synth, Xte[:, :N5]), yte, idte, rng),
              report("AFTER rules + shape checks", last(rules), yte, idte, rng),
              report("AFTER model, synthetic only (8 feat + checks)", last(learn.predict(new_synth, Xte)), yte, idte, rng),
              report("AFTER model, synthetic + his labels (proposed)", last(learn.predict(after, Xte)), yte, idte, rng),
              report("AFTER blend (0.5 rules + 0.5 proposed)", last(0.5 * rules + 0.5 * learn.predict(after, Xte)), yte, idte, rng),
              f"{'baseline: share of test sites good':44s} {yte.mean():.2f}",
              "", f"proposed model coefficients: {learn.coefficients(after)}"]
    learn.save_model(after, args.model_out)
    lines.append(f"\nSaved proposed model to: {args.model_out}")
    text = "\n".join(lines)
    print(text)
    Path(args.report_out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
