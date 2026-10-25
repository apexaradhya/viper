"""Draw VIPER's top access sites on every image, with the vein path through each site traced. Run:
    python scripts/visualize_sites.py                      (Kaggle train + test images, with labels)
    python scripts/visualize_sites.py --input my_photos/   (any folder of PNG/JPG images)

Writes one annotated picture per image, a contact sheet per set, and summary.txt to --out (default data/viz/,
gitignored because the Kaggle images may not be redistributed). Where labeled sites, each suggested site is
checked against labels: a suggestion counts as "on a labeled site" if within 3 px of one. Sites he never
labeled are counted separately, never as hits.

Colors: light-blue line = traced vein, amber circle + number = suggestion, small blue square = GOOD label,
small orange square = BAD label.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake import labels, learn, overlay, pipeline, preprocess, sites  # noqa: E402
from snake.config import DEFAULT  # noqa: E402

GOOD, BAD = (255, 140, 0), (0, 140, 255)   # BGR blue / orange


def annotate(gray, res, top, image_labels):
    rgb = overlay.draw_sites_with_paths(res.enhanced, top, res.skel, res.arm)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for r in image_labels:
        c = (int(r["x"]), int(r["y"]))
        cv2.rectangle(bgr, (c[0] - 3, c[1] - 3), (c[0] + 3, c[1] + 3), GOOD if r["good"] else BAD, -1)
    return bgr


def match(top, image_labels):
    """For each suggestion: 'good', 'bad' or 'unlabeled' according to the nearest label within 3 px."""
    out = []
    for s in top:
        near = [r for r in image_labels if np.hypot(r["x"] - s.x, r["y"] - s.y) <= labels.SAME_POINT_PX]
        out.append("unlabeled" if not near else ("good" if near[0]["good"] else "bad"))
    return out


def contact_sheet(tiles, cols=6, width=320):
    tiles = [cv2.resize(t, (width, int(t.shape[0] * width / t.shape[1]))) for t in tiles]
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="folder of images (default: the local Kaggle train and test images)")
    ap.add_argument("--out", default="data/viz")
    ap.add_argument("--mode", default="Blend", choices=["Rules", "Learned", "Blend"])
    ap.add_argument("--n", type=int, default=DEFAULT.n_sites)
    args = ap.parse_args()
    p = DEFAULT
    from dataclasses import replace
    p = replace(p, n_sites=args.n)
    model = learn.load_model() or learn.train(*learn.make_training_set(12, DEFAULT, seed0=0))
    store = labels.LabelStore()

    if args.input:
        sets = {"input": [(f.stem, f) for f in sorted(Path(args.input).iterdir()) if f.suffix.lower() in (".png", ".jpg", ".jpeg")]}
    else:
        sets = {sp: [(f"{i:03d}", labels.image_path(sp, i)) for i in labels.available(sp)] for sp in labels.SPLITS}
    lines, counts = [f"mode {args.mode}, top {args.n} sites per image", ""], {}
    for split, items in sets.items():
        if not items:
            continue
        out_dir = Path(args.out) / split
        out_dir.mkdir(parents=True, exist_ok=True)
        tiles, c = [], {"good": 0, "bad": 0, "unlabeled": 0, "none": 0}
        for name, path in items:
            gray = preprocess.resize_max(preprocess.load_gray(path.read_bytes()), p.max_side)
            res = pipeline.run(gray, p)
            top = sites.pick_top(learn.blend_scores(res.candidates, model, args.mode, p), p)
            img_labels = [r for r in store.for_split(split) if f"{r['image']:03d}" == name] if split in labels.SPLITS else []
            verdicts = match(top, img_labels) if img_labels else ["unlabeled"] * len(top)
            if not top:
                c["none"] += 1
            for v in verdicts:
                c[v] += 1
            bgr = annotate(gray, res, top, img_labels)
            cv2.putText(bgr, f"{split} {name}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imwrite(str(out_dir / f"{name}.jpg"), bgr)
            tiles.append(bgr)
            why = "; ".join(f"{k}: {s.note} [{v}]" for k, (s, v) in enumerate(zip(top, verdicts), 1)) or "no safe site found"
            lines.append(f"{split} {name}: {why}")
        cv2.imwrite(str(Path(args.out) / f"{split}_sheet.jpg"), contact_sheet(tiles))
        counts[split] = c
        lines.append("")
    lines.append("Suggestions checked against labels (within 3 px):")
    for split, c in counts.items():
        judged = c["good"] + c["bad"]
        rate = f"{c['good'] / judged:.2f}" if judged else "n/a"
        lines.append(f"  {split}: on GOOD label {c['good']}, on BAD label {c['bad']}, not labeled {c['unlabeled']}, "
                     f"images with no site {c['none']} | good share of judged suggestions {rate}")
    text = "\n".join(lines)
    (Path(args.out) / "summary.txt").write_text(text + "\n")
    print("\n".join(lines[-1 - len(counts):]))
    print(f"Pictures and summary in {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
