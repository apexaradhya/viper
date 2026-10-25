"""Developer-only labeling page: Aradhya marks VIPER's candidate sites on Kaggle photos as good or bad.

Design choices (so the labels are fair to test with):
- Scores are hidden and sites are numbered top-to-bottom, so the labeler is not nudged by VIPER's opinion.
- The sites shown are chosen by the synthetic-only model and the rules, never by a model trained on these labels,
  so the set of sites to label stays the same while training changes.
- Colors: blue = good, orange = bad, white = not yet labeled, plus a letter, so meaning never depends on red/green.
"""
from dataclasses import replace

import cv2
import numpy as np
import streamlit as st

from snake import labels, pipeline
from snake.config import DEFAULT

from . import state

COLORS = {True: (255, 140, 0), False: (0, 140, 255), None: (255, 255, 255)}   # BGR: blue good, orange bad
TAGS = {True: "G", False: "B", None: ""}


@st.cache_data(show_spinner="Finding candidate sites...")
def _candidates(split: str, i: int):
    # clear-vein check off: weak photos are still worth labeling (their labels show why the check exists)
    res = pipeline.run(labels.load_image(split, i), replace(DEFAULT, min_photo_vein_strength=0.0))
    return res, labels.sites_to_label(res, state.synthetic_model())


def _draw(gray, sites, store, split, i):
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for n, s in enumerate(sites, 1):
        lab = store.get(split, i, s.x, s.y)
        c = (int(s.x), int(s.y))
        cv2.circle(bgr, c, 11, COLORS[lab], 2)
        cv2.putText(bgr, f"{n}{TAGS[lab]}", (c[0] + 13, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[lab], 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _first_unlabeled(split: str, store) -> int:
    ids = labels.available(split)
    todo = [i for i in ids if i not in store.images_done(split)]
    return (todo or ids)[0]


def render() -> None:
    st.title("Label sites")
    st.caption("Mark each numbered site as a good or bad place to insert a needle. Your labels train and test VIPER's model.")
    split = st.radio("Image set", ["train", "test"], horizontal=True,
                     format_func=lambda s: "Train (images 1-60)" if s == "train" else "Test (images 61-70)")
    ids = labels.available(split)
    if not ids:
        st.info(f"No Kaggle images found in {labels.KAGGLE_DIR / split}. They exist only on the developer's computer.")
        return
    store = labels.LabelStore()
    st.progress(len(store.images_done(split) & set(ids)) / len(ids),
                text=f"{len(store.images_done(split) & set(ids))} of {len(ids)} images labeled, {len(store.for_split(split))} sites")

    key = f"label_img_{split}"
    if key not in st.session_state or st.session_state[key] not in ids:
        st.session_state[key] = _first_unlabeled(split, store)
    i = st.selectbox("Image", ids, key=key)

    res, sites = _candidates(split, i)
    if not sites:
        st.warning("VIPER found no candidate sites on this image. Nothing to label; move on.")
    else:
        # the contrast-enhanced image, so veins are visible; no skeleton drawn, so the labeler judges the photo itself
        st.image(_draw(res.enhanced, sites, store, split, i), caption="Blue G = good, orange B = bad, white = not labeled yet")
        for n, s in enumerate(sites, 1):
            lab = store.get(split, i, s.x, s.y)
            c1, c2, c3 = st.columns([4, 1, 1])
            c1.write(f"**Site {n}** ({s.x:.0f}, {s.y:.0f}): " + {True: "good", False: "bad", None: "not labeled"}[lab])
            if c2.button("Good", key=f"g{split}{i}{n}", type="primary" if lab is True else "secondary"):
                store.set(split, i, s.x, s.y, True)
                st.rerun()
            if c3.button("Bad", key=f"b{split}{i}{n}", type="primary" if lab is False else "secondary"):
                store.set(split, i, s.x, s.y, False)
                st.rerun()
    if st.button("Next unlabeled image"):
        st.session_state.pop(key)
        st.rerun()
    st.caption("Then run `python scripts/train_real.py` to train on the train labels and test on the test labels.")
