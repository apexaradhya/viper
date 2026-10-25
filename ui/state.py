"""Settings shared by every page, plus the cached site model.

Why session_state: Streamlit forgets a widget's value when you leave the page it is on. Copying each setting back
into st.session_state on every run (keep_settings) keeps choices when the user switches between pages.

User settings (Settings page) are the two things a user plausibly wants to change. Everything that tunes the
algorithm lives on the Developer page, which only appears when the URL has ?dev=1 (or VIPER_DEV=1 is set).
"""
import os
from dataclasses import replace

import streamlit as st

from snake import learn, rotation
from snake.config import DEFAULT, Params

FEEDBACK_PATH = "data/feedback.json"
SITE_LOG_PATH = "data/site_log.json"
# Where a user's site history and Good/Bad choices live. On a public server every visitor shares one disk, so by
# default each visitor gets a private in-memory copy (gone when the tab closes). Set VIPER_LOCAL_FILES=1 when
# running on your own computer to keep them in data/*.json across restarts.
LOCAL_FILES = os.environ.get("VIPER_LOCAL_FILES") == "1"

USER_DEFAULTS = {"n_sites": DEFAULT.n_sites, "rotation_days": DEFAULT.rotation_days, "arm_side": "Left", "swap_axis": False}
DEV_DEFAULTS = {
    "mode": "Blend",
    "clahe_clip": DEFAULT.clahe_clip,
    "vesselness_thresh": DEFAULT.vesselness_thresh,
    "difficulty": "Clean",
    "seed": 0,
    "use_student": False,
    "demo_sample": False,
}


def keep_settings() -> None:
    for key, default in {**USER_DEFAULTS, **DEV_DEFAULTS}.items():
        st.session_state[key] = st.session_state.get(key, default)


def dev_mode() -> bool:
    """Once ?dev=1 is seen, stay in dev mode for this browser session (page links drop the query string)."""
    if st.query_params.get("dev") == "1" or os.environ.get("VIPER_DEV") == "1":
        st.session_state["dev"] = True
    return bool(st.session_state.get("dev", False))


def params() -> Params:
    s = st.session_state
    return replace(DEFAULT, clahe_clip=s.clahe_clip, vesselness_thresh=s.vesselness_thresh,
                   n_sites=s.n_sites, rotation_days=s.rotation_days)


@st.cache_resource(show_spinner="Preparing VIPER (one time)...")
def synthetic_model() -> learn.SiteModel:
    X, y = learn.make_training_set(12, DEFAULT, seed0=0)
    return learn.train(X, y)


def pretrained_model() -> learn.SiteModel:
    """The model saved by scripts/train_real.py (synthetic + Aradhya's labels) if present, else synthetic only."""
    return _saved_model() or synthetic_model()


@st.cache_resource(show_spinner=False)
def _saved_model() -> learn.SiteModel | None:
    return learn.load_model()


def current_model() -> learn.SiteModel:
    """Pretrained model, refit with this user's accept/reject choices if there are any."""
    model = pretrained_model()
    Xf, yf = feedback_store().arrays()
    return learn.update_from_feedback(model, Xf, yf) if len(Xf) else model


def site_log() -> rotation.SiteLog:
    if LOCAL_FILES:
        return rotation.SiteLog(SITE_LOG_PATH)
    if "site_log_obj" not in st.session_state:
        st.session_state.site_log_obj = rotation.SiteLog(None)
    return st.session_state.site_log_obj


def feedback_store() -> learn.FeedbackStore:
    if LOCAL_FILES:
        return learn.FeedbackStore(FEEDBACK_PATH)
    if "feedback_obj" not in st.session_state:
        st.session_state.feedback_obj = learn.FeedbackStore(None)
    return st.session_state.feedback_obj
