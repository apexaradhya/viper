"""Home screen: start a scan (NIR camera placeholder) or upload a photo, then the suggested sites as cards.

Layout follows the mobile reference Aradhya chose: header, a dark primary card (Scan), a white secondary card
(Upload), a big rounded result image, then one card per suggested site with a plain-language reason and actions.
"""
from dataclasses import replace

import cv2
import numpy as np
import streamlit as st
from scipy.ndimage import distance_transform_edt

from snake import learn, overlay, pipeline, preprocess, rotation, sites, synthetic

from . import state, style

DIFFICULTY = {"Clean": (5.0, 1.0), "Noisy": (10.0, 0.7), "Faint + noisy": (14.0, 0.5)}


@st.cache_data(show_spinner=False)
def run_cached(gray_bytes: bytes, shape: tuple, params, use_student: bool):
    gray = np.frombuffer(gray_bytes, np.uint8).reshape(shape)
    return pipeline.run(gray, params, use_student)


def _scan_clicked() -> None:
    st.session_state.scan_note = True


def _start_cards(params):
    """Scan first (NIR camera, not connected yet), Upload second. Returns a grayscale image or None."""
    with st.container(key="scancard"):
        st.markdown("### Scan an arm")
        st.markdown("Hold the near-infrared camera over the forearm. VIPER finds the veins and suggests where to insert the needle.")
        st.button("Scan with NIR camera", icon=":material/photo_camera:", on_click=_scan_clicked, use_container_width=True)
    if st.session_state.get("scan_note"):
        # Placeholder until the NIR camera is connected (hardware integration comes later).
        st.info("The NIR camera isn't connected yet. For now, upload a near-infrared photo below.", icon=":material/info:")
    with st.container(key="card_upload"):
        st.markdown("**Or upload a photo**")
        st.caption("Roll sleeves up and keep clothing out of the frame: fabric patterns can look like veins.")
        upload = st.file_uploader("Near-infrared arm photo (PNG or JPG)", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    if upload is None:
        return None
    try:
        return preprocess.resize_max(preprocess.load_gray(upload.getvalue()), params.max_side)
    except ValueError:
        st.error("That file isn't a readable image. Try a PNG or JPG photo.", icon=":material/error:")
        return None


def _why(s: sites.Site) -> str:
    """Plain-language reason for a site, from the same measurements the score uses."""
    parts = [f"{s.width_px:.0f} px wide vein", f"straight for {s.straight_px:.0f} px",
             "far from branches" if s.branch_px >= 30 else f"{s.branch_px:.0f} px from a branch"]
    return " · ".join(parts)


def _site_cards(top, frame, log, store, params, arm_side, key_prefix):
    for i, s in enumerate(top, 1):
        with st.container(key=f"card_site{i}"):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(f"**Site {i}**")
                st.badge(f"score {s.score:.2f}", color="blue")
            st.caption(_why(s))
            if "near a site used" in s.note:
                st.badge(s.note.split(";")[0], icon=":material/history:", color="red")
            with st.container(horizontal=True):
                if st.button("Mark used", key=f"use{key_prefix}{i}", type="primary", icon=":material/check:",
                             disabled=frame is None, help="Log this site as used today, so VIPER rotates away from it"):
                    t, side = rotation.to_arm_coords(s.x, s.y, frame)
                    log.add(t, side, note=f"Site {i}", arm=arm_side)
                    st.toast(f"Site {i} logged on the {arm_side.lower()} arm")
                    st.rerun()
                if st.button("Good", key=f"acc{key_prefix}{i}", icon=":material/thumb_up:", help="VIPER learns from this"):
                    store.add(learn.featurize(s, params), True)
                    st.toast("Thanks, VIPER will learn from that")
                    st.rerun()
                if st.button("Bad", key=f"rej{key_prefix}{i}", icon=":material/thumb_down:", help="VIPER learns from this"):
                    store.add(learn.featurize(s, params), False)
                    st.toast("Thanks, VIPER will learn from that")
                    st.rerun()


def _swap_axis() -> None:
    st.session_state.swap_axis = not st.session_state.swap_axis


def _arm_frame(res, default_wrist, default_elbow):
    """Wrist/elbow (auto-detected on real photos), adjustable in an expander. Returns ArmFrame or None."""
    if res.arm is not None:
        guess = rotation.frame_from_arm(res.arm)
        default_wrist, default_elbow = guess.wrist, guess.elbow
    if st.session_state.swap_axis:
        default_wrist, default_elbow = default_elbow, default_wrist
    with st.expander("Wrist and elbow", icon=":material/straighten:"):
        st.caption("VIPER stores each used site as a position between wrist and elbow, so it can recognise the same spot in the next photo.")
        st.button("Swap wrist and elbow", on_click=_swap_axis, help="Use this if W is drawn at the elbow end.")
        c1, c2 = st.columns(2)
        wx = c1.number_input("Wrist x", value=float(default_wrist[0]))
        wy = c1.number_input("Wrist y", value=float(default_wrist[1]))
        ex = c2.number_input("Elbow x", value=float(default_elbow[0]))
        ey = c2.number_input("Elbow y", value=float(default_elbow[1]))
        frame = rotation.ArmFrame((wx, wy), (ex, ey)) if (wx, wy) != (ex, ey) else None
        if frame is None:
            st.error("Wrist and elbow are the same point. Move one of them.")
        else:
            st.image(overlay.draw_arm_frame(overlay.to_bgr(res.enhanced)[:, :, ::-1], frame.wrist, frame.elbow),
                     caption="W (blue) = wrist, E (orange) = elbow")
    return frame


def _how_it_works(gray, res, arm):
    with st.expander("How VIPER found these veins", icon=":material/visibility:"):
        st.caption("1. Contrast enhancement (CLAHE) makes faint veins stand out from skin.")
        c1, c2 = st.columns(2)
        c1.image(gray, caption="Photo", clamp=True)
        c2.image(res.enhanced, caption="Enhanced", clamp=True)
        st.caption("2. A ridge filter finds dark tube-like shapes; the thin line is each vein's center.")
        c1, c2 = st.columns(2)
        c1.image(np.clip(res.vess * 255, 0, 255).astype(np.uint8), caption="Vein-likeness", clamp=True)
        c2.image(overlay.overlay_mask(res.enhanced, res.mask), caption="Detected veins")
        if arm is not None and res.skel.any():
            d_true, d_skel = distance_transform_edt(~arm.vein_centerlines), distance_transform_edt(~res.skel)
            st.caption(f"Demo arm check: {100 * float((d_true[res.skel] <= 4).mean()):.0f}% of detected center line is on a true vein; "
                       f"{100 * float((d_skel[arm.vein_centerlines] <= 4).mean()):.0f}% of true veins found (synthetic image, not clinical accuracy).")


def render() -> None:
    style.header("Vein Imaging and Puncture-site Evaluation with Rotation")
    params = state.params()

    arm = None
    if st.session_state.demo_sample:            # developer-only demo image (Developer page)
        noise, contrast = DIFFICULTY[st.session_state.difficulty]
        arm = synthetic.make_arm(seed=int(st.session_state.seed), noise=noise, contrast=contrast)
        gray = arm.image
        st.badge("Demo sample arm (developer mode)", icon=":material/science:", color="blue")
        default_wrist, default_elbow = arm.wrist, arm.elbow
    else:
        gray = _start_cards(params)
        if gray is None:
            return
        h, w = gray.shape
        default_wrist, default_elbow = (w * 0.06, h * 0.5), (w * 0.94, h * 0.5)

    res = run_cached(np.ascontiguousarray(gray).tobytes(), gray.shape, params, st.session_state.use_student)
    store = state.feedback_store()
    log = state.site_log()

    st.subheader("Suggested sites")
    st.caption("Each circle sits on a traced vein. Spots near sites used recently on this arm move down the list.")
    st.segmented_control("Arm", rotation.ARMS, key="arm_side", label_visibility="collapsed")
    arm_side = st.session_state.get("arm_side") or "Left"

    cands = learn.blend_scores(res.candidates, state.current_model(), st.session_state.mode, params)
    pool = sites.pick_top(cands, replace(params, n_sites=max(6, params.n_sites)))
    frame = _arm_frame(res, default_wrist, default_elbow)
    top = rotation.rank_with_rotation(pool, log, frame, params, arm=arm_side)[: params.n_sites] if frame else pool[: params.n_sites]

    if not res.clear:
        with st.container(key="card_unclear"):
            st.markdown("**Veins aren't clear enough in this scan**")
            st.caption("VIPER won't guess on faint veins. Try again with the forearm closer to the camera, a tourniquet "
                       "applied, or steadier lighting.")
    elif not top:
        st.info("No safe site found in this photo. Try another photo with the forearm filling more of the frame.", icon=":material/search_off:")
    else:
        drawn = overlay.draw_sites_with_paths(res.enhanced, top, res.skel, res.arm)
        with st.container(key="hero"):
            st.image(drawn)
        _site_cards(top, frame, log, store, params, arm_side, st.session_state.seed)
        png = cv2.imencode(".png", drawn[:, :, ::-1])[1].tobytes()
        st.download_button("Save image", data=png, file_name="viper_sites.png", mime="image/png", icon=":material/download:")
    _how_it_works(gray, res, arm)
