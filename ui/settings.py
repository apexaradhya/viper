"""Settings screen: the two choices a user needs, their data, and About VIPER. Algorithm tuning is on the
Developer page (only with ?dev=1)."""
import streamlit as st

from . import state, style

# Mission Statement
MISSION = "For people with hemophilia and other conditions that require frequent infusions, VIPER aims to make finding a vein faster and kinder, protecting veins for the years of care ahead."


def render() -> None:
    style.header("Adjust how VIPER suggests and rotates sites.")
    with st.container(key="card_prefs"):
        st.slider("Suggested sites", 1, 6, key="n_sites",
                  help="How many puncture-site suggestions VIPER shows for each photo.")
        st.slider("Rotation memory (days)", 1, 30, key="rotation_days",
                  help="How long a used site counts as recent. VIPER steers suggestions away from recent sites.")

    with st.container(key="card_data"):
        store = state.feedback_store()
        st.markdown("**Your feedback**")
        st.caption(f"VIPER has learned from {len(store.rows)} of your Good/Bad choices.")
        if st.button("Reset learning", disabled=not store.rows, icon=":material/restart_alt:"):
            store.clear()
            st.rerun()

    with st.expander("About VIPER", icon=":material/info:"):
        st.markdown(MISSION)
        st.caption("VIPER (Vein Imaging and Puncture-site Evaluation with Rotation) is the software core of S.N.A.K.E. "
                   "Designed and directed by Aradhya Pasikanti.")
