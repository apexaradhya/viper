"""Developer page: algorithm knobs and diagnostics. Hidden from users; open the app with ?dev=1 to see it."""
import streamlit as st

from snake import learn, pipeline

from . import state


def render() -> None:
    st.title("Developer")
    st.caption("Hidden from users. Shown because the app was opened with ?dev=1.")

    st.subheader("Ranking")
    st.radio("Site ranking", ["Rules", "Learned", "Blend"], key="mode", horizontal=True,
             help="Rules = hand-set weights. Learned = model trained on synthetic arms plus feedback. Blend = average.")
    saved = state._saved_model()
    st.caption(f"Model in use: synthetic arms + {saved.n_seen} labeled sites ({learn.MODEL_PATH})" if saved
               else "Model in use: synthetic arms only (run scripts/train_real.py to add your labels)")
    with st.expander("What did the model learn?"):
        st.write("Positive numbers = this feature raises the learned score.")
        st.json(learn.coefficients(state.current_model()))

    st.subheader("Image processing")
    st.slider("CLAHE clip limit", 1.0, 8.0, step=0.5, key="clahe_clip")
    st.slider("Vein threshold (fraction of max)", 0.02, 0.6, step=0.01, key="vesselness_thresh")

    st.subheader("Synthetic sample")
    st.toggle("Show the demo sample arm on Home instead of Scan/Upload", key="demo_sample")
    st.number_input("Synthetic seed", 0, 99, key="seed")
    st.selectbox("Synthetic difficulty", ["Clean", "Noisy", "Faint + noisy"], key="difficulty")

    st.subheader("Implementation")
    overrides = pipeline.student_overrides()
    st.toggle("Use student/ functions", key="use_student", disabled=not overrides)
    st.caption("Student-written: " + (", ".join(overrides) if overrides else "none yet"))
