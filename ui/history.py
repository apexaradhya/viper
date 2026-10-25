"""History screen: every logged needle site as a card, newest first, so the user can see what is still healing."""
from datetime import datetime

import streamlit as st

from . import state, style


def render() -> None:
    style.header("Sites you've used. Recent ones are avoided until they've had time to heal.")
    log = state.site_log()
    days = st.session_state.rotation_days
    now = datetime.now()
    if not log.entries:
        with st.container(key="card_empty"):
            st.markdown("**No sites logged yet**")
            st.caption('After an infusion, tap "Mark used" on the site you used. VIPER will then suggest different spots for the next few days.')
        return
    recent = log.recent(days, now)
    st.caption(f"{len(recent)} of {len(log.entries)} uses still count as recent (rotation memory: {days} days, set in Settings).")
    for k, e in enumerate(reversed(log.entries)):
        age = (now - datetime.fromisoformat(e["when"])).total_seconds() / 86400
        with st.container(key=f"card_use{k}"):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(f"**{e.get('arm') or 'Unknown'} arm · {e['note'] or 'site'}**")
                if e in recent:
                    st.badge("resting", icon=":material/bedtime:", color="red")
                else:
                    st.badge("ready", icon=":material/check:", color="blue")
            when = "today" if age < 1 else f"{age:.0f} day{'s' if age >= 1.5 else ''} ago"
            st.caption(f"{when} · {round(100 * e['t'])}% of the way from wrist to elbow")
    if st.button("Clear history", icon=":material/delete:"):
        log.clear()
        st.rerun()
