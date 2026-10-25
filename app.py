"""VIPER: Vein Imaging and Puncture-site Evaluation with Rotation. Streamlit app.

Run:  streamlit run app.py          (add ?dev=1 to the URL for the Developer and Label pages)
Designed and directed by Aradhya Pasikanti; reference implementation by Claude Code (see CONTRIBUTIONS.md).

This file only sets up the look and navigation. Pages live in ui/: home (scan/upload + suggested sites), history
(used sites), settings (user), developer and label (hidden tools). Navigation is a floating bar at the bottom, like
a phone app's tab bar, because VIPER is designed as a mobile app.
"""
import streamlit as st

from ui import developer, history, home, label, settings, state, style

st.set_page_config(page_title="VIPER", page_icon=str(style.ICON.with_name("viper_favicon.png")), layout="centered")
style.apply()
state.keep_settings()

pages = [
    st.Page(home.render, title="Home", icon=":material/home:", url_path="home", default=True),
    st.Page(history.render, title="History", icon=":material/history:", url_path="history"),
    st.Page(settings.render, title="Settings", icon=":material/settings:", url_path="settings"),
]
if state.dev_mode():
    pages += [st.Page(developer.render, title="Dev", icon=":material/tune:", url_path="developer"),
              st.Page(label.render, title="Label", icon=":material/label:", url_path="label")]

current = st.navigation(pages, position="hidden")
style.bottom_nav(pages)
current.run()
