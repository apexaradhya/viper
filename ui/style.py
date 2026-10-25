"""Shared look for every page: frosted-glass cards over soft logo-colored glows, accent gradient from the logo,
floating frosted bottom navigation, app header.

Base colors and font come from .streamlit/config.toml. CSS is used only for what the theme has no option for
(Aradhya asked for this specific look): the background glows, glass cards, gradient accents and the fixed bottom
bar. Card and nav rules target containers by key (class "st-key-<key>"), so they cannot leak onto other widgets.

Palette (from the logo): slate blue #4C5D7A, red #C44A57, deep ink #15171C.
"""
from pathlib import Path

import streamlit as st

ICON = Path(__file__).resolve().parents[1] / "assets" / "viper_icon.png"

SLATE, RED, INK = "#4C5D7A", "#C44A57", "#15171C"
ACCENT = f"linear-gradient(135deg, {SLATE} 0%, #6B5274 55%, {RED} 100%)"   # slate -> plum -> red
GLASS = """
  background: rgba(255, 255, 255, 0.38);
  backdrop-filter: blur(18px) saturate(150%); -webkit-backdrop-filter: blur(18px) saturate(150%);
  border: 1px solid rgba(255, 255, 255, 0.6);
  box-shadow: 0 8px 30px rgba(76, 93, 122, 0.14), inset 0 1px 0 rgba(255, 255, 255, 0.7);
"""

CSS = f"""
<style>
.stApp {{
  background:
    radial-gradient(36rem 36rem at 0% 0%, rgba(196, 74, 87, 0.42), transparent 60%),
    radial-gradient(40rem 40rem at 100% 25%, rgba(76, 93, 122, 0.50), transparent 60%),
    radial-gradient(34rem 34rem at 5% 75%, rgba(76, 93, 122, 0.38), transparent 60%),
    radial-gradient(30rem 30rem at 95% 100%, rgba(196, 74, 87, 0.34), transparent 60%),
    #EEF0F3;
  background-attachment: fixed;
}}
.block-container {{ max-width: 480px; }}

/* frosted glass cards */
[class*="st-key-card"] {{ {GLASS} border-radius: 24px; padding: 16px 16px 12px 16px; }}
[class*="st-key-card"] img, .st-key-hero img {{ border-radius: 18px; }}
.st-key-hero {{ {GLASS} border-radius: 26px; padding: 8px; }}
[data-testid="stExpander"] details {{ {GLASS} border-radius: 20px; }}
[data-testid="stFileUploaderDropzone"] {{ background: rgba(255, 255, 255, 0.35); border-radius: 18px; }}

/* scan card: glass with a slate-to-red tint */
.st-key-scancard {{
  background: linear-gradient(135deg, rgba(76, 93, 122, 0.78), rgba(107, 82, 116, 0.70) 55%, rgba(196, 74, 87, 0.62));
  backdrop-filter: blur(18px) saturate(150%); -webkit-backdrop-filter: blur(18px) saturate(150%);
  border: 1px solid rgba(255, 255, 255, 0.35); border-radius: 28px; padding: 20px;
  box-shadow: 0 12px 34px rgba(76, 93, 122, 0.30);
}}
.st-key-scancard p, .st-key-scancard h3 {{ color: #FFFFFF; }}
.st-key-scancard button[kind], .st-key-scancard [data-testid^="stBaseButton"] {{
  background: rgba(255, 255, 255, 0.94) !important; border: none !important;
  box-shadow: 0 4px 14px rgba(21, 23, 28, 0.18);
}}
.st-key-scancard button, .st-key-scancard button p, .st-key-scancard button span {{ color: {INK} !important; }}

/* accent: primary buttons and the app name use the logo gradient */
button[kind="primary"], [data-testid="stBaseButton-primary"] {{
  background: {ACCENT} !important; border: none !important;
  box-shadow: 0 4px 14px rgba(196, 74, 87, 0.25);
}}
button[kind="secondary"], [data-testid="stBaseButton-secondary"] {{
  background: rgba(255, 255, 255, 0.55) !important; border: 1px solid rgba(255, 255, 255, 0.7) !important;
}}
.st-key-brand h3 {{
  background: {ACCENT}; -webkit-background-clip: text; background-clip: text; color: transparent;
  letter-spacing: 0.04em;
}}

/* floating frosted bottom bar */
.st-key-bottomnav {{
  position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%);
  width: min(420px, calc(100% - 32px)); z-index: 1000;
  background: rgba(21, 23, 28, 0.62);
  backdrop-filter: blur(20px) saturate(160%); -webkit-backdrop-filter: blur(20px) saturate(160%);
  border: 1px solid rgba(255, 255, 255, 0.16); border-radius: 999px; padding: 8px 12px;
  box-shadow: 0 10px 30px rgba(21, 23, 28, 0.28);
}}
.st-key-bottomnav a {{ border-radius: 999px; justify-content: center; }}
.st-key-bottomnav a, .st-key-bottomnav a p, .st-key-bottomnav a span {{ color: #FFFFFF !important; }}
</style>
"""


def apply() -> None:
    st.html(CSS)


def header(subtitle: str) -> None:
    """Logo + app name (in the logo gradient), then a one-sentence explainer for the page."""
    with st.container(key="brand", horizontal=True, vertical_alignment="center", gap="small"):
        st.image(str(ICON), width=52)
        st.markdown("### VIPER")
    st.caption(subtitle)


def bottom_nav(pages: list) -> None:
    """Floating frosted pill bar with one link per page (like a phone app's tab bar)."""
    with st.container(key="bottomnav", horizontal=True, horizontal_alignment="distribute"):
        for page in pages:
            st.page_link(page)
