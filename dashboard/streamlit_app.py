"""Entry point of the research dashboard: ``streamlit run dashboard/streamlit_app.py`` (see docs/dashboard.md)."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "src"):  # works from a clean clone without installing the package
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import sections  # noqa: E402

from tennis_data.dashboard import data as D  # noqa: E402

st.set_page_config(page_title="ATP prediction research", page_icon=":material/sports_tennis:", layout="wide",
                   initial_sidebar_state="auto")  # fmt: skip
st.markdown("""<style>
[data-testid="stMetricValue"] { font-size: 1.55rem; }
[data-testid="stMetricLabel"] p { font-size: 0.85rem; color: #52606d; }
.block-container { padding-top: 2rem; max-width: 1280px; }
h1 { font-size: 1.9rem !important; letter-spacing: -0.01em; }
h2, h3 { margin-top: 1.4rem !important; }
</style>""", unsafe_allow_html=True)  # fmt: skip

with st.sidebar:
    st.markdown("### ATP prediction research")
    st.caption("Frozen M4A model · historical evaluation")
    section = st.radio("Section", list(sections.SECTIONS), key="section", label_visibility="collapsed")
    st.divider()
    g = D.gates()
    st.markdown(f"**Live predictions:** {'enabled' if g.live_predictions_available else 'unavailable'}  \n"
                f"Gate `{g.prospective_predictions}` · backfill `{g.backfill}`")  # fmt: skip
    man = sections.manifest()
    if man:
        st.caption(f"Model `{man['model']['frozen_hash']}` · config `{man['model']['configuration_hash']}`  \n"
                   f"Dataset built {man['built_at'][:10]}")  # fmt: skip
    else:
        st.caption("Public dashboard dataset not found: aggregate mode.")
    st.caption("Data: Jeff Sackmann, CC BY-NC-SA 4.0  \nResearch only, not betting advice.")

try:
    sections.SECTIONS[section]()
except Exception as exc:  # a failing view must not take the whole site down
    st.error("This section could not be displayed. The other sections are unaffected.", icon=":material/error:")
    with st.expander("Technical details"):
        st.exception(exc)
