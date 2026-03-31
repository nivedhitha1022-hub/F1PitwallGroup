"""
app.py  —  PitWall Analytics Dashboard  (Group Edition)
────────────────────────────────────────────────────────
Run with:   streamlit run app.py

Project structure
    app.py
    data_generator.py
    model_utils.py
    theme.py
    tab1_descriptive.py
    tab2_diagnostic.py
    tab3_predictive.py
    tab4_prescriptive.py
    tab5_regression.py
    data/
        PitWall_Analytics_Cleaned.xlsx
    .streamlit/
        config.toml
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent
for p in [str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

st.set_page_config(
    page_title="PitWall Analytics",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from theme          import F1_CSS
from data_generator import load_data
import tab1_descriptive
import tab2_diagnostic
import tab3_predictive
import tab4_prescriptive
import tab5_regression

# ── Inject global CSS ──────────────────────────────────────────────────────────
st.markdown(F1_CSS, unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load():
    return load_data()

with st.spinner("🏎  Loading race data…"):
    subs, sess, mrr = _load()

# ── Dashboard header — Light & Premium ────────────────────────────────────────
st.markdown(
    """
    <div style="
        background: #FFFFFF;
        border-bottom: 3px solid #E8002D;
        border-top: 1px solid #E5E5E5;
        padding: 20px 28px 16px 28px;
        margin: -1rem -1rem 1.5rem -1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    ">
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:6px;">
            <div style="
                background: #E8002D;
                color: white;
                font-size: 8px;
                font-weight: 800;
                letter-spacing: 3px;
                text-transform: uppercase;
                padding: 4px 10px;
                border-radius: 3px;
                font-family: 'Titillium Web', Arial, sans-serif;
            ">F1 ANALYTICS</div>
            <div style="
                font-size: 8px; letter-spacing: 3.5px; color: #9B9B9B; font-weight: 700;
                text-transform: uppercase; font-family: 'Titillium Web', Arial, sans-serif;
            ">SUBSCRIBER RETENTION INTELLIGENCE</div>
        </div>
        <div style="
            font-size: 30px; font-weight: 900; color: #1A1A1A; letter-spacing: 2px;
            font-family: 'Titillium Web', Arial Black, sans-serif; margin-bottom: 4px;
        ">
            🏎&nbsp; PITWALL ANALYTICS
        </div>
        <div style="
            font-size: 12px; color: #9B9B9B; margin-top: 4px;
            font-family: 'Titillium Web', Arial, sans-serif; letter-spacing: 0.5px;
        ">
            800 Subscribers &nbsp;·&nbsp; 29,240 Sessions &nbsp;·&nbsp;
            3 Plan Tiers &nbsp;·&nbsp; Seasons 2023 – 2024 &nbsp;·&nbsp;
            <span style="color:#E8002D;font-weight:700;">GROUP EDITION</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tab navigation ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋  Descriptive",
    "🔍  Diagnostic",
    "🔮  Predictive",
    "🎯  Prescriptive",
    "📈  Regression",
])

with tab1:
    tab1_descriptive.render(subs, sess, mrr)

with tab2:
    tab2_diagnostic.render(subs, sess, mrr)

with tab3:
    tab3_predictive.render(subs, sess, mrr)

with tab4:
    tab4_prescriptive.render(subs, sess, mrr)

with tab5:
    tab5_regression.render(subs, sess, mrr)
