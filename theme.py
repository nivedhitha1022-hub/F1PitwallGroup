# theme.py  —  PitWall Analytics  ·  F1 Light & Premium Design System

# ── Core Palette ──────────────────────────────────────────────────────────────
F1_RED      = "#E8002D"       # F1 official red
F1_BLACK    = "#1A1A1A"       # near-black text
F1_WHITE    = "#FFFFFF"       # pure white cards
F1_SILVER   = "#6B6B6B"       # medium grey text
F1_GOLD     = "#D4A843"       # champagne gold
F1_GREY     = "#F0F0F0"       # light card bg
F1_DGREY    = "#E5E5E5"       # border / divider

CARBON      = "#2C2C2C"       # carbon fibre dark
LIGHT_BG    = "#F8F8F8"       # page background
CARD_BG     = "#FFFFFF"       # card background
BORDER      = "#E5E5E5"       # card border

ACCENT_TEAL   = "#0A9396"
ACCENT_GREEN  = "#2D9E5F"
ACCENT_AMBER  = "#D4A843"
ACCENT_PURPLE = "#7B5EA7"

# ── Semantic colour mappings ───────────────────────────────────────────────────
PLAN_COLORS = {
    "Pit Lane":     "#6B6B6B",
    "Podium":       F1_RED,
    "Paddock Club": F1_GOLD,
}

CHANNEL_COLORS = {
    "Paid Ad":      F1_RED,
    "Organic":      ACCENT_GREEN,
    "Social Media": ACCENT_AMBER,
    "Referral":     ACCENT_TEAL,
}

NPS_COLORS = {
    "Promoter":  ACCENT_GREEN,
    "Passive":   ACCENT_AMBER,
    "Detractor": F1_RED,
}

CHURN_COLORS = {
    "Active":  ACCENT_GREEN,
    "Churned": F1_RED,
}

RISK_COLORS = {
    "Low Risk":    ACCENT_GREEN,
    "Medium Risk": ACCENT_AMBER,
    "High Risk":   F1_RED,
}

SEGMENT_COLORS = {
    "Champions": F1_GOLD,
    "Engaged":   ACCENT_GREEN,
    "At Risk":   ACCENT_AMBER,
    "Dormant":   F1_RED,
}

CLASSIFIER_COLORS = {
    "Random Forest":     F1_RED,
    "Logistic Reg.":     ACCENT_TEAL,
    "Decision Tree":     F1_GOLD,
    "KNN":               ACCENT_GREEN,
    "Naive Bayes":       ACCENT_PURPLE,
    "SVM":               CARBON,
}


def hex_to_rgba(hex_color: str, alpha: float = 0.25) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def base_layout(title: str = "", height: int = 400) -> dict:
    return dict(
        title=dict(
            text=title,
            font=dict(color=F1_BLACK, size=14, family="Arial Black, sans-serif"),
            x=0.01,
            pad=dict(b=4),
        ),
        paper_bgcolor=CARD_BG,
        plot_bgcolor="#FAFAFA",
        font=dict(color=F1_BLACK, family="Arial, sans-serif", size=12),
        height=height,
        margin=dict(l=48, r=24, t=52, b=44),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(color=F1_BLACK, size=11),
        ),
        xaxis=dict(
            gridcolor="#EBEBEB",
            linecolor=BORDER,
            zerolinecolor=BORDER,
            tickfont=dict(color=F1_SILVER),
        ),
        yaxis=dict(
            gridcolor="#EBEBEB",
            linecolor=BORDER,
            zerolinecolor=BORDER,
            tickfont=dict(color=F1_SILVER),
        ),
        hoverlabel=dict(
            bgcolor=CARD_BG,
            bordercolor=BORDER,
            font=dict(color=F1_BLACK, size=12),
        ),
    )


F1_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Titillium Web', Arial, sans-serif !important;
    background-color: #F8F8F8 !important;
    color: #1A1A1A !important;
}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main, .main .block-container,
[data-testid="stTabsContent"] {
    background-color: #F8F8F8 !important;
}
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 2px solid #E8002D;
}

/* ── Tab nav ── */
[data-testid="stTabs"] > div:first-child {
    border-bottom: 2px solid #E5E5E5;
    background: #FFFFFF;
    border-radius: 8px 8px 0 0;
    padding: 0 8px;
}
[data-testid="stTabs"] button {
    font-family: 'Titillium Web', sans-serif !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    letter-spacing: 1.2px !important;
    color: #9B9B9B !important;
    padding: 12px 22px !important;
    border-bottom: 3px solid transparent !important;
    background: transparent !important;
    text-transform: uppercase;
    transition: color 0.2s;
}
[data-testid="stTabs"] button:hover { color: #2C2C2C !important; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #E8002D !important;
    border-bottom: 3px solid #E8002D !important;
}
[data-testid="stTabsContent"] { padding-top: 1.5rem; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E5E5 !important;
    border-top: 4px solid #E8002D !important;
    border-radius: 8px !important;
    padding: 16px 20px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}
[data-testid="metric-container"] label {
    color: #9B9B9B !important;
    font-size: 10px !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
}
[data-testid="stMetricValue"] {
    color: #1A1A1A !important;
    font-size: 26px !important;
    font-weight: 900 !important;
}
[data-testid="stMetricDelta"] span { font-size: 11px !important; }

/* ── Typography ── */
h1 {
    color: #E8002D !important;
    font-weight: 900 !important;
    letter-spacing: 3px !important;
}
h2 {
    color: #1A1A1A !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #E8002D;
    padding-bottom: 8px;
    margin-top: 0.5rem;
}
h3 { color: #6B6B6B !important; font-weight: 600 !important; font-size: 0.85rem !important; }
hr { border-color: #E5E5E5 !important; margin: 1.4rem 0 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #E5E5E5;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

/* ── Insight / Rec / Warn boxes ── */
.insight-box {
    background: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-left: 4px solid #E8002D;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 8px 0 14px 0;
    font-size: 13px;
    line-height: 1.75;
    color: #2C2C2C;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.insight-box b { color: #E8002D; }

.rec-box {
    background: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-left: 4px solid #2D9E5F;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 8px 0 14px 0;
    font-size: 13px;
    line-height: 1.75;
    color: #2C2C2C;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.rec-box b { color: #2D9E5F; }

.warn-box {
    background: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-left: 4px solid #D4A843;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 8px 0 14px 0;
    font-size: 13px;
    line-height: 1.75;
    color: #2C2C2C;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.warn-box b { color: #D4A843; }

/* ── Section label ── */
.section-label {
    font-size: 9px;
    letter-spacing: 3.5px;
    text-transform: uppercase;
    color: #E8002D;
    font-weight: 700;
    margin-bottom: 6px;
    font-family: 'Titillium Web', sans-serif;
}

/* ── Chart card wrapper ── */
.chart-card {
    background: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 12px;
}

/* ── Model comparison table ── */
.model-compare-table {
    background: #FFFFFF;
    border-radius: 8px;
    border: 1px solid #E5E5E5;
    overflow: hidden;
}

#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stHeader"] { background-color: #F8F8F8 !important; }
</style>
"""


def section_label(text: str) -> str:
    return f'<div class="section-label">{text}</div>'

def insight_box(html: str) -> str:
    return f'<div class="insight-box">{html}</div>'

def rec_box(html: str) -> str:
    return f'<div class="rec-box">{html}</div>'

def warn_box(html: str) -> str:
    return f'<div class="warn-box">{html}</div>'
