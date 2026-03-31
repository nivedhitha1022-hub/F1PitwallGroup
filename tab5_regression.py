"""
tab5_regression.py  —  Regression Analysis
Linear, Ridge & Lasso regression predicting Lifetime Revenue (LTV)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from theme import (
    base_layout, section_label, insight_box, rec_box, hex_to_rgba,
    F1_RED, F1_BLACK, F1_SILVER, F1_GOLD, ACCENT_TEAL, ACCENT_GREEN,
    ACCENT_AMBER, CARBON,
)
from model_utils import engineer_features, train_regression_models, REG_FEATURE_LABELS


# Model display config
MODEL_COLORS = {
    "Linear Regression": F1_RED,
    "Ridge (alpha=10)":  ACCENT_TEAL,
    "Lasso (alpha=1)":   F1_GOLD,
}


@st.cache_data(show_spinner=False)
def _run_regression(subs: pd.DataFrame, sess: pd.DataFrame) -> dict:
    df = engineer_features(subs, sess)
    return train_regression_models(df)


def render(subs: pd.DataFrame, sess: pd.DataFrame, mrr: pd.DataFrame) -> None:
    st.markdown("## Regression — *What Predicts Subscriber Lifetime Revenue?*")
    st.markdown(
        "Linear, Ridge, and Lasso regression models predicting **Lifetime Revenue (USD)** "
        "from subscriber behaviour, plan, and engagement features. "
        "Regularisation comparison shows how Ridge and Lasso improve on vanilla OLS."
    )
    st.markdown("---")

    with st.spinner("Training regression models…"):
        results = _run_regression(subs, sess)

    meta   = results["_meta"]
    models = {k: v for k, v in results.items() if k != "_meta"}

    # ── Model scorecard ───────────────────────────────────────────────────────
    st.markdown(section_label("MODEL PERFORMANCE SUMMARY"), unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (name, res) in zip(cols, models.items()):
        col.metric(f"{name} — R²",   f"{res['r2']:.3f}")
        col.metric(f"{name} — RMSE", f"${res['rmse']:,.2f}")
        col.metric(f"{name} — MAE",  f"${res['mae']:,.2f}")
    st.markdown("---")

    # ── Full comparison table ──────────────────────────────────────────────────
    st.markdown(section_label("FULL METRICS COMPARISON"), unsafe_allow_html=True)
    comp = pd.DataFrame([
        {"Model": name, "R²": res["r2"], "RMSE ($)": res["rmse"], "MAE ($)": res["mae"]}
        for name, res in models.items()
    ])
    st.dataframe(comp, use_container_width=True, hide_index=True)
    st.markdown("---")

    # ── R² and RMSE bar comparison ────────────────────────────────────────────
    col1, col2 = st.columns(2)
    names  = list(models.keys())
    colors = [MODEL_COLORS.get(n, ACCENT_TEAL) for n in names]

    with col1:
        st.markdown(section_label("R² SCORE COMPARISON"), unsafe_allow_html=True)
        r2s = [models[n]["r2"] for n in names]
        fig = go.Figure(go.Bar(
            x=names, y=r2s,
            marker=dict(color=colors, line=dict(color="white", width=1)),
            text=[f"{v:.3f}" for v in r2s],
            textposition="outside",
            textfont=dict(color=F1_BLACK, size=12),
            hovertemplate="<b>%{x}</b><br>R² = %{y:.3f}<extra></extra>",
        ))
        lo = base_layout("R² Score — Higher is Better", height=320)
        lo["yaxis"]["title"] = "R² Score"
        lo["yaxis"]["range"] = [0, max(r2s) * 1.3]
        fig.update_layout(**lo)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(section_label("RMSE COMPARISON"), unsafe_allow_html=True)
        rmses = [models[n]["rmse"] for n in names]
        fig2  = go.Figure(go.Bar(
            x=names, y=rmses,
            marker=dict(color=colors, line=dict(color="white", width=1)),
            text=[f"${v:,.0f}" for v in rmses],
            textposition="outside",
            textfont=dict(color=F1_BLACK, size=12),
            hovertemplate="<b>%{x}</b><br>RMSE = $%{y:,.0f}<extra></extra>",
        ))
        lo2 = base_layout("RMSE — Lower is Better", height=320)
        lo2["yaxis"]["title"] = "RMSE (USD)"
        lo2["yaxis"]["range"] = [0, max(rmses) * 1.3]
        fig2.update_layout(**lo2)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Actual vs Predicted — all 3 models ────────────────────────────────────
    st.markdown("---")
    st.markdown(section_label("ACTUAL vs PREDICTED LIFETIME REVENUE — ALL MODELS"), unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)
    for col, (name, res), color in zip([col3, col4, col5], models.items(), colors):
        with col:
            y_test = res["y_test"]
            y_pred = res["y_pred"]
            lim    = max(float(np.max(y_test)), float(np.max(y_pred))) * 1.05
            fig_av = go.Figure()
            fig_av.add_trace(go.Scatter(
                x=y_test, y=y_pred, mode="markers",
                marker=dict(color=color, opacity=0.45, size=5),
                name="Predictions",
                hovertemplate="Actual: $%{x:,.0f}<br>Predicted: $%{y:,.0f}<extra></extra>",
            ))
            fig_av.add_trace(go.Scatter(
                x=[0, lim], y=[0, lim], mode="lines",
                line=dict(color=F1_SILVER, dash="dash", width=1.5),
                name="Perfect Fit",
            ))
            lo_av = base_layout(f"{name}<br>R²={res['r2']:.3f}", height=320)
            lo_av["xaxis"]["title"]  = "Actual LTV ($)"
            lo_av["yaxis"]["title"]  = "Predicted LTV ($)"
            lo_av["showlegend"]      = False
            fig_av.update_layout(**lo_av)
            st.plotly_chart(fig_av, use_container_width=True)

    # ── Residuals distribution ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(section_label("RESIDUALS DISTRIBUTION — ALL MODELS"), unsafe_allow_html=True)
    col6, col7, col8 = st.columns(3)
    for col, (name, res), color in zip([col6, col7, col8], models.items(), colors):
        with col:
            residuals = res["residuals"]
            fig_res = go.Figure()
            fig_res.add_trace(go.Histogram(
                x=residuals, nbinsx=35,
                marker=dict(color=color, opacity=0.75,
                            line=dict(color="white", width=0.5)),
                name="Residuals",
                hovertemplate="Residual: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
            ))
            fig_res.add_vline(x=0, line=dict(color=F1_RED, dash="dash", width=2))
            lo_res = base_layout(f"{name} — Residuals", height=280)
            lo_res["xaxis"]["title"] = "Residual ($)"
            lo_res["yaxis"]["title"] = "Count"
            lo_res["showlegend"]     = False
            fig_res.update_layout(**lo_res)
            st.plotly_chart(fig_res, use_container_width=True)

    # ── Coefficient comparison ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(section_label("STANDARDISED COEFFICIENTS — LINEAR vs RIDGE vs LASSO"), unsafe_allow_html=True)
    st.markdown(
        "Shows how regularisation (Ridge/Lasso) shrinks and eliminates coefficients "
        "compared to plain Linear Regression. Lasso drives some to exactly 0."
    )

    col9, col10, col11 = st.columns(3)
    for col, (name, res), color in zip([col9, col10, col11], models.items(), colors):
        with col:
            coefs = res["coefs"].sort_values()
            # For Lasso: show only non-zero
            show  = coefs[coefs.abs() > 0.001] if "Lasso" in name else coefs
            bar_colors = [color if v >= 0 else F1_RED for v in show.values]

            fig_cf = go.Figure(go.Bar(
                y=show.index, x=show.values,
                orientation="h",
                marker=dict(color=bar_colors, line=dict(color="white", width=0.5)),
                text=[f"{v:+.2f}" for v in show.values],
                textposition="outside",
                textfont=dict(size=9, color=F1_BLACK),
                hovertemplate="<b>%{y}</b><br>Coefficient: %{x:+.3f}<extra></extra>",
            ))
            n_zero = int((coefs.abs() <= 0.001).sum()) if "Lasso" in name else 0
            title  = f"{name}" + (f"<br><span style='font-size:11px;color:{F1_SILVER}'>{n_zero} features zeroed out</span>" if n_zero > 0 else "")
            lo_cf = base_layout(title, height=400)
            lo_cf["xaxis"]["title"] = "Coefficient"
            lo_cf["margin"]["r"]    = 70
            fig_cf.update_layout(**lo_cf)
            st.plotly_chart(fig_cf, use_container_width=True)

    # Lasso sparsity callout
    lasso_coefs = models["Lasso (alpha=1)"]["coefs"]
    n_zero_lasso = int((lasso_coefs.abs() <= 0.001).sum())
    if n_zero_lasso > 0:
        st.markdown(insight_box(
            f"<b>Lasso feature selection:</b> {n_zero_lasso} of {len(lasso_coefs)} features "
            f"were driven to exactly zero by L1 regularisation — automatic feature elimination. "
            f"This makes Lasso more interpretable in high-dimensional subscriber data."
        ), unsafe_allow_html=True)

    # ── Coefficient comparison heatmap ────────────────────────────────────────
    st.markdown("---")
    st.markdown(section_label("COEFFICIENT HEATMAP — REGULARISATION EFFECT"), unsafe_allow_html=True)
    coef_df = pd.DataFrame({
        name: res["coefs"] for name, res in models.items()
    }).fillna(0)

    fig_heat = go.Figure(go.Heatmap(
        z=coef_df.values,
        x=coef_df.columns.tolist(),
        y=coef_df.index.tolist(),
        colorscale=[[0, F1_RED], [0.5, "#F8F8F8"], [1, ACCENT_GREEN]],
        zmid=0,
        text=coef_df.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=10, color=F1_BLACK),
        hovertemplate="<b>%{y} — %{x}</b><br>Coefficient: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Coefficient", tickfont=dict(color=F1_SILVER)),
    ))
    lo_heat = base_layout(
        "Coefficient Heatmap — How Regularisation Shrinks Features", height=440
    )
    fig_heat.update_layout(**lo_heat)
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Business insights ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(section_label("KEY REGRESSION INSIGHTS"), unsafe_allow_html=True)

    # Find top positive predictor from linear regression
    lr_coefs   = models["Linear Regression"]["coefs"]
    top_pos    = lr_coefs.idxmax()
    top_neg    = lr_coefs.idxmin()
    best_model = max(models.items(), key=lambda x: x[1]["r2"])

    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown(insight_box(
            f"📈 <b>{top_pos}</b> is the strongest positive predictor of lifetime revenue. "
            f"Subscribers with higher values on this feature generate disproportionately "
            f"more revenue over their lifecycle — prioritise this segment in upsell campaigns."
        ), unsafe_allow_html=True)
    with i2:
        st.markdown(insight_box(
            f"🏆 <b>{best_model[0]}</b> achieves the best R² ({best_model[1]['r2']:.3f}) "
            f"and lowest RMSE (${best_model[1]['rmse']:,.0f}). "
            f"The relatively modest R² reflects genuine noise in subscriber behaviour — "
            f"LTV is influenced by external factors beyond platform engagement alone."
        ), unsafe_allow_html=True)
    with i3:
        st.markdown(rec_box(
            f"🔧 <b>Lasso's sparsity is operationally valuable:</b> by eliminating "
            f"{n_zero_lasso} low-signal features, it produces a leaner model that is "
            f"easier to explain to stakeholders and less prone to overfitting on "
            f"future subscriber cohorts with different demographic mixes."
        ), unsafe_allow_html=True)
