"""
tab3_predictive.py  —  Predictive Analytics
All 6 classifiers + comparison + enhanced KMeans segmentation
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from theme import (
    base_layout, section_label, insight_box, warn_box, hex_to_rgba,
    F1_RED, F1_WHITE, F1_SILVER, F1_GREY, F1_DGREY, F1_GOLD, F1_BLACK,
    ACCENT_TEAL, ACCENT_GREEN, ACCENT_AMBER, CARBON,
    PLAN_COLORS, RISK_COLORS, SEGMENT_COLORS, CLASSIFIER_COLORS,
)
from model_utils import (
    engineer_features, train_churn_model, get_model_metrics,
    train_all_classifiers, segment_customers, get_kmeans_elbow,
    FEATURE_LABELS,
)


@st.cache_data(show_spinner=False)
def _run_pipeline(subs: pd.DataFrame, sess: pd.DataFrame):
    df       = engineer_features(subs, sess)
    out      = train_churn_model(df)
    clf, X_tr, X_te, y_tr, y_te, y_pred, y_prob, imp_df, df_sc = out
    metrics  = get_model_metrics(y_te, y_pred, y_prob)
    df_sc    = segment_customers(df_sc)
    all_clf  = train_all_classifiers(df)
    k_range, inertias = get_kmeans_elbow(df_sc)
    return df_sc, metrics, imp_df, y_te, y_pred, y_prob, all_clf, k_range, inertias


def render(subs: pd.DataFrame, sess: pd.DataFrame, mrr: pd.DataFrame) -> None:
    st.markdown("## Predictive — *Who Will Churn Next?*")
    st.markdown(
        "Six classification algorithms benchmarked · feature importance · "
        "risk scoring · KMeans behavioural segmentation with elbow analysis."
    )
    st.markdown("---")

    with st.spinner("Training all classification models…"):
        df, metrics, imp_df, y_te, y_pred, y_prob, all_clf, k_range, inertias = \
            _run_pipeline(subs, sess)

    # ── Sub-tabs ──────────────────────────────────────────────────────────────
    sub1, sub2, sub3 = st.tabs([
        "🏆 Classifier Comparison",
        "🌲 Random Forest Deep-Dive",
        "🔵 KMeans Segmentation",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB 1 — ALL CLASSIFIERS
    # ══════════════════════════════════════════════════════════════════════════
    with sub1:
        st.markdown("### All Classification Algorithms — Performance Comparison")
        st.markdown(
            "Same 75/25 stratified train-test split across all models. "
            "Target: **Churn (Yes/No)**."
        )
        st.markdown("---")

        # ── Comparison table ──────────────────────────────────────────────────
        st.markdown(section_label("PERFORMANCE METRICS — ALL 6 CLASSIFIERS"), unsafe_allow_html=True)

        rows = []
        for name, res in all_clf.items():
            rows.append({
                "Model":     name,
                "Accuracy":  f"{res['accuracy']:.1%}",
                "Precision": f"{res['precision']:.1%}",
                "Recall":    f"{res['recall']:.1%}",
                "F1 Score":  f"{res['f1']:.1%}",
                "ROC-AUC":   f"{res['auc']:.3f}",
            })
        compare_df = pd.DataFrame(rows)
        st.dataframe(compare_df, use_container_width=True, hide_index=True)
        st.markdown("---")

        # ── Grouped bar: metrics comparison ───────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(section_label("ACCURACY & F1 BY MODEL"), unsafe_allow_html=True)
            names  = list(all_clf.keys())
            accs   = [all_clf[n]["accuracy"]  for n in names]
            f1s    = [all_clf[n]["f1"]         for n in names]
            colors = [CLASSIFIER_COLORS.get(n, ACCENT_TEAL) for n in names]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Accuracy", x=names, y=accs,
                marker_color=colors, opacity=0.9,
                text=[f"{v:.1%}" for v in accs], textposition="outside",
                textfont=dict(size=10, color=F1_BLACK),
            ))
            fig.add_trace(go.Bar(
                name="F1 Score", x=names, y=f1s,
                marker_color=colors, opacity=0.5,
                text=[f"{v:.1%}" for v in f1s], textposition="outside",
                textfont=dict(size=10, color=F1_BLACK),
            ))
            lo = base_layout("Accuracy vs F1 Score by Classifier", height=380)
            lo["barmode"]        = "group"
            lo["yaxis"]["title"] = "Score"
            lo["yaxis"]["range"] = [0, 1.15]
            lo["yaxis"]["tickformat"] = ".0%"
            fig.update_layout(**lo)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(section_label("PRECISION vs RECALL TRADE-OFF"), unsafe_allow_html=True)
            precs  = [all_clf[n]["precision"] for n in names]
            recs   = [all_clf[n]["recall"]    for n in names]
            aucs   = [all_clf[n]["auc"]       for n in names]

            fig2 = go.Figure()
            for i, name in enumerate(names):
                fig2.add_trace(go.Scatter(
                    x=[recs[i]], y=[precs[i]],
                    mode="markers+text",
                    name=name,
                    text=[name],
                    textposition="top center",
                    textfont=dict(size=10, color=F1_BLACK),
                    marker=dict(
                        size=aucs[i] * 40,
                        color=CLASSIFIER_COLORS.get(name, ACCENT_TEAL),
                        line=dict(color="white", width=2),
                        opacity=0.85,
                    ),
                    hovertemplate=(
                        f"<b>{name}</b><br>Precision: {precs[i]:.1%}<br>"
                        f"Recall: {recs[i]:.1%}<br>AUC: {aucs[i]:.3f}<extra></extra>"
                    ),
                ))
            lo2 = base_layout("Precision vs Recall  (bubble size = ROC-AUC)", height=380)
            lo2["xaxis"]["title"]        = "Recall"
            lo2["yaxis"]["title"]        = "Precision"
            lo2["xaxis"]["range"]        = [0, 1.1]
            lo2["yaxis"]["range"]        = [0, 1.1]
            lo2["xaxis"]["tickformat"]   = ".0%"
            lo2["yaxis"]["tickformat"]   = ".0%"
            lo2["showlegend"]            = False
            fig2.update_layout(**lo2)
            st.plotly_chart(fig2, use_container_width=True)

        # ── ROC curve overlay ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(section_label("ROC CURVES — ALL CLASSIFIERS OVERLAID"), unsafe_allow_html=True)
        fig_roc = go.Figure()
        for name, res in all_clf.items():
            fig_roc.add_trace(go.Scatter(
                x=res["fpr"], y=res["tpr"],
                mode="lines",
                name=f"{name}  (AUC={res['auc']:.3f})",
                line=dict(color=CLASSIFIER_COLORS.get(name, ACCENT_TEAL), width=2.5),
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline",
            line=dict(color=F1_SILVER, width=1.5, dash="dash"),
        ))
        lo_roc = base_layout("ROC Curves — All Classifiers", height=440)
        lo_roc["xaxis"]["title"] = "False Positive Rate"
        lo_roc["yaxis"]["title"] = "True Positive Rate"
        fig_roc.update_layout(**lo_roc)
        st.plotly_chart(fig_roc, use_container_width=True)

        # ── AUC bar ────────────────────────────────────────────────────────────
        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            st.markdown(section_label("ROC-AUC RANKING"), unsafe_allow_html=True)
            auc_df = pd.DataFrame({
                "Model": names,
                "AUC":   aucs,
            }).sort_values("AUC", ascending=True)
            fig_auc = go.Figure(go.Bar(
                y=auc_df["Model"], x=auc_df["AUC"],
                orientation="h",
                marker=dict(
                    color=[CLASSIFIER_COLORS.get(n, ACCENT_TEAL) for n in auc_df["Model"]],
                    line=dict(color="white", width=1),
                ),
                text=[f"{v:.3f}" for v in auc_df["AUC"]],
                textposition="outside",
                textfont=dict(size=11, color=F1_BLACK),
            ))
            lo_auc = base_layout("AUC Ranking — Best to Worst", height=340)
            lo_auc["xaxis"]["title"] = "ROC-AUC"
            lo_auc["xaxis"]["range"] = [0, 1.15]
            fig_auc.update_layout(**lo_auc)
            st.plotly_chart(fig_auc, use_container_width=True)

        with col4:
            st.markdown(section_label("CONFUSION MATRICES — TOP 4 MODELS"), unsafe_allow_html=True)
            top4 = sorted(all_clf.items(), key=lambda x: x[1]["auc"], reverse=True)[:4]
            for i, (name, res) in enumerate(top4):
                cm = res["cm"]
                st.markdown(f"**{name}**  — AUC: {res['auc']:.3f} | Acc: {res['accuracy']:.1%}")
                fig_cm = go.Figure(go.Heatmap(
                    z=cm,
                    x=["Pred: Active", "Pred: Churned"],
                    y=["Actual: Active", "Actual: Churned"],
                    colorscale=[[0, "#F0F0F0"], [1, F1_RED]],
                    text=cm, texttemplate="<b>%{text}</b>",
                    textfont=dict(size=18, color=F1_BLACK),
                    showscale=False,
                    hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
                ))
                lo_cm = base_layout("", height=200)
                lo_cm["margin"] = dict(l=40, r=10, t=10, b=40)
                fig_cm.update_layout(**lo_cm)
                st.plotly_chart(fig_cm, use_container_width=True)

        # ── Insights ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(section_label("KEY CLASSIFICATION INSIGHTS"), unsafe_allow_html=True)
        best_model  = max(all_clf.items(), key=lambda x: x[1]["auc"])
        best_f1     = max(all_clf.items(), key=lambda x: x[1]["f1"])
        best_recall = max(all_clf.items(), key=lambda x: x[1]["recall"])

        i1, i2, i3 = st.columns(3)
        with i1:
            st.markdown(insight_box(
                f"🏆 <b>{best_model[0]}</b> achieves the highest ROC-AUC "
                f"({best_model[1]['auc']:.3f}), making it the best overall discriminator "
                f"between churners and retainers across all probability thresholds."
            ), unsafe_allow_html=True)
        with i2:
            st.markdown(insight_box(
                f"⚖️ <b>{best_f1[0]}</b> leads on F1 Score ({best_f1[1]['f1']:.1%}), "
                f"balancing precision and recall — ideal when both false positives "
                f"(wasted offers) and false negatives (missed churners) carry business cost."
            ), unsafe_allow_html=True)
        with i3:
            st.markdown(insight_box(
                f"🔍 <b>{best_recall[0]}</b> has the highest Recall "
                f"({best_recall[1]['recall']:.1%}), catching the most actual churners. "
                f"Prefer this model when missing a churner is more costly than "
                f"offering an unnecessary discount."
            ), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB 2 — RANDOM FOREST DEEP DIVE
    # ══════════════════════════════════════════════════════════════════════════
    with sub2:
        st.markdown("### Random Forest — Full Deep-Dive")
        st.markdown("---")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy",  f"{metrics['accuracy']:.1%}")
        m2.metric("Precision", f"{metrics['precision']:.1%}")
        m3.metric("Recall",    f"{metrics['recall']:.1%}")
        m4.metric("F1 Score",  f"{metrics['f1']:.1%}")
        m5.metric("ROC-AUC",   f"{metrics['auc']:.3f}")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(section_label("FEATURE IMPORTANCE — WHAT DRIVES CHURN?"), unsafe_allow_html=True)
            med = float(imp_df["importance"].median())
            fig1 = go.Figure(go.Bar(
                y=imp_df["feature"], x=imp_df["importance"],
                orientation="h",
                marker=dict(
                    color=[F1_RED if v > med else F1_SILVER for v in imp_df["importance"]],
                    line=dict(color="white", width=0.5),
                ),
                text=[f"{v:.3f}" for v in imp_df["importance"]],
                textposition="outside",
                textfont=dict(color=F1_BLACK, size=10),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            ))
            lo1 = base_layout("Random Forest — Feature Importance", height=480)
            lo1["xaxis"]["title"] = "Gini Importance"
            lo1["margin"]["r"]    = 70
            fig1.update_layout(**lo1)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown(section_label("ROC CURVE"), unsafe_allow_html=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=metrics["fpr"], y=metrics["tpr"], mode="lines",
                name=f"RF  (AUC={metrics['auc']:.3f})",
                line=dict(color=F1_RED, width=3),
                fill="tozeroy", fillcolor=hex_to_rgba(F1_RED, 0.08),
            ))
            fig2.add_trace(go.Scatter(
                x=[0,1], y=[0,1], mode="lines", name="Random Baseline",
                line=dict(color=F1_SILVER, width=1.5, dash="dash"),
            ))
            lo2 = base_layout(f"ROC Curve  —  AUC = {metrics['auc']:.3f}", height=320)
            lo2["xaxis"]["title"] = "False Positive Rate"
            lo2["yaxis"]["title"] = "True Positive Rate"
            fig2.update_layout(**lo2)
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown(section_label("CONFUSION MATRIX"), unsafe_allow_html=True)
            cm = metrics["cm"]
            fig3 = go.Figure(go.Heatmap(
                z=cm, x=["Pred: Active","Pred: Churned"],
                y=["Actual: Active","Actual: Churned"],
                colorscale=[[0,"#F0F0F0"],[1,F1_RED]],
                text=cm, texttemplate="<b>%{text}</b>",
                textfont=dict(size=28, color=F1_BLACK), showscale=False,
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            ))
            lo3 = base_layout("Confusion Matrix", height=240)
            lo3["margin"] = dict(l=48, r=24, t=48, b=44)
            fig3.update_layout(**lo3)
            st.plotly_chart(fig3, use_container_width=True)

        # Risk distribution + churn prob by plan
        st.markdown("---")
        col5, col6 = st.columns(2)
        active = df[df["churn_flag"] == 0].copy()
        active["risk_label"] = pd.cut(
            active["churn_prob"], bins=[0,0.33,0.66,1.0],
            labels=["Low Risk","Medium Risk","High Risk"],
        )

        with col5:
            st.markdown(section_label("CHURN RISK DISTRIBUTION — ACTIVE SUBSCRIBERS"), unsafe_allow_html=True)
            rc = active["risk_label"].value_counts()
            fig4 = go.Figure(go.Pie(
                labels=rc.index.tolist(), values=rc.values.tolist(), hole=0.52,
                marker=dict(
                    colors=[RISK_COLORS.get(str(l), F1_SILVER) for l in rc.index],
                    line=dict(color="white", width=2),
                ),
                textinfo="label+percent+value",
                textfont=dict(color=F1_BLACK, size=12),
                hovertemplate="<b>%{label}</b><br>%{value} subs (%{percent})<extra></extra>",
            ))
            fig4.update_layout(**base_layout("Active Subscribers — Churn Risk Tier", height=320))
            fig4.add_annotation(
                text=f"<b>{len(active):,}</b><br>active",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color=F1_BLACK, size=14),
            )
            st.plotly_chart(fig4, use_container_width=True)

        with col6:
            st.markdown(section_label("CHURN PROBABILITY BY PLAN"), unsafe_allow_html=True)
            fig5 = go.Figure()
            for plan, color in PLAN_COLORS.items():
                vals = active[active["Plan"] == plan]["churn_prob"].dropna()
                fig5.add_trace(go.Violin(
                    y=vals, name=plan,
                    fillcolor=hex_to_rgba(color, 0.2),
                    line_color=color, meanline_visible=True, box_visible=True,
                ))
            lo5 = base_layout("Predicted Churn Probability by Plan", height=320)
            lo5["yaxis"]["title"]      = "Predicted Churn Probability"
            lo5["yaxis"]["tickformat"] = ".0%"
            fig5.update_layout(**lo5)
            st.plotly_chart(fig5, use_container_width=True)

        # High-risk watchlist
        st.markdown("---")
        st.markdown(section_label("HIGH-RISK WATCHLIST — TOP 30 ACTIVE SUBSCRIBERS"), unsafe_allow_html=True)
        watchlist = active[active["churn_prob"] >= 0.45].copy()
        watchlist["priority_score"] = (watchlist["churn_prob"] * watchlist["Monthly Price Usd"]).round(2)
        watchlist = watchlist.sort_values("priority_score", ascending=False).head(30)
        disp = watchlist[[
            "Subscriber Id","Plan","Region","Monthly Price Usd",
            "avg_engagement","avg_duration","churn_prob","priority_score",
        ]].copy()
        disp.columns = [
            "Subscriber","Plan","Region","Price ($)",
            "Avg Engagement","Avg Duration (min)","Churn Prob","Priority Score",
        ]
        disp["Churn Prob"]         = disp["Churn Prob"].map("{:.1%}".format)
        disp["Avg Engagement"]     = disp["Avg Engagement"].map("{:.1f}".format)
        disp["Avg Duration (min)"] = disp["Avg Duration (min)"].map("{:.1f}".format)
        disp["Priority Score"]     = disp["Priority Score"].map("{:.2f}".format)
        st.dataframe(disp, use_container_width=True, height=380)

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB 3 — KMEANS SEGMENTATION
    # ══════════════════════════════════════════════════════════════════════════
    with sub3:
        st.markdown("### KMeans Customer Segmentation — 4 Clusters")
        st.markdown(
            "Segmented on: avg engagement, avg session duration, total sessions, "
            "tenure, mobile %, high-engagement session %."
        )
        st.markdown("---")

        # Elbow chart
        st.markdown(section_label("ELBOW METHOD — OPTIMAL K SELECTION"), unsafe_allow_html=True)
        fig_elb = go.Figure()
        fig_elb.add_trace(go.Scatter(
            x=k_range, y=inertias, mode="lines+markers",
            line=dict(color=F1_RED, width=2.5),
            marker=dict(color=F1_RED, size=8, line=dict(color="white", width=2)),
            hovertemplate="k=%{x}<br>Inertia: %{y:,.0f}<extra></extra>",
        ))
        fig_elb.add_vline(x=4, line=dict(color=ACCENT_AMBER, dash="dash", width=2))
        fig_elb.add_annotation(
            x=4.15, y=max(inertias) * 0.9,
            text="k=4 selected", showarrow=False,
            font=dict(color=ACCENT_AMBER, size=12),
        )
        lo_elb = base_layout("Elbow Method — KMeans Inertia by k", height=320)
        lo_elb["xaxis"]["title"] = "Number of Clusters (k)"
        lo_elb["yaxis"]["title"] = "Inertia (Within-Cluster SSE)"
        fig_elb.update_layout(**lo_elb)
        st.plotly_chart(fig_elb, use_container_width=True)

        st.markdown("---")
        seg_sum = (
            df.groupby("segment_label")
            .agg(
                count      = ("Subscriber Id", "count"),
                avg_eng    = ("avg_engagement", "mean"),
                avg_dur    = ("avg_duration",   "mean"),
                avg_tenure = ("Tenure Months",  "mean"),
                churn_rate = ("churn_flag",      "mean"),
                avg_nps    = ("Nps Score",       "mean"),
            )
            .reset_index()
            .sort_values("avg_eng", ascending=False)
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(section_label("SEGMENT SIZES"), unsafe_allow_html=True)
            fig7 = go.Figure(go.Bar(
                x=seg_sum["count"], y=seg_sum["segment_label"],
                orientation="h",
                marker=dict(
                    color=[SEGMENT_COLORS.get(s, F1_SILVER) for s in seg_sum["segment_label"]],
                    line=dict(color="white", width=1),
                ),
                text=seg_sum["count"].tolist(),
                textposition="outside",
                textfont=dict(color=F1_BLACK),
                customdata=seg_sum[["avg_eng","churn_rate"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>Subscribers: %{x}<br>"
                    "Avg Engagement: %{customdata[0]:.1f}<br>"
                    "Churn Rate: %{customdata[1]:.1%}<extra></extra>"
                ),
            ))
            lo7 = base_layout("Segment Sizes", height=300)
            lo7["xaxis"]["title"] = "Subscribers"
            lo7["margin"]["r"]    = 60
            fig7.update_layout(**lo7)
            st.plotly_chart(fig7, use_container_width=True)

        with col2:
            st.markdown(section_label("SEGMENT SCATTER — ENGAGEMENT vs DURATION"), unsafe_allow_html=True)
            samp = df.sample(min(700, len(df)), random_state=42)
            fig8 = px.scatter(
                samp, x="avg_duration", y="avg_engagement",
                color="segment_label", color_discrete_map=SEGMENT_COLORS,
                size="Tenure Months", size_max=18, opacity=0.72,
                hover_data={"Subscriber Id": True, "Plan": True, "churn_prob": ":.1%"},
                labels={
                    "avg_duration": "Avg Session Duration (min)",
                    "avg_engagement": "Avg Engagement Score",
                    "segment_label": "Segment",
                },
            )
            fig8.update_layout(**base_layout(
                "Engagement vs Duration  (bubble = Tenure)", height=360
            ))
            st.plotly_chart(fig8, use_container_width=True)

        # Segment profiles
        st.markdown("---")
        st.markdown(section_label("SEGMENT PROFILE TABLE"), unsafe_allow_html=True)
        profile_disp = seg_sum.copy()
        profile_disp.columns = [
            "Segment","Count","Avg Engagement","Avg Duration (min)",
            "Avg Tenure (mo)","Churn Rate","Avg NPS",
        ]
        profile_disp["Churn Rate"]        = profile_disp["Churn Rate"].map("{:.1%}".format)
        profile_disp["Avg Engagement"]    = profile_disp["Avg Engagement"].map("{:.1f}".format)
        profile_disp["Avg Duration (min)"]= profile_disp["Avg Duration (min)"].map("{:.1f}".format)
        profile_disp["Avg Tenure (mo)"]   = profile_disp["Avg Tenure (mo)"].map("{:.1f}".format)
        profile_disp["Avg NPS"]           = profile_disp["Avg NPS"].map("{:.1f}".format)
        st.dataframe(profile_disp, use_container_width=True, hide_index=True)

        st.markdown("---")
        col9, col10 = st.columns(2)

        with col9:
            st.markdown(section_label("CHURN RATE BY SEGMENT"), unsafe_allow_html=True)
            fig9 = go.Figure(go.Bar(
                x=seg_sum["segment_label"], y=seg_sum["churn_rate"] * 100,
                marker=dict(
                    color=[SEGMENT_COLORS.get(s, F1_SILVER) for s in seg_sum["segment_label"]],
                    line=dict(color="white", width=1),
                ),
                text=[f"{v*100:.1f}%" for v in seg_sum["churn_rate"]],
                textposition="outside",
                textfont=dict(color=F1_BLACK, size=12),
            ))
            lo9 = base_layout("Churn Rate by KMeans Segment", height=320)
            lo9["yaxis"]["title"] = "Churn Rate (%)"
            lo9["yaxis"]["range"] = [0, float(seg_sum["churn_rate"].max()) * 145]
            fig9.update_layout(**lo9)
            st.plotly_chart(fig9, use_container_width=True)

        with col10:
            st.markdown(section_label("AVG NPS SCORE BY SEGMENT"), unsafe_allow_html=True)
            fig10 = go.Figure(go.Bar(
                x=seg_sum["segment_label"], y=seg_sum["avg_nps"],
                marker=dict(
                    color=[SEGMENT_COLORS.get(s, F1_SILVER) for s in seg_sum["segment_label"]],
                    line=dict(color="white", width=1),
                ),
                text=[f"{v:.1f}" for v in seg_sum["avg_nps"]],
                textposition="outside",
                textfont=dict(color=F1_BLACK, size=12),
            ))
            lo10 = base_layout("Avg NPS Score by Segment", height=320)
            lo10["yaxis"]["title"] = "Avg NPS Score"
            lo10["yaxis"]["range"] = [0, float(seg_sum["avg_nps"].max()) * 1.25]
            fig10.update_layout(**lo10)
            st.plotly_chart(fig10, use_container_width=True)

        # Segment persona cards
        st.markdown("---")
        st.markdown(section_label("SEGMENT PERSONAS & RECOMMENDED ACTIONS"), unsafe_allow_html=True)
        personas = {
            "Champions": {
                "icon": "🏆",
                "desc": "Highest engagement, longest sessions, lowest churn. These are your brand advocates.",
                "action": "Activate as referrers. Offer early access to new content. Use their NPS for social proof.",
                "color": F1_GOLD,
            },
            "Engaged": {
                "icon": "🟢",
                "desc": "Good engagement and session depth. Stable subscribers with moderate tenure.",
                "action": "Upgrade campaign: offer Paddock Club trial. Personalise content to deepen commitment.",
                "color": ACCENT_GREEN,
            },
            "At Risk": {
                "icon": "🟡",
                "desc": "Declining engagement, shorter sessions. Showing early warning signs of churn.",
                "action": "Trigger 30-day re-engagement sequence. Offer 1-month discount. Survey for pain points.",
                "color": ACCENT_AMBER,
            },
            "Dormant": {
                "icon": "🔴",
                "desc": "Lowest engagement, shortest sessions, highest churn probability. Likely to leave soon.",
                "action": "Race Week challenge push. Last-chance offer at 50% off. Flag for human outreach if high-value.",
                "color": F1_RED,
            },
        }
        p1, p2, p3, p4 = st.columns(4)
        for col, (seg, info) in zip([p1, p2, p3, p4], personas.items()):
            row = seg_sum[seg_sum["segment_label"] == seg]
            cnt = int(row["count"].values[0]) if len(row) else 0
            cr  = float(row["churn_rate"].values[0]) if len(row) else 0
            col.markdown(f"""
            <div style="background:white;border:1px solid #E5E5E5;border-top:4px solid {info['color']};
                        border-radius:8px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.06)">
              <div style="font-size:1.8rem">{info['icon']}</div>
              <div style="font-size:1rem;font-weight:800;color:#1A1A1A;margin:4px 0">{seg}</div>
              <div style="font-size:0.72rem;color:#9B9B9B;margin-bottom:8px">
                {cnt} subscribers · {cr:.0%} churn rate</div>
              <div style="font-size:0.78rem;color:#2C2C2C;line-height:1.5;margin-bottom:8px">
                {info['desc']}</div>
              <div style="font-size:0.75rem;color:{info['color']};font-weight:700;line-height:1.5">
                ▶ {info['action']}</div>
            </div>
            """, unsafe_allow_html=True)

        # Insights
        st.markdown("---")
        st.markdown(section_label("KEY CLUSTERING INSIGHTS"), unsafe_allow_html=True)
        top_feat   = imp_df.iloc[-1]["feature"]
        high_risk  = len(df[df["churn_flag"] == 0][df[df["churn_flag"] == 0]["churn_prob"] >= 0.66])
        dormant_n  = len(df[df["segment_label"] == "Dormant"])

        i1, i2, i3 = st.columns(3)
        with i1:
            st.markdown(insight_box(
                f"🎯 <b>'{top_feat}'</b> is the single strongest churn predictor in RF. "
                f"Interventions targeting this variable yield the highest incremental "
                f"retention improvement across all tiers."
            ), unsafe_allow_html=True)
        with i2:
            st.markdown(insight_box(
                f"🚨 <b>{high_risk} active subscribers</b> have predicted churn probability "
                f"above 66%. Ranked by Priority Score (prob × price) in the watchlist above."
            ), unsafe_allow_html=True)
        with i3:
            st.markdown(warn_box(
                f"😴 <b>{dormant_n} Dormant subscribers</b> need immediate re-engagement. "
                f"Low engagement + short sessions + high churn probability. "
                f"A targeted Race Week challenge is the highest-ROI intervention."
            ), unsafe_allow_html=True)
