"""
model_utils.py
──────────────
Feature engineering, all classification models, KMeans segmentation,
and regression models (LR, Ridge, Lasso) for MRR/LTV prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso,
)
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve, mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(subs: pd.DataFrame, sess: pd.DataFrame) -> pd.DataFrame:
    agg = (
        sess.groupby("Subscriber Id")
        .agg(
            total_sessions      = ("Session Duration Min", "count"),
            avg_engagement      = ("Engagement Score",     "mean"),
            std_engagement      = ("Engagement Score",     "std"),
            avg_duration        = ("Session Duration Min", "mean"),
            total_duration      = ("Session Duration Min", "sum"),
            mobile_sessions     = ("Device",         lambda x: (x == "Mobile").sum()),
            weekend_sessions    = ("Is Weekend",      "sum"),
            high_eng_sessions   = ("Engagement Tier", lambda x: (x == "High").sum()),
            medium_eng_sessions = ("Engagement Tier", lambda x: (x == "Medium").sum()),
        )
        .reset_index()
    )
    agg["mobile_pct"]     = (agg["mobile_sessions"]   / agg["total_sessions"]).round(4)
    agg["weekend_pct"]    = (agg["weekend_sessions"]  / agg["total_sessions"]).round(4)
    agg["high_eng_pct"]   = (agg["high_eng_sessions"] / agg["total_sessions"]).round(4)
    agg["std_engagement"] = agg["std_engagement"].fillna(0)

    top_content = (
        sess.groupby("Subscriber Id")["Content Type"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index().rename(columns={"Content Type": "top_content"})
    )

    df = subs.merge(agg, on="Subscriber Id", how="left")
    df = df.merge(top_content, on="Subscriber Id", how="left")

    for c in ["total_sessions","avg_engagement","std_engagement",
              "avg_duration","total_duration","mobile_pct","weekend_pct","high_eng_pct"]:
        df[c] = df[c].fillna(0)
    df["top_content"] = df["top_content"].fillna("Unknown")

    cat_map = {
        "Plan": "plan_enc", "Region": "region_enc",
        "Acquisition Channel": "channel_enc",
        "Age Group": "age_group_enc", "top_content": "content_enc",
    }
    for src, dst in cat_map.items():
        df[dst] = LabelEncoder().fit_transform(df[src].astype(str))

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE COLUMNS
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "plan_enc", "Monthly Price Usd", "region_enc", "channel_enc",
    "Age", "age_group_enc", "Tenure Months", "Renewal Count", "Nps Score",
    "total_sessions", "avg_engagement", "std_engagement", "avg_duration",
    "mobile_pct", "weekend_pct", "high_eng_pct", "content_enc",
]

FEATURE_LABELS = [
    "Plan Tier", "Monthly Price", "Region", "Acquisition Channel",
    "Age", "Age Group", "Tenure (months)", "Renewal Count", "NPS Score",
    "Total Sessions", "Avg Engagement Score", "Engagement Variability",
    "Avg Session Duration", "Mobile Usage %", "Weekend Usage %",
    "High-Engagement Session %", "Top Content Type",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  SINGLE CHURN MODEL (RF — used by prescriptive)
# ═══════════════════════════════════════════════════════════════════════════════

def train_churn_model(df: pd.DataFrame):
    X = df[FEATURE_COLS].fillna(0)
    y = df["churn_flag"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    importance_df = (
        pd.DataFrame({"feature": FEATURE_LABELS, "importance": clf.feature_importances_})
        .sort_values("importance", ascending=True).reset_index(drop=True)
    )
    df_scored = df.copy()
    df_scored["churn_prob"] = clf.predict_proba(X)[:, 1]
    df_scored["churn_pred"] = clf.predict(X)

    return clf, X_train, X_test, y_train, y_test, y_pred, y_prob, importance_df, df_scored


def get_model_metrics(y_test, y_pred, y_prob) -> dict:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
    return {
        "accuracy":   round(accuracy_score(y_test, y_pred), 4),
        "precision":  round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":     round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":         round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc":        round(roc_auc_score(y_test, y_prob), 4),
        "cm":         confusion_matrix(y_test, y_pred),
        "fpr": fpr, "tpr": tpr,
        "prec_curve": prec_c, "rec_curve": rec_c,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ALL CLASSIFIERS COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def train_all_classifiers(df: pd.DataFrame) -> dict:
    """
    Train 6 classifiers on the same train/test split.
    Returns dict keyed by model name with metrics + curve data.
    """
    X_raw = df[FEATURE_COLS].fillna(0)
    y     = df["churn_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.25, random_state=42, stratify=y
    )

    # Scale for distance-based + linear models
    sc      = StandardScaler()
    Xtr_s   = sc.fit_transform(X_train)
    Xte_s   = sc.transform(X_test)

    classifiers = {
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=4,
                class_weight="balanced", random_state=42, n_jobs=-1,
            ),
            X_train, X_test, False,
        ),
        "Logistic Reg.": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            Xtr_s, Xte_s, True,
        ),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
            X_train, X_test, False,
        ),
        "KNN": (
            KNeighborsClassifier(n_neighbors=9),
            Xtr_s, Xte_s, True,
        ),
        "Naive Bayes": (
            GaussianNB(),
            Xtr_s, Xte_s, True,
        ),
        "SVM": (
            SVC(probability=True, class_weight="balanced", random_state=42),
            Xtr_s, Xte_s, True,
        ),
    }

    results = {}
    for name, (clf, Xtr, Xte, _) in classifiers.items():
        clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xte)
        y_prob = clf.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "auc":       round(roc_auc_score(y_test, y_prob), 4),
            "cm":        cm,
            "fpr":       fpr,
            "tpr":       tpr,
            "y_test":    y_test,
            "y_pred":    y_pred,
            "y_prob":    y_prob,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  KMEANS SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def segment_customers(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    feats = ["avg_engagement","avg_duration","total_sessions",
             "Tenure Months","mobile_pct","high_eng_pct"]
    X_raw    = df[feats].fillna(0)
    X_scaled = StandardScaler().fit_transform(X_raw)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["segment"] = km.fit_predict(X_scaled)
    rank = (
        df.groupby("segment")["avg_engagement"].mean()
        .sort_values(ascending=False).index.tolist()
    )
    labels = ["Champions", "Engaged", "At Risk", "Dormant"]
    df["segment_label"] = df["segment"].map(
        {seg: lbl for seg, lbl in zip(rank, labels)}
    )
    return df


def get_kmeans_elbow(df: pd.DataFrame, max_k: int = 10) -> tuple[list, list]:
    """Return (k_range, inertias) for elbow chart."""
    feats = ["avg_engagement","avg_duration","total_sessions",
             "Tenure Months","mobile_pct","high_eng_pct"]
    X_scaled = StandardScaler().fit_transform(df[feats].fillna(0))
    inertias = []
    k_range  = list(range(2, max_k + 1))
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    return k_range, inertias


# ═══════════════════════════════════════════════════════════════════════════════
#  REGRESSION — Predict Lifetime Revenue USD
# ═══════════════════════════════════════════════════════════════════════════════

REG_FEATURES = [
    "Monthly Price Usd", "Tenure Months", "Renewal Count", "Nps Score",
    "total_sessions", "avg_engagement", "avg_duration",
    "mobile_pct", "weekend_pct", "high_eng_pct",
    "plan_enc", "region_enc", "channel_enc",
]

REG_FEATURE_LABELS = [
    "Monthly Price", "Tenure (months)", "Renewal Count", "NPS Score",
    "Total Sessions", "Avg Engagement", "Avg Session Duration",
    "Mobile Usage %", "Weekend Usage %", "High-Eng Session %",
    "Plan Tier", "Region", "Acquisition Channel",
]


def train_regression_models(df: pd.DataFrame) -> dict:
    """
    Train Linear, Ridge, Lasso on Lifetime Revenue USD prediction.
    Returns dict with per-model metrics and coefficients.
    """
    target = "Lifetime Revenue Usd"
    df2 = df.dropna(subset=[target]).copy()
    X   = df2[REG_FEATURES].fillna(df2[REG_FEATURES].median())
    y   = df2[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    sc      = StandardScaler()
    Xtr_s   = sc.fit_transform(X_train)
    Xte_s   = sc.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge (alpha=10)":  Ridge(alpha=10.0),
        "Lasso (alpha=1)":   Lasso(alpha=1.0, max_iter=5000),
    }

    results = {}
    for name, mdl in models.items():
        mdl.fit(Xtr_s, y_train)
        yp = mdl.predict(Xte_s)
        results[name] = {
            "model":   mdl,
            "r2":      round(float(r2_score(y_test, yp)), 3),
            "rmse":    round(float(np.sqrt(mean_squared_error(y_test, yp))), 2),
            "mae":     round(float(mean_absolute_error(y_test, yp)), 2),
            "y_test":  y_test,
            "y_pred":  yp,
            "residuals": y_test - yp,
            "coefs":   pd.Series(mdl.coef_, index=REG_FEATURE_LABELS),
        }

    results["_meta"] = {
        "sc": sc, "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_labels": REG_FEATURE_LABELS,
    }
    return results
