"""ChurnLens — subscription churn EDA + prediction dashboard.

Tabs: Data quality | Segments | Tenure cohorts | Models | Retention playbook.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src import data_gen, pipeline as pl  # noqa: E402

st.set_page_config(page_title="ChurnLens", page_icon="🔭", layout="wide")

st.title("🔭 ChurnLens")
st.caption(
    "End-to-end churn analysis: pandas data quality, segment & cohort EDA, "
    "interpretable logistic model vs gradient boosting, lift table."
)


@st.cache_data(show_spinner="Generating dataset…")
def load(n: int, seed: int) -> pd.DataFrame:
    return data_gen.generate(n=n, seed=seed)


with st.sidebar:
    st.header("Data")
    n = st.slider("Customers", 3_000, 30_000, 12_000, 1_000)
    seed = st.number_input("Seed", 0, 9999, 11)
    st.divider()
    st.markdown(
        "**What's inside**\n\n"
        "Synthetic streaming-subscription data with realistic traps: "
        "month-to-month contracts churn more, new tenants churn early, "
        "usage metrics have missing values, billing issues hurt."
    )

df = load(int(n), int(seed))
base = float(df[pl.TARGET].mean())

tab_q, tab_seg, tab_coh, tab_mod, tab_play = st.tabs(
    ["🧹 Data quality", "🧭 Segments", "📆 Tenure cohorts",
     "🤖 Models", "💡 Retention playbook"]
)

# ------------------------------------------------------------- data quality ----
with tab_q:
    st.subheader("First five rows")
    st.dataframe(df.head(), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Overall churn rate", f"{base:.1%}")
    c3.metric("Columns", df.shape[1])

    st.subheader("Completeness & cardinality")
    st.dataframe(pl.data_quality(df), use_container_width=True)
    if (dq := pl.data_quality(df))["missing"].sum() > 0:
        st.info("Missing usage hours are median-imputed inside the model "
                "pipeline — see the Models tab.")

# --------------------------------------------------------------- segments ----
with tab_seg:
    st.subheader("Churn rate by segment")
    col = st.selectbox("Segment by", ["contract", "plan", "device", "region"])
    seg = pl.churn_by(df, col)
    fig = px.bar(seg, x=col, y="churn_rate", text="units",
                 labels={"churn_rate": "churn rate", "units": "customers"},
                 color="churn_rate", color_continuous_scale="Reds")
    fig.add_hline(y=base, line_dash="dash", annotation_text="overall rate")
    fig.update_layout(height=400, margin=dict(t=30, b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(seg.style.format({"churn_rate": "{:.1%}"}),
                 use_container_width=True)

    st.subheader("Usage vs churn")
    num_col = st.selectbox("Numeric driver", ["weekly_hours", "tenure_months",
                                              "support_tickets", "monthly_fee"])
    tmp = df[[num_col, pl.TARGET]].copy()
    tmp["churned"] = tmp[pl.TARGET].map({0: "stayed", 1: "churned"})
    fig = px.box(tmp, x="churned", y=num_col, points=False,
                 color="churned")
    fig.update_layout(height=380, margin=dict(t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------- cohorts ----
with tab_coh:
    st.subheader("Early-life churn: tenure cohorts")
    coh = pl.tenure_cohorts(df)
    fig = px.bar(coh, x="cohort", y="churn_rate", text="units",
                 labels={"cohort": "tenure cohort (months)",
                         "churn_rate": "churn rate"})
    fig.add_hline(y=base, line_dash="dash", annotation_text="overall rate")
    fig.update_layout(height=400, margin=dict(t=30, b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "New customers churn at the highest rate — onboarding and the first "
        "renewal decision are the highest-leverage moments to intervene."
    )

# ------------------------------------------------------------------ models ----
with tab_mod:
    st.subheader("Predict: who churns next?")
    train, test = pl.time_aware_split(df)
    st.write(f"Train {len(train):,} · test {len(test):,} (stratified holdout)")

    lr = pl.train_logistic(train, test)
    gbm = pl.train_gbm(train, test)

    c1, c2 = st.columns(2)
    c1.metric("Logistic ROC-AUC", f"{lr.roc_auc:.3f}",
              delta=f"AP {lr.avg_precision:.3f}", delta_color="off")
    c2.metric("GBM ROC-AUC", f"{gbm.roc_auc:.3f}",
              delta=f"AP {gbm.avg_precision:.3f}", delta_color="off")

    st.subheader("What drives the prediction? (logistic odds ratios)")
    st.write(
        "Odds ratio > 1 raises churn odds, < 1 lowers them. Because features "
        "are standardised, coefficient magnitudes are comparable."
    )
    odds = lr.odds_ratios
    top = pd.concat([odds.head(8), odds.tail(4)]).drop_duplicates()
    fig = px.bar(top.iloc[::-1], x="coefficient", y="feature",
                 orientation="h", color="coefficient",
                 color_continuous_scale="RdBu_r",
                 labels={"coefficient": "standardised log-odds coefficient"})
    fig.update_layout(height=420, margin=dict(t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(odds.head(12).style.format({"coefficient": "{:+.3f}",
                                             "odds_ratio": "{:.3f}"}),
                 use_container_width=True)

    st.subheader("Lift table (GBM on holdout)")
    feats = ["tenure_months", "weekly_hours", "support_tickets",
             "billing_issues", "monthly_fee", "age",
             "contract", "plan", "device", "region"]
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import GradientBoostingClassifier
    pre = pl.build_preprocessor()
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                     learning_rate=0.08, random_state=42)
    pipe = Pipeline([("prep", pre), ("model", clf)])
    X_tr, y_tr = train[feats], train[pl.TARGET]
    X_te, y_te = test[feats], test[pl.TARGET]
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    lift = pl.risk_deciles(test.reset_index(drop=True), proba)
    top_dec = lift.iloc[0]
    st.write(f"**Top decile churns at {top_dec['churn_rate']:.1%}** "
             f"({top_dec['churn_rate'] / max(base, 1e-9):.1f}× the base rate) "
             f"and captures {top_dec['cumulative_capture']:.0%} of all churners.")
    fig = px.bar(lift, x="decile", y="churn_rate", text="units",
                 labels={"decile": "risk decile (10 = riskiest)",
                         "churn_rate": "observed churn rate"})
    fig.add_hline(y=base, line_dash="dash", annotation_text="base rate")
    fig.update_layout(height=380, margin=dict(t=30, b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(lift.style.format({"churn_rate": "{:.1%}",
                                    "cumulative_capture": "{:.0%}"}),
                 use_container_width=True)

# ---------------------------------------------------------------- playbook ----
with tab_play:
    st.subheader("Retention playbook — from findings to actions")
    seg_c = pl.churn_by(df, "contract").iloc[0]
    seg_b = df[df["billing_issues"] == 1][pl.TARGET].mean()
    seg_n = pl.tenure_cohorts(df).iloc[0]["churn_rate"]
    st.markdown(f"""
1. **Win back {seg_c['contract']} users ({seg_c['churn_rate']:.0%} churn).**
   Annual-contract incentives and save-offers at the renewal moment.
2. **Fix billing, fast ({seg_b:.0%} churn after a billing issue).**
   Proactive credits and a retry-friendly dunning flow.
3. **Onboard the first cohort ({seg_n:.0%} churn in months 0–6).**
   Guided setup, usage nudges, and early win-state activation.
4. **Score & target daily.** The top risk decile concentrates churn
   several-fold — route it to the save desk instead of blasting everyone.
""")
    st.caption("Every number above is computed live from the current dataset.")

