"""Unit tests for ChurnLens data generator + pipeline.

Run:  pytest churnlens/tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_gen, pipeline as pl  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return data_gen.generate(n=4000, seed=11)


def test_schema_and_target(df):
    assert set(df.columns) == {
        "customer_id", "contract", "plan", "device", "region",
        "tenure_months", "weekly_hours", "support_tickets",
        "billing_issues", "monthly_fee", "age", "churned",
    }
    assert set(df["churned"].unique()) <= {0, 1}
    assert df["customer_id"].is_unique


def test_reproducible_with_same_seed():
    a = data_gen.generate(n=1000, seed=3)
    b = data_gen.generate(n=1000, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_churn_rate_in_plausible_range(df):
    assert 0.05 < df["churned"].mean() < 0.60


def test_month_to_month_churns_most(df):
    seg = pl.churn_by(df, "contract")
    assert seg.iloc[0]["contract"] == "month-to-month"
    assert seg["churn_rate"].is_monotonic_decreasing


def test_billing_issues_raise_churn(df):
    yes = df[df["billing_issues"] == 1]["churned"].mean()
    no = df[df["billing_issues"] == 0]["churned"].mean()
    assert yes > no


def test_missingness_is_handled(df):
    assert df["weekly_hours"].isna().sum() > 0  # traps exist…
    train, test = pl.time_aware_split(df)
    rep = pl.train_logistic(train, test)  # …yet models still fit
    assert 0.5 < rep.roc_auc <= 1.0


def test_stratified_split_preserves_rate(df):
    train, test = pl.time_aware_split(df, test_size=0.25)
    assert len(test) == pytest.approx(len(df) * 0.25, abs=2)
    assert test["churned"].mean() == pytest.approx(
        df["churned"].mean(), abs=0.03
    )


def test_logistic_odds_direction(df):
    train, test = pl.time_aware_split(df)
    rep = pl.train_logistic(train, test)
    odds = rep.odds_ratios.set_index("feature")["coefficient"]
    # monthly contracts and billing issues should push churn UP
    mtm = [c for c in odds.index if "month-to-month" in c]
    bill = [c for c in odds.index if "billing_issues" in c]
    assert mtm and odds[mtm[0]] > 0
    assert bill and odds[bill[0]] > 0


def test_gbm_beats_baseline(df):
    train, test = pl.time_aware_split(df)
    rep = pl.train_gbm(train, test)
    assert rep.roc_auc > 0.70
    assert rep.avg_precision > df["churned"].mean()


def test_lift_table_top_decile_concentrates_risk(df):
    train, test = pl.time_aware_split(df)
    feats = [c for c in train.columns if c not in ("churned", "customer_id")]
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("prep", pl.build_preprocessor()),
                     ("model", GradientBoostingClassifier(random_state=0))])
    pipe.fit(train[feats], train["churned"])
    proba = pipe.predict_proba(test[feats])[:, 1]
    lift = pl.risk_deciles(test.reset_index(drop=True), proba)
    assert len(lift) == 10
    top = lift.iloc[0]
    assert top["churn_rate"] > test["churned"].mean() * 1.5
    assert lift["cumulative_capture"].iloc[-1] == pytest.approx(1.0)


def test_tenure_cohorts_shape(df):
    coh = pl.tenure_cohorts(df)
    assert list(coh.columns) == ["cohort", "units", "churn_rate"]
    assert coh["units"].sum() == len(df)
