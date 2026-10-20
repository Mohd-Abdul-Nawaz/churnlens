"""EDA summaries, segmentation tables and churn models for ChurnLens.

Streamlit-free: the app is a thin presentation layer over these routines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["tenure_months", "weekly_hours", "support_tickets",
           "billing_issues", "monthly_fee", "age"]
CATEGORICAL = ["contract", "plan", "device", "region"]
TARGET = "churned"


def data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column completeness, uniques and dtype — the first EDA table."""
    rows = []
    for col in df.columns:
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "missing": int(df[col].isna().sum()),
            "missing_pct": float(df[col].isna().mean() * 100),
            "n_unique": int(df[col].nunique()),
        })
    return pd.DataFrame(rows)


def churn_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Churn rate + exposure per category, sorted by churn rate desc."""
    g = (df.groupby(col)[TARGET]
           .agg(units="size", churned="sum")
           .reset_index())
    g["churn_rate"] = g["churned"] / g["units"]
    return g.sort_values("churn_rate", ascending=False).reset_index(drop=True)


def tenure_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    """Churn rate by 6-month tenure cohorts (early-life churn signal)."""
    tmp = df.copy()
    tmp["cohort"] = pd.cut(tmp["tenure_months"],
                           bins=[0, 6, 12, 24, 36, 49],
                           labels=["0-6", "7-12", "13-24", "25-36", "37-48"])
    g = tmp.groupby("cohort", observed=True)[TARGET].agg(units="size",
                                                         churn_rate="mean")
    return g.reset_index()


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])


def time_aware_split(df: pd.DataFrame, test_size: float = 0.25,
                     seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split (proxy for out-of-time validation)."""
    return train_test_split(df, test_size=test_size, random_state=seed,
                            stratify=df[TARGET])


@dataclass
class ModelReport:
    name: str
    roc_auc: float
    avg_precision: float
    odds_ratios: pd.DataFrame | None = None


def _get_feature_names(pre: ColumnTransformer) -> list[str]:
    try:
        return [str(c) for c in pre.get_feature_names_out()]
    except Exception:
        return NUMERIC + CATEGORICAL


def train_logistic(train: pd.DataFrame, test: pd.DataFrame) -> ModelReport:
    pre = build_preprocessor()
    clf = LogisticRegression(max_iter=2000)
    pipe = Pipeline([("prep", pre), ("model", clf)])
    X_tr, y_tr = train.drop(columns=[TARGET, "customer_id"]), train[TARGET]
    X_te, y_te = test.drop(columns=[TARGET, "customer_id"]), test[TARGET]
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]

    feats = _get_feature_names(pipe.named_steps["prep"])
    coef = pipe.named_steps["model"].coef_.ravel()
    odds = pd.DataFrame({
        "feature": feats,
        "coefficient": coef,
        "odds_ratio": np.exp(coef),
    }).sort_values("coefficient", ascending=False).reset_index(drop=True)
    return ModelReport(
        name="Logistic Regression",
        roc_auc=float(roc_auc_score(y_te, proba)),
        avg_precision=float(average_precision_score(y_te, proba)),
        odds_ratios=odds,
    )


def train_gbm(train: pd.DataFrame, test: pd.DataFrame, seed: int = 42) -> ModelReport:
    pre = build_preprocessor()
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                     learning_rate=0.08, random_state=seed)
    pipe = Pipeline([("prep", pre), ("model", clf)])
    X_tr, y_tr = train.drop(columns=[TARGET, "customer_id"]), train[TARGET]
    X_te, y_te = test.drop(columns=[TARGET, "customer_id"]), test[TARGET]
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    return ModelReport(
        name="Gradient Boosting",
        roc_auc=float(roc_auc_score(y_te, proba)),
        avg_precision=float(average_precision_score(y_te, proba)),
    )


def risk_deciles(df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """Lift table: top decile should contain a multiple of the base rate."""
    tmp = df[[TARGET]].copy()
    tmp["proba"] = proba
    tmp["decile"] = pd.qcut(tmp["proba"], 10, labels=False,
                            duplicates="drop") + 1
    lift = (tmp.groupby("decile")[TARGET]
               .agg(units="size", churned="sum", churn_rate="mean")
               .reset_index()
               .sort_values("decile", ascending=False)
               .reset_index(drop=True))
    total_churned = int(tmp[TARGET].sum())
    lift["cumulative_capture"] = lift["churned"].cumsum() / max(total_churned, 1)
    return lift
