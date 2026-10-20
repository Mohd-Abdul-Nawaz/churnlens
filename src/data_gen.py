"""Synthetic subscription-churn dataset generator (seeded, reproducible).

Models a streaming-subscription service: contract type, tenure, usage,
billing problems and support contacts drive churn through a known logit
so behavioural stories are visible but not trivially separable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONTRACTS = ["month-to-month", "one-year", "two-year"]
DEVICES = ["mobile", "tv", "web", "console"]
PLANS = ["basic", "standard", "premium"]
REGIONS = ["north", "south", "east", "west"]


def generate(n: int = 12_000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    contract = rng.choice(CONTRACTS, size=n, p=[0.45, 0.30, 0.25])
    plan = rng.choice(PLANS, size=n, p=[0.30, 0.45, 0.25])
    device = rng.choice(DEVICES, size=n, p=[0.40, 0.30, 0.20, 0.10])
    region = rng.choice(REGIONS, size=n, p=[0.25, 0.25, 0.25, 0.25])

    tenure_months = rng.integers(1, 49, size=n).astype(float)
    weekly_hours = np.clip(rng.normal(8.0, 4.0, size=n), 0.0, None)
    support_tickets = rng.poisson(0.6, size=n).astype(float)
    billing_issues = rng.binomial(1, 0.08, size=n).astype(float)
    monthly_fee = np.where(plan == "basic", 9.99,
                   np.where(plan == "standard", 15.99, 22.99))
    monthly_fee = monthly_fee + rng.normal(0, 0.5, size=n)
    age = rng.integers(18, 76, size=n).astype(float)

    # ~3% missingness in usage metrics (observational realism)
    for col, arr in (("weekly_hours", weekly_hours),):
        mask = rng.random(n) < 0.03
        arr = arr.copy()
        arr[mask] = np.nan
        weekly_hours = arr

    logit = (
        -1.2
        + np.where(contract == "month-to-month", 1.1, 0.0)
        + np.where(contract == "two-year", -0.6, 0.0)
        - 0.03 * tenure_months
        - 0.09 * np.nan_to_num(weekly_hours, nan=8.0)
        + 0.45 * support_tickets
        + 1.15 * billing_issues
        - 0.4 * (plan == "premium")
        + 0.2 * (plan == "basic")
    )
    prob = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, prob, size=n).astype(int)

    return pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(n)],
        "contract": contract,
        "plan": plan,
        "device": device,
        "region": region,
        "tenure_months": tenure_months,
        "weekly_hours": weekly_hours,
        "support_tickets": support_tickets,
        "billing_issues": billing_issues,
        "monthly_fee": monthly_fee.round(2),
        "age": age,
        "churned": churned,
    })
