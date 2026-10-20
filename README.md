# 🔭 ChurnLens — Subscription Churn EDA & Prediction

End-to-end churn analysis for a streaming-subscription business:
pandas data-quality audit → segment & tenure-cohort EDA → interpretable
logistic model vs. gradient boosting → lift table → retention playbook.

## What it does

- 🧹 **Data quality** — completeness/cardinality audit (incl. missing usage)
- 🧭 **Segments** — churn by contract/plan/device/region + usage-vs-churn
- 📆 **Tenure cohorts** — early-life churn curve
- 🤖 **Models** — stratified holdout; ROC-AUC + average precision;
  standardised logistic odds ratios; GBM lift/decile table
- 💡 **Retention playbook** — actions auto-computed from live segment stats

Data: seeded synthetic generator (`src/data_gen.py`, 12k customers,
contract/tenure/usage/billing/support effects + missing values).

## Run locally

```bash
cd churnlens
pip install -r requirements.txt
streamlit run app.py
pytest tests -q
```

## Deploy

Same pattern: GitHub → <https://share.streamlit.io> → `churnlens/app.py`
(or a Hugging Face Streamlit Space).

## Files

- `app.py` — Streamlit UI
- `src/data_gen.py` — reproducible synthetic churn data
- `src/pipeline.py` — EDA summaries, splits, models, lift table
- `tests/test_pipeline.py` — 11 tests (directional effects, lift math,
  missingness handling, determinism)
