# Credit Card Fraud Detection: IBM Synthetic Transactions

End-to-end fraud detection on **24.4 million** credit card transactions (IBM's synthetic dataset, 2,000 users, 1991–2020). The work starts with a careful EDA and a leakage-checked feature pipeline before any model is trained.

> **Status:** EDA ✅ · Split and features ✅ · Feature selection ✅ · Models next

## Key findings so far

| Finding | What it means for modeling |
|---|---|
| Fraud is **0.122%** of transactions (1 in 818) | Accuracy is useless. Evaluate with PR-AUC and recall at a fixed precision |
| Online is **11% of volume but ~62% of fraud** | Payment channel is a strong feature |
| Foreign merchant states show **50–100% fraud** vs ~0.03% domestic | Probably a synthetic-data shortcut. Measured with an ablation, not trusted blindly |
| **2020 has 0 frauds** in 336k transactions (~400 expected) | Labels look missing, so 2020 is excluded from evaluation |
| A first purchase at a new merchant is **12.8× more likely** to be fraud | Per-card history features carry real signal |
| Missing merchant state/zip is structural (online transactions) | Encoded explicitly, not imputed |
| Four selection methods agree on dropping 6 of 36 features, with no loss in validation PR-AUC | Every model uses the same 30 selected features |
| Adding `merchant_state` raises validation PR-AUC from 0.73 to 0.90 | The geographic shortcut is large, so results are reported with and without it |
| User snapshot features (age, income, FICO…) add nothing | Excluded: leakage risk with no benefit |

## Approach

- **Chronological split:** train 1991–2017 · validation 2018 (2,491 frauds) · test 2019 (2,087 frauds). A random split would let the model learn from the future.
- **Per-card history features** use only *past* transactions: activity in the last 1h/24h, recent errors, time since the last transaction, new-merchant flag, and amount vs the card's own history.
- **Leakage is tested, not assumed:** the history features are identical whether or not future data is present (truncation test), windowed counts match a brute-force recount, and encoders are fitted on the training period only.
- **No label-derived features.** In production, fraud labels arrive weeks late.

## Roadmap

1. ~~EDA~~ ([`01_eda.ipynb`](notebooks/01_eda.ipynb))
2. ~~Checked table joins, chronological split and feature engineering~~ ([`02_features_split.ipynb`](notebooks/02_features_split.ipynb))
3. ~~Feature selection with four techniques: correlation + mutual information, L1 logistic regression, CatBoost importance, permutation importance~~ ([`03_feature_selection.ipynb`](notebooks/03_feature_selection.ipynb))
4. Traditional models: class-weighted logistic regression and Random Forest, then tuning
5. CatBoost, then tuning
6. FT-Transformer (deep learning for tabular data), then tuning
7. Ablations: geographic shortcut, user-snapshot leakage, per-segment performance
8. Final evaluation of all models on the 2019 test year (once), with bootstrap confidence intervals and SHAP

Every model uses the same split, the same metrics (PR-AUC, recall at fixed precision) and a decision threshold chosen on validation. The original reasoning behind the phases is in section 8 of the EDA notebook.

## Project structure

```
├── data/
│   ├── raw/          # Kaggle CSVs (not committed, see data/README.md)
│   └── cache/        # generated Parquet files (not committed)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features_split.ipynb
│   └── 03_feature_selection.ipynb
├── reports/
│   ├── figures/
│   └── feature_selection.csv   # per-feature votes from the four methods
├── src/
│   ├── data.py       # loading, cleaning, Parquet caching, checked user/card joins
│   ├── features.py   # split definition, feature engineering, encoders
│   └── modeling.py   # sampling, preprocessing, shared model helpers
├── environment.yml
└── requirements.txt
```

## Setup

```bash
conda env create -f environment.yml
conda activate fraud-detection
python -m ipykernel install --user --name fraud-detection
```

Download the data as described in [`data/README.md`](data/README.md), then run the notebooks in order. The first load parses the 2.3 GB CSV (about 2 minutes) and caches it as Parquet. Later loads take seconds.

## Data

[Credit Card Transactions](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions) by Erik Altman (IBM), on Kaggle. The data is **synthetic**: patterns reflect the generator, so results won't transfer directly to real fraud.
