# Credit Card Fraud Detection: IBM Synthetic Transactions

End-to-end fraud detection on **24.4 million** credit card transactions (IBM's synthetic dataset, 2,000 users, 1991–2020). The work starts with a careful EDA and a leakage-checked feature pipeline before any model is trained.

> **Status:** complete. All four models were scored once on the 2019 test year (see [Results](#results-2019-test-year)).

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
4. ~~Traditional models: logistic regression and Random Forest, tuned with Optuna~~ ([`04_traditional_models.ipynb`](notebooks/04_traditional_models.ipynb))
5. ~~CatBoost, then tuning~~ ([`05_catboost.ipynb`](notebooks/05_catboost.ipynb))
6. ~~FT-Transformer (deep learning for tabular data), then tuning~~ ([`06_ft_transformer.ipynb`](notebooks/06_ft_transformer.ipynb))
7. ~~Ablations: geographic shortcut, user-snapshot leakage, per-segment performance~~ ([`07_ablations.ipynb`](notebooks/07_ablations.ipynb))
8. ~~Final evaluation of all models on the 2019 test year (once), with bootstrap confidence intervals and SHAP~~ ([`08_final_evaluation.ipynb`](notebooks/08_final_evaluation.ipynb))

Every model uses the same split, the same metrics (PR-AUC, recall at fixed precision) and a decision threshold chosen on validation. The original reasoning behind the phases is in section 8 of the EDA notebook.

## Results: 2019 test year

The test year (1.72M transactions, 2,087 frauds) was scored **once**, after every model, feature set and threshold was frozen on validation. **B** is the headline variant: geography removed (`is_international`, `zip_freq`). **A** keeps it.

| Model | Val PR-AUC (B) | **Test PR-AUC (B)** | 95% bootstrap CI | Test PR-AUC (A) |
|---|---|---|---|---|
| CatBoost | 0.480 | **0.481** | 0.454–0.504 | 0.873 |
| Random Forest | 0.483 | **0.463** | 0.443–0.485 | 0.737 |
| FT-Transformer | 0.424 | **0.417** | 0.392–0.441 | 0.478 |
| Logistic regression | 0.020 | **0.011** | 0.010–0.013 | 0.132 |

- **Validation scores held on test.** Every B model is within 0.02 of its validation score, so model selection did not overfit validation.
- **CatBoost and Random Forest are statistically tied** on B: CatBoost leads by 0.018, with a paired bootstrap CI of −0.012 to +0.044. Both beat FT-Transformer significantly. Gradient-boosted and bagged trees beat the deep tabular model here.
- **The geographic shortcut is worth 0.39 PR-AUC** (CatBoost A − B, CI 0.378–0.410), which confirms the ablation finding on unseen data.

**Limitations**
- **All 2,087 test frauds are in-store international transactions**, none domestic or online (in validation, 94% were). The models are therefore tested on one fraud pattern. Within that segment, CatBoost B reaches 0.877 PR-AUC.
- **Geography may still leak into B through `merchant_city_freq`.** It is the top SHAP feature for Random Forest B (0.39 of mean |SHAP|) and for CatBoost B and FT-Transformer B. Rare foreign cities may stand in for "international", as `zip_freq` did. The test year is spent, so this is reported here and not tested with another ablation.
- The data is synthetic (see [Data](#data)).

Figures: [`test_pr_curves.png`](reports/figures/test_pr_curves.png), [`shap_share_heatmap.png`](reports/figures/shap_share_heatmap.png), [`shap_beeswarms.png`](reports/figures/shap_beeswarms.png). Tables: `reports/test_*.csv`, `reports/shap_importance_share.csv`.

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
