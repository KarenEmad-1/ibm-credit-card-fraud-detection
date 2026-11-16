# Data

Raw data isn't committed (the main file is about 2.3 GB).

1. Download **Credit Card Transactions** (IBM synthetic data) from Kaggle:
   https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions
2. Put these files in `data/raw/`:
   - `credit_card_transactions-ibm_v2.csv` (24.4M transactions)
   - `sd254_users.csv` (2,000 users)
   - `sd254_cards.csv` (6,146 cards)
   - `User0_credit_card_transactions.csv` (single-user sample, optional)

`data/cache/` holds generated Parquet files and is also git-ignored.
