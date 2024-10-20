# Import
import sys
sys.path.append("/kaggle/src")

from pathlib import Path

import polars as pl
from sklearn.preprocessing import LabelEncoder

from common.constant import TRAIN_CSV_PATH, TEST_CSV_PATH, OUTPUT_DIR
from run.lgb001 import run_kfold

# Data
trn_df = pl.read_csv(TRAIN_CSV_PATH)
tst_df = pl.read_csv(TEST_CSV_PATH)

base_cols = [col for col in trn_df.columns if col in tst_df.columns]
df = pl.concat([trn_df[base_cols], tst_df])

# Feature engineering
features = pl.concat(
    [
        df,
    ],
    how="horizontal",
)

trn_id = trn_df.drop_nulls(subset=["sii"])["id"].to_list()
tst_id = tst_df["id"].to_list()
features = pl.concat([
    features.slice(0, len(trn_df)).filter(pl.col("id").is_in(trn_id)),
    features.slice(len(trn_df)).filter(pl.col("id").is_in(tst_id))
])

# Preprocessing
cat_cols = features.select(pl.col(pl.Utf8)).columns
le = LabelEncoder()
for col in cat_cols:
    encoded = le.fit_transform(features[col].to_numpy())
    features = features.with_columns(pl.Series(encoded).alias(col))

# Model parameters
model_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting": "gbdt",
    "num_leaves": 64,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "learning_rate": 0.1,
    "seed": 1,
    "num_threads": 4,
}

save_dir = OUTPUT_DIR / Path(__file__).stem
save_dir.mkdir(exist_ok=True, parents=True)

run_kfold(
    
)