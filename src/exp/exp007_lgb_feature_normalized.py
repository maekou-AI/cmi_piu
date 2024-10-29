# Import
import sys
sys.path.append("/kaggle/src")  # Change on kaggle notebook

from pathlib import Path

import polars as pl
from sklearn.preprocessing import LabelEncoder

from common.constant import TRAIN_CSV_PATH, TEST_CSV_PATH, OUTPUT_DIR
from feature.table import build_feature
from preprocess.table import fill_missing, remove_outlier
from run.lgb002 import run_kfold

# Data
trn_df = pl.read_csv(TRAIN_CSV_PATH)
tst_df = pl.read_csv(TEST_CSV_PATH)

base_cols = [col for col in trn_df.columns if col in tst_df.columns]
df = pl.concat([trn_df[base_cols], tst_df])

# Feature engineering
features = pl.concat(
    [
        df,
        build_feature(fill_missing(remove_outlier(df)))
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
cat_cols.remove("id")
le = LabelEncoder()
for col in cat_cols:
    encoded = le.fit_transform(features[col].to_numpy())
    features = features.with_columns(pl.Series(encoded).alias(col))

# Model parameters
model_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting": "gbdt",
    "num_leaves": 140,
    "max_depth": 16,
    "min_data_in_leaf": 100,
    "bagging_fraction": 0.8,
    "bagging_freq": 6,
    "feature_fraction": 0.7,
    "min_child_sample": 55,
    "learning_rate": 0.01,
    "lambda_l1": 0.2,
    "lambda_l2": 1e-6,
    "seed": 1,
    "num_threads": 4,
    "verbose": -1
}

save_dir = Path(OUTPUT_DIR) / Path(__file__).stem  # Change on kaggle notebook
save_dir.mkdir(exist_ok=True, parents=True)

run_kfold(
    features=features.drop("id").to_pandas(),
    trn_targets=trn_df.filter(pl.col("id").is_in(trn_id))["sii"].to_pandas(),
    categorical_features=cat_cols,
    n_splits=10,
    save_dir=save_dir,
    model_params=model_params,
    weights=None,
    trn_id=trn_id,
    tst_id=tst_id,
)