import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold

from common.constant import SPLIT_RANDOM_SEED

def train_kfold(features, targets, categorical_features, n_splits, model_params, weights=None):
    if weights is None:
        weights = np.ones(len(features))
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=SPLIT_RANDOM_SEED
    )
    boosters = []
    val_qwks = []
    oof_preds = np.zeros(len(features))

    for trn_idx, val_idx in skf.split(features, targets):
        trn_data = lgb.Dataset()




def run_kfold(
        features, trn_targets, categorical_features, n_splits, model_params, save_dir, 
        weights=None, trn_id=None, tst_id=None
):
    trn_features = features.iloc[:len(trn_targets)]
    tst_features = features.iloc[len(trn_targets):]
    boosters, qwks, oof_preds = train_kfold(
        trn_features, trn_targets, categorical_features, n_splits, model_params, weights
    )