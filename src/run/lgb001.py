import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold


NUM_BOOST_ROUND = 5000
STOPPING_ROUNDS = 20
VERBOSE_EVAL = 30


def quadratic_weighted_kappa(preds, data):
    y_true = data.get_label()
    y_pred = preds.clip(0, 3).round().astype(int)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return "QWK", qwk, True


def threshold_rounder(oof_preds, thresholds):
    labels = []
    for value in oof_preds:
        if value < thresholds[0]:
            labels.append(0)
        elif value < thresholds[1]:
            labels.append(1)
        elif value < thresholds[2]:
            labels.append(2)
        else:
            labels.append(3)
    return np.array(labels)


def evaluate_predictions(thresholds, y_true, oof_preds):
    oof_preds_rounded = threshold_rounder(oof_preds, thresholds)
    return -cohen_kappa_score(y_true, oof_preds_rounded, weights="quadratic")


def train_kfold(features, targets, categorical_features, n_splits, model_params, weights=None):
    if weights is None:
        weights = np.ones(len(features))
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=1
    )
    boosters = []
    val_qwks = []
    oof_preds = np.zeros(len(features))

    for trn_idx, val_idx in skf.split(features, targets):
        trn_data = lgb.Dataset(
            data=features.iloc[trn_idx],
            label=targets.iloc[trn_idx],
            weight=weights[trn_idx],
            categorical_feature=categorical_features,
        )
        val_data = lgb.Dataset(
            data=features.iloc[val_idx],
            label=targets.iloc[val_idx],
            weight=weights[val_idx],
            reference=trn_data,
        )
        eval_result = {}
        callbacks = [
            lgb.early_stopping(STOPPING_ROUNDS),
            lgb.log_evaluation(VERBOSE_EVAL),
            lgb.record_evaluation(eval_result),
        ]
        booster = lgb.train(
            model_params,
            trn_data,
            valid_sets=[trn_data, val_data],
            valid_names=["trn", "val"],
            num_boost_round=NUM_BOOST_ROUND,
            feval=quadratic_weighted_kappa,
            callbacks=callbacks,
        )
        boosters.append(booster)
        val_qwks.append(
            eval_result["val"]["QWK"][booster.best_iteration - 1]
        )
        oof_preds[val_idx] = booster.predict(features.iloc[val_idx])
    print(f"Avg. Kappa: {np.mean(val_qwks):.4f}")
    return boosters, eval_result, val_qwks, oof_preds


def run_kfold(
        features, trn_targets, categorical_features, n_splits, model_params, save_dir, 
        weights=None, trn_id=None, tst_id=None
):
    trn_features = features.iloc[:len(trn_targets)]
    tst_features = features.iloc[len(trn_targets):]
    boosters, eval_result, qwks, oof_preds = train_kfold(
        trn_features, trn_targets, categorical_features, n_splits, model_params, weights
    )
    # save_loss
    loss = pd.DataFrame(
        data={
            "trn_rmse": eval_result["trn"]["rmse"],
            "val_rmse": eval_result["val"]["rmse"],
            "trn_qwk": eval_result["trn"]["QWK"],
            "val_qwk": eval_result["val"]["QWK"],
        }
    )
    loss.to_csv(save_dir / "log_metric.csv", index=False)
    # kappa_optimizer
    kappa_optimizer = minimize(
        evaluate_predictions,
        x0=[0.5, 1.5, 2.5],
        args=(trn_targets.to_numpy(), oof_preds),
        method="Nelder-Mead",
    )
    # oof_preds
    oof_preds = threshold_rounder(oof_preds, kappa_optimizer.x)
    qwk_optimized = cohen_kappa_score(trn_targets.to_numpy(), oof_preds, weights="quadratic")
    print(f"Optimized Kappa: {qwk_optimized:.4f}")
    oof_preds = pd.DataFrame(
        data={"id": trn_id, "sii": oof_preds}
    )
    oof_preds.to_csv(save_dir / "oof_prediction.csv", index=False)
    (save_dir / "score.txt").write_text(f"Kappa: {str(np.mean(qwks))}, Optimized Kappa: {str(qwk_optimized)}")
    # tst_preds
    tst_preds = np.zeros(len(tst_features))
    for i, booster in enumerate(boosters):
        tst_preds += booster.predict(tst_features) / n_splits
        booster.save_model(str(save_dir / f"booster.{i}.txt"))
    tst_preds = threshold_rounder(tst_preds, kappa_optimizer.x)
    tst_preds = pd.DataFrame(
        data={"id": tst_id, "sii": tst_preds}
    )
    tst_preds.to_csv("/kaggle/working/submission.csv", index=False)
    