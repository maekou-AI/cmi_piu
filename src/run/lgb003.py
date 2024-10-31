import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold


NUM_BOOST_ROUND = 5000
STOPPING_ROUNDS = 100
VERBOSE_EVAL = 50


def quadratic_weighted_kappa(preds, data):
    y_true = data.get_label()
    y_pred = preds.clip(0, 3).round()
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return 'QWK', qwk, True


def qwk_obj(preds, dtrain):
    a = 0.5804093567251462
    b = 0.5944115283335043
    labels = dtrain.get_label()
    preds = preds.clip(0, 3)
    f = 1/2 * np.sum((preds - labels)**2)
    g = 1/2 * np.sum((preds - a)**2 + b)
    df = preds - labels
    dg = preds - a
    grad = (df/g - f*dg/g**2)*len(labels)
    hess = np.ones(len(labels))
    return grad, hess


def train_kfold(features, targets, categorical_features, n_splits, model_params, init_score):
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    folds = [(trn_idx, val_idx) for trn_idx, val_idx in skf.split(features, targets)]
    eval_result = {}
    
    cv = lgb.cv(
        params=model_params,
        train_set=lgb.Dataset(
            features, targets, init_score=[init_score]*len(features)
        ),
        categorical_feature=categorical_features,
        num_boost_round=NUM_BOOST_ROUND,
        folds=folds,
        feval=quadratic_weighted_kappa,
        callbacks=[
            lgb.early_stopping(STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(VERBOSE_EVAL),
            lgb.record_evaluation(eval_result),
        ],
        eval_train_metric=True,
        return_cvbooster=True,
    )
    print(f"Kappa: {cv['valid QWK-mean'][-1]}")
    boosters = cv["cvbooster"].boosters

    oof_preds = np.zeros(len(features))
    for booster, (_, val_idx) in zip(boosters, folds):
        oof_preds[val_idx] = booster.predict(features.iloc[val_idx]) + init_score
    oof_preds = oof_preds.clip(0, 3).round()

    return boosters, eval_result, oof_preds


def run_kfold(
        features, trn_targets, categorical_features, n_splits, model_params, save_dir, trn_id=None, tst_id=None
):
    trn_features = features.iloc[:len(trn_targets)]
    tst_features = features.iloc[len(trn_targets):]
    
    init_score = 2.0
    boosters, eval_result, oof_preds = train_kfold(
        trn_features, trn_targets, categorical_features, n_splits, model_params, init_score
    )

    # save qwk score transition
    score = pd.DataFrame(
        data={
            "trn_qwk": eval_result["train"]["QWK-mean"],
            "val_qwk": eval_result["valid"]["QWK-mean"],
        }
    )
    score.to_csv(save_dir / "log_metric.csv", index=False)

    # save oof_preds
    pd.DataFrame(data={"id": trn_id, "sii": oof_preds}).to_csv(save_dir / "oof_prediction.csv", index=False)

    # save submission.csv
    tst_preds = np.zeros(len(tst_features))
    for i, booster in enumerate(boosters):
        tst_preds += booster.predict(tst_features) / n_splits
        booster.save_model(str(save_dir / f"booster.{i}.txt"))
    tst_preds = (tst_preds + init_score).clip(0, 3).round()
    pd.DataFrame(data={"id": tst_id, "sii": tst_preds}).to_csv("/kaggle/working/submission.csv", index=False)