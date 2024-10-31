import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.metrics import confusion_matrix

from common.constant import TRAIN_CSV_PATH

def process_oof_prediction(exp_num):
    trn_df = pl.read_csv(TRAIN_CSV_PATH)
    base_dir = "/kaggle/temp"
    for dir_name in os.listdir(base_dir):
        if dir_name.startswith(exp_num):
            path = os.path.join(base_dir, dir_name, "oof_prediction.csv")
            oof_preds = pl.read_csv(path)
            oof_preds = (
                oof_preds
                .select(pl.col("id"), pl.col("sii").alias("preds"))
                .join(trn_df, on="id", how="left")
                .select("id", "sii", "preds", pl.exclude("id", "sii", "preds"))
            )
    return oof_preds


def plot_confusion_matrix(oof_preds):
    cm = confusion_matrix(oof_preds["sii"], oof_preds["preds"])
    plt.figure(figsize=(6, 3))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=oof_preds["preds"].unique(),
        yticklabels=oof_preds["sii"].unique(),
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")


def plot_importance(exp_num, width, height):
    base_dir = "/kaggle/temp"
    for dir_name in os.listdir(base_dir):
        if dir_name.startswith(exp_num):
            model_file = os.path.join(base_dir, dir_name, "model.0.txt")
            booster = lgb.Booster(model_file=model_file)
            ax = lgb.plot_importance(booster, figsize=(width, height))
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)
            plt.tight_layout()


def plot_learning_curve(exp_num):
    base_dir = "/kaggle/temp"
    for dir_name in os.listdir(base_dir):
        if dir_name.startswith(exp_num):
            path = os.path.join(base_dir, dir_name, "log_metric.csv")
            eval_result = pl.read_csv(path)
            plt.figure(figsize=(6, 3))
            plt.plot(eval_result["trn_qwk"].to_numpy(), label="trn")
            plt.plot(eval_result["val_qwk"].to_numpy(), label="val")
            plt.xlabel("iteration")
            plt.ylabel("Quadritic Weigted Kappa")
            plt.legend()
