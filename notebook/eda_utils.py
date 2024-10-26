import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


def boxplot_sii(df, col):
    plt.figure(figsize=(4, 4))
    sns.boxplot(x="sii", y=col, data=df.to_pandas())


def lineplot_age(df, col):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.lineplot(x="Basic_Demos-Age", y=col, data=df.to_pandas(), ax=axes[0])
    axes[0].set_title("All")
    sns.lineplot(x="Basic_Demos-Age", y=col, data=df.filter(pl.col("Basic_Demos-Sex") == 0).to_pandas(), ax=axes[1])
    sns.lineplot(x="Basic_Demos-Age", y=col, data=df.filter(pl.col("Basic_Demos-Sex") == 1).to_pandas(), ax=axes[1])
    axes[1].set_title("Sex")
