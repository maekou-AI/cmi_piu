import polars as pl


def edit(trn_df):
    pciat_cols = [f"PCIAT-PCIAT_0{i}" for i in range(1, 10)] + [f"PCIAT-PCIAT_{i}" for i in range(10, 21)]
    rm_id = ["053d7d31", "6b9a25e6"]
    update = (
        trn_df
        .with_columns(
            pl.sum_horizontal(pl.col(pciat_cols).is_null()).alias("num_nulls")
        )
        .with_columns(
            pl.when(pl.col("id").is_in(rm_id))
            .then(None)
            .otherwise(pl.col("PCIAT-PCIAT_Total") + pl.col("num_nulls") * 3)
            .alias("PCIAT-PCIAT_Total")
        )
        .with_columns(
            pl.when(pl.col("PCIAT-PCIAT_Total").is_between(0, 30)).then(0)
            .when(pl.col("PCIAT-PCIAT_Total").is_between(31, 49)).then(1)
            .when(pl.col("PCIAT-PCIAT_Total").is_between(50, 79)).then(2)
            .when(pl.col("PCIAT-PCIAT_Total") >= 80).then(3)
            .otherwise(None)
            .alias("sii")
        )
    )
    for col in pciat_cols:
        mean = trn_df.group_by("sii").agg(pl.col(col).mean().alias(f"{col}_mean"))
        mean_dict = {sii: mean for sii, mean in mean.iter_rows()}
        update = update.with_columns(
            pl.when((pl.col(col).is_null()) & (pl.col("sii") == 0)).then(round(mean_dict[0]))
            .when((pl.col(col).is_null()) & (pl.col("sii") == 1)).then(round(mean_dict[1]))
            .when((pl.col(col).is_null()) & (pl.col("sii") == 2)).then(round(mean_dict[2]))
            .when((pl.col(col).is_null()) & (pl.col("sii") == 3)).then(round(mean_dict[3]))
            .otherwise(pl.col(col))
            .alias(col)
        )
    update = update.drop("num_nulls")
    return update