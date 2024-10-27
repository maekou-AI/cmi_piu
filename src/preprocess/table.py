import polars as pl


def label_edit(trn_df):
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


def fill_missing(df):
    update = (
        df
        # Physical-BMI
        .with_columns(
            pl.when(pl.col("Physical-BMI") == 0)
            .then(None)
            .otherwise(pl.col("Physical-BMI"))
            .alias("Physical-BMI")
        )
        .with_columns(
            pl.when(pl.col("Physical-BMI").is_null())
            .then(pl.col("BIA-BIA_BMI"))
            .otherwise(pl.col("Physical-BMI"))
            .alias("Physical-BMI")
        )
        # Physical-Weight
        .with_columns(
            pl.when(pl.col("Physical-Weight") == 0)
            .then(None)
            .otherwise(pl.col("Physical-Weight"))
            .alias("Physical-Weight")
        )
        .with_columns(
            pl.when(pl.col("Physical-Weight").is_null())
            .then(pl.col("BIA-BIA_FFM") + pl.col("BIA-BIA_Fat"))
            .otherwise(pl.col("Physical-Weight"))
            .alias("Physical-Weight")
        )
        # Physical-Height
        .with_columns(
            pl.when(pl.col("Physical-Height").is_null())
            .then((pl.col("Physical-Weight") * 0.45 / pl.col("Physical-BMI")).sqrt() * 100 / 2.54)
            .otherwise(pl.col("Physical-Height"))
            .alias("Physical-Height")   
        )
    )
    return update


def remove_outlier(df):
    update = (
        df
        # CGAS-CGAS_Score
        .with_columns(
            pl.when(pl.col("CGAS-CGAS_Score") > 100)
            .then(None)
            .otherwise(pl.col("CGAS-CGAS_Score"))
            .alias("CGAS-CGAS_Score")
        )
        # Physical-BMI
        .with_columns(
            pl.when(pl.col("Physical-BMI") == 0)
            .then(None)
            .otherwise(pl.col("Physical-BMI"))
            .alias("Physical-BMI")
        )
        # Physical-Weight
        .with_columns(
            pl.when(pl.col("Physical-Weight") == 0)
            .then(None)
            .otherwise(pl.col("Physical-Weight"))
            .alias("Physical-Weight")
        )
        # Physical-Diastolic_BP, Physical-Systolic_BP
        .with_columns(
            pl.when(pl.col("Physical-Diastolic_BP") == 0)
            .then(None)
            .when((pl.col("Physical-Systolic_BP") - pl.col("Physical-Diastolic_BP")) < 0)
            .then(None)
            .otherwise(pl.col("Physical-Diastolic_BP"))
            .alias("Physical-Diastolic_BP")
        )
        .with_columns(
            pl.when(pl.col("Physical-Systolic_BP") == 0)
            .then(None)
            .otherwise(pl.col("Physical-Systolic_BP"))
            .alias("Physical-Systolic_BP")
        )
        # BIA-BIA_BMC (Clipping)
        .with_columns(
            pl.when((pl.col("BIA-BIA_BMC") > 30) | (pl.col("BIA-BIA_BMC") < 0)).then(None)
            .when(pl.col("BIA-BIA_BMC") > 10).then(10)
            .when(pl.col("BIA-BIA_BMC") < 2).then(2)
            .otherwise(pl.col("BIA-BIA_BMC"))
            .alias("BIA-BIA_BMC")
        )
        # BIA-BIA_BMR (Clipping)
        .with_columns(
            pl.when(pl.col("BIA-BIA_BMR") > 10000).then(None)
            .when(pl.col("BIA-BIA_BMR") > 2000).then(2000)
            .otherwise(pl.col("BIA-BIA_BMR"))
            .alias("BIA-BIA_BMR")
        )
        # BIA-BIA_DEE (Recalculate)
        .with_columns(
            pl.when(pl.col("BIA-BIA_Activity_Level_num") == 1).then(pl.col("BIA-BIA_BMR") * 1.3)
            .when(pl.col("BIA-BIA_Activity_Level_num") == 2).then(pl.col("BIA-BIA_BMR") * 1.55)
            .when(pl.col("BIA-BIA_Activity_Level_num") == 3).then(pl.col("BIA-BIA_BMR") * 1.65)
            .when(pl.col("BIA-BIA_Activity_Level_num") == 4).then(pl.col("BIA-BIA_BMR") * 2.05)
            .when(pl.col("BIA-BIA_Activity_Level_num") == 5).then(pl.col("BIA-BIA_BMR") * 2.35)
            .otherwise(pl.col("BIA-BIA_DEE"))
            .alias("BIA-BIA_DEE")
        )        
        # BIA-BIA_TBW (Remove outlier)
        .with_columns(
            pl.when(pl.col("Physical-Weight") - pl.col("BIA-BIA_TBW") < 0).then(None)
            .otherwise(pl.col("BIA-BIA_TBW"))
            .alias("BIA-BIA_TBW")
        )
        # BIA-BIA_ECW (Remove outlier)
        .with_columns(
            pl.when(pl.col("BIA-BIA_TBW").is_null()).then(None)
            .otherwise(pl.col("BIA-BIA_ECW"))
            .alias("BIA-BIA_ECW")
        )
        # BIA-BIA_ICW (Remove outlier)
        .with_columns(
            pl.when(pl.col("BIA-BIA_TBW").is_null()).then(None)
            .otherwise(pl.col("BIA-BIA_ICW"))
            .alias("BIA-BIA_ICW")
        )
        # BIA-BIA_Fat
        .with_columns(
            pl.when(pl.col("BIA-BIA_Fat") < 0).then(None)
            .otherwise(pl.col("BIA-BIA_Fat"))
            .alias("BIA-BIA_Fat")
        )
        # BIA-BIA_FFM
        .with_columns(
            pl.when(pl.col("BIA-BIA_Fat").is_null()).then(None)
            .otherwise(pl.col("BIA-BIA_FFM"))
            .alias("BIA-BIA_FFM")
        )
        # BIA-BIA_FFMI
        .with_columns(
            pl.when(pl.col("BIA-BIA_Fat").is_null()).then(None)
            .otherwise(pl.col("BIA-BIA_FFMI"))
            .alias("BIA-BIA_FFMI")
        )
        # BIA-BIA_FMI
        .with_columns(
            pl.when(pl.col("BIA-BIA_Fat").is_null()).then(None)
            .otherwise(pl.col("BIA-BIA_FMI"))
            .alias("BIA-BIA_FMI")
        )
        # BIA-BIA_LDM (recalculate)
        .with_columns(
            (pl.col("BIA-BIA_FFM") - pl.col("BIA-BIA_TBW")).alias("BIA-BIA_LDM")
        )
        # BIA-BIA_LST (Remove outlier)
        .with_columns(
            pl.when((pl.col("BIA-BIA_FFM") - pl.col("BIA-BIA_LST") < 0) | (pl.col("Physical-Weight") - pl.col("BIA-BIA_LST") < 0))
            .then(None)
            .otherwise(pl.col("BIA-BIA_LST"))
            .alias("BIA-BIA_LST")
        )
        # BIA-BIA_SMM
        .with_columns(
            pl.when(pl.col("Physical-Weight") - pl.col("BIA-BIA_SMM") < 0)
            .then(None)
            .otherwise(pl.col("BIA-BIA_SMM"))
            .alias("BIA-BIA_SMM")
        )
    )
    return update