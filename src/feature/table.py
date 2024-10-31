import polars as pl


def interaction(df):
    features = (
        df
        .with_columns(
            (pl.col("Basic_Demos-Age") * pl.col("Physical-BMI")).alias("Age_BMI"),
            (pl.col("Basic_Demos-Age") * pl.col("PreInt_EduHx-computerinternet_hoursday")).alias("Age_internet_hoursday"),
            (pl.col("Physical-BMI") * pl.col("PreInt_EduHx-computerinternet_hoursday")).alias("BMI_internet_hoursday"),
        )
        .with_columns(
            (pl.col("BIA-BIA_Fat") / pl.col("BIA-BIA_BMI")).alias("fat_bmi"),
            (pl.col("BIA-BIA_FFMI") / pl.col("BIA-BIA_Fat")).alias("ffmi_fat"),
            (pl.col("BIA-BIA_FMI") / pl.col("BIA-BIA_Fat")).alias("fmi_fat"),
            (pl.col("BIA-BIA_LST") / pl.col("BIA-BIA_TBW")).alias("lst_tbw"),
            (pl.col("BIA-BIA_Fat") / pl.col("BIA-BIA_BMR")).alias("bfp_bmr"),
            (pl.col("BIA-BIA_Fat") / pl.col("BIA-BIA_DEE")).alias("fat_dee"),
            (pl.col("BIA-BIA_BMR") / pl.col("Physical-Weight")).alias("bmr_weight"),
            (pl.col("BIA-BIA_DEE") / pl.col("Physical-Weight")).alias("dee_weight"),
            (pl.col("BIA-BIA_SMM") / pl.col("Physical-Height")).alias("smm_height"),
            (pl.col("BIA-BIA_SMM") / pl.col("BIA-BIA_Fat")).alias("smm_fat"),
            (pl.col("BIA-BIA_TBW") / pl.col("Physical-Weight")).alias("tbw_weight"),
            (pl.col("BIA-BIA_ICW") / pl.col("BIA-BIA_TBW")).alias("icw_tbw")
        )
    )
    features_cols = [
        "Age_BMI", "Age_internet_hoursday", "BMI_internet_hoursday", 
        "fat_bmi", "ffmi_fat", "fmi_fat", "lst_tbw", "bfp_bmr", "fat_dee", "bmr_weight", "dee_weight",
        "smm_height", "smm_fat", "tbw_weight", "icw_tbw",
    ]
    return features.select(features_cols)


def normalized(df, norm_col, target_col, prefix):
    for col in target_col:
        median = df.group_by(norm_col).agg(pl.col(col).median().alias(f"median_{col}"))
        std = df.group_by(norm_col).agg(pl.col(col).std().alias(f"std_{col}"))
        df = (
            df
            .join(median, on=norm_col, how="left")
            .join(std, on=norm_col, how="left")
            .with_columns(
                ((pl.col(col) - pl.col(f"median_{col}")) / pl.col(f"std_{col}")).alias(f"{prefix}_{col}")
            )
            .drop(f"median_{col}", f"std_{col}")
        )
    return df.select([f"{prefix}_{col}" for col in target_col])


def build_feature(df):
    # Make age-sex column
    df = (
        df
        .with_columns(
            pl.when(pl.col("Basic_Demos-Sex") == 0)
            .then(pl.col("Basic_Demos-Age").cast(pl.Utf8) + "_male")
            .when(pl.col("Basic_Demos-Sex") == 1)
            .then(pl.col("Basic_Demos-Age").cast(pl.Utf8) + "_female")
            .otherwise(None)
            .alias("age-sex")
        )
    )
    # Normalized by age
    age_norm_cols = ["Physical-BMI", "Physical-HeartRate", "Physical-Systolic_BP"]
    age_norm_features = normalized(df, "Basic_Demos-Age", age_norm_cols, "age")

    # Normalized by sex
    sex_norm_cols = ["FGC-FGC_SRL", "FGC-FGC_SRR"]
    sex_norm_features = normalized(df, "Basic_Demos-Sex", sex_norm_cols, "sex")

    # Normalized by age-sex
    age_sex_norm_cols = [
        "Physical-Weight", "Physical-Waist_Circumference", "FGC-FGC_CU", "FGC-FGC_GSND", "FGC-FGC_GSD",
        "FGC-FGC_PU", "FGC-FGC_TL", "BIA-BIA_BMC", "BIA-BIA_BMR", "BIA-BIA_DEE", "BIA-BIA_ECW", "BIA-BIA_FFM",
        "BIA-BIA_FFMI", "BIA-BIA_FMI", "BIA-BIA_Fat", "BIA-BIA_ICW", "BIA-BIA_TBW", "BIA-BIA_LDM",
        "BIA-BIA_LST", "BIA-BIA_SMM"
    ]
    age_sex_norm_features = normalized(df, "age-sex", age_sex_norm_cols, "age-sex")

    features = pl.concat(
        [
            interaction(df),
            # age_norm_features,
            # sex_norm_features,
            # age_sex_norm_features,
        ],
        how="horizontal"
    )
    return features