import polars as pl


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
            df.select("age-sex"),
            age_norm_features,
            sex_norm_features,
            age_sex_norm_features,
        ],
        how="horizontal"
    )
    return features