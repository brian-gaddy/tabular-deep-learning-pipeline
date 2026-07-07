import pandas as pd

from src.features.preprocessing import FeatureCleaner, build_preprocessing_pipeline


def test_feature_cleaner_removes_id_constant_and_days_employed_sentinel():
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "constant": [1, 1],
            "DAYS_EMPLOYED": [365243, -1000],
            "AMT_INCOME_TOTAL": [100000, 120000],
        }
    )

    cleaner = FeatureCleaner()
    transformed = cleaner.fit_transform(df)

    assert "SK_ID_CURR" not in transformed.columns
    assert "constant" not in transformed.columns
    assert transformed["DAYS_EMPLOYED"].isna().sum() == 1


def test_preprocessing_pipeline_returns_numeric_matrix_for_mixed_data():
    df = pd.DataFrame(
        {
            "numeric_feature": [1.0, None, 3.0],
            "categorical_feature": ["A", "B", "A"],
            "DAYS_EMPLOYED": [365243, -500, -1000],
        }
    )

    pipeline = build_preprocessing_pipeline(df)
    transformed = pipeline.fit_transform(df)

    assert transformed.shape[0] == 3
    assert transformed.shape[1] >= 3
