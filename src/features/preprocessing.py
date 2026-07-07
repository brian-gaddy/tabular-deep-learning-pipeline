import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureCleaner(BaseEstimator, TransformerMixin):
    """Clean tabular features before model preprocessing.

    The transformer removes identifier-style leakage columns, drops constant
    features learned from the training data, and handles the Home Credit
    DAYS_EMPLOYED sentinel value.
    """

    def __init__(self, id_columns=None):
        self.id_columns = id_columns or ["SK_ID_CURR"]

    def fit(self, X, y=None):
        X = X.copy()
        self.constant_cols_ = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=self.id_columns, errors="ignore")
        X = X.drop(columns=getattr(self, "constant_cols_", []), errors="ignore")

        if "DAYS_EMPLOYED" in X.columns:
            X["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].replace(365243, np.nan)

        return X


def build_preprocessing_pipeline(X: pd.DataFrame):
    """Build a preprocessing pipeline for mixed numeric/categorical tabular data."""
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("cleaner", FeatureCleaner()),
            ("preprocessor", preprocessor),
        ]
    )
