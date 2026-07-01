import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class FeatureCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.constant_cols_ = [c for c in X.columns if X[c].nunique() <= 1]
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=self.constant_cols_, errors='ignore')

        # Handle DAYS_EMPLOYED sentinel
        if 'DAYS_EMPLOYED' in X.columns:
            X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace(365243, np.nan)

        return X


def build_preprocessing_pipeline():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    return pipeline
