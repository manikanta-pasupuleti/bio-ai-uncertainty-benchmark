from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_numeric_categorical_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [col for col in X.columns if col not in numeric_columns]
    return numeric_columns, categorical_columns


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    numeric_columns, categorical_columns = split_numeric_categorical_columns(X)
    transformers = []

    if numeric_columns:
        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median")),
        ]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError("The input data contains no columns that can be processed.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def transform_features(X_train: pd.DataFrame, X_valid: pd.DataFrame | None = None, X_test: pd.DataFrame | None = None, scale_numeric: bool = True):
    preprocessor = make_preprocessor(X_train, scale_numeric=scale_numeric)
    X_train_t = preprocessor.fit_transform(X_train)
    X_valid_t = preprocessor.transform(X_valid) if X_valid is not None else None
    X_test_t = preprocessor.transform(X_test) if X_test is not None else None
    return X_train_t, X_valid_t, X_test_t, preprocessor
