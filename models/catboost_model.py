### File: models/catboost_model.py

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from typing import Optional, List, Union
from .base_model import BaseModel


class CatBoostModel(BaseModel):
    def __init__(
        self,
        params: Optional[dict] = None,
        cat_features: Optional[List[Union[int, str]]] = None,
        random_state: int = 42,
    ):
        super().__init__()
        self.params = params or {
            "iterations": 800,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "loss_function": "MultiClass",
            "eval_metric": "TotalF1",
            "random_seed": random_state,
            "verbose": False,
            "thread_count": -1,
            "early_stopping_rounds": 50,
        }
        self.cat_features = cat_features or []
        self.model = CatBoostClassifier(**self.params)
        self.feature_names_ = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        eval_set=None,
    ):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        self.feature_names_ = list(X.columns)

        # --- FIX CRÍTICO ---
        # Convertir columnas categóricas a string (CatBoost no acepta float)
        X_processed = X.copy()
        cat_idx = []

        if self.cat_features:
            for col in self.cat_features:
                if isinstance(col, int):
                    col_name = X.columns[col]
                else:
                    col_name = col

                if col_name in X_processed.columns:
                    # Convertir a string y rellenar NaN
                    X_processed[col_name] = (
                        X_processed[col_name]
                        .astype(str)
                        .replace({"nan": "missing", "None": "missing", "NaN": "missing"})
                    )
                    cat_idx.append(col_name)

        train_pool = Pool(
            data=X_processed,
            label=y,
            weight=sample_weight,
            cat_features=cat_idx if cat_idx else None,
        )

        if eval_set is not None:
            X_val, y_val = eval_set
            if isinstance(X_val, np.ndarray):
                X_val = pd.DataFrame(X_val, columns=self.feature_names_)

            X_val_processed = X_val.copy()
            for col in cat_idx:
                if col in X_val_processed.columns:
                    X_val_processed[col] = (
                        X_val_processed[col]
                        .astype(str)
                        .replace({"nan": "missing", "None": "missing", "NaN": "missing"})
                    )

            eval_pool = Pool(X_val_processed, label=y_val, cat_features=cat_idx)
            self.model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
        else:
            self.model.fit(train_pool)

        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_)

        X_processed = X.copy()
        for col in (self.cat_features or []):
            col_name = self.feature_names_[col] if isinstance(col, int) else col
            if col_name in X_processed.columns:
                X_processed[col_name] = (
                    X_processed[col_name]
                    .astype(str)
                    .replace({"nan": "missing", "None": "missing", "NaN": "missing"})
                )

        return self.model.predict_proba(X_processed)

    def get_feature_importance(self) -> pd.Series:
        if self.feature_names_ is None:
            return pd.Series(dtype=float)
        return pd.Series(
            self.model.get_feature_importance(),
            index=self.feature_names_,
        ).sort_values(ascending=False)
