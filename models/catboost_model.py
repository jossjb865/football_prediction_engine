import logging
from typing import List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from .base_model import BaseMatchModel

logger = logging.getLogger(__name__)


class CatBoostMatchModel(BaseMatchModel):
    """
    Multiclass 1X2 CatBoost model with robust handling of categorical features.
    Automatically converts float / NaN categorical columns to safe strings.
    """

    def __init__(
        self,
        params: Optional[dict] = None,
        cat_features: Optional[List[Union[int, str]]] = None,
        random_seed: int = 42,
    ):
        self.params = params or {
            "iterations": 800,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "loss_function": "MultiClass",
            "eval_metric": "TotalF1",
            "random_seed": random_seed,
            "verbose": False,
            "thread_count": -1,
            "early_stopping_rounds": 50,
            "allow_writing_files": False,
        }
        self.cat_features = cat_features or []
        self.model: Optional[CatBoostClassifier] = None
        self.feature_names_: List[str] = []
        self.cat_col_names_: List[str] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sanitize_cat_column(self, series: pd.Series) -> pd.Series:
        """Convert a column to a clean string representation safe for CatBoost."""
        s = series.copy()
        s = s.fillna("missing")

        def clean_val(val):
            if val == "missing":
                return "missing"
            if isinstance(val, (float, np.floating)) and val.is_integer():
                return str(int(val))
            return str(val).strip()

        return s.apply(clean_val)

    def _resolve_cat_columns(self, df: pd.DataFrame) -> List[str]:
        """Resolve cat_features (indexes or names) to actual column names."""
        cat_cols = []
        for col in self.cat_features:
            if isinstance(col, int):
                if 0 <= col < len(df.columns):
                    cat_cols.append(df.columns[col])
            else:
                if col in df.columns:
                    cat_cols.append(col)
        return list(dict.fromkeys(cat_cols))  # preserve order, remove duplicates

    def _prepare_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with categorical columns sanitized."""
        X_proc = X.copy()
        for col in self.cat_col_names_:
            if col in X_proc.columns:
                X_proc[col] = self._sanitize_cat_column(X_proc[col])
        return X_proc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        eval_set=None,
    ) -> "CatBoostMatchModel":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        self.feature_names_ = list(X.columns)
        self.cat_col_names_ = self._resolve_cat_columns(X)

        X_train = self._prepare_dataframe(X)

        train_pool = Pool(
            data=X_train,
            label=y,
            weight=sample_weight,
            cat_features=self.cat_col_names_ if self.cat_col_names_ else None,
        )

        self.model = CatBoostClassifier(**self.params)

        if eval_set is not None:
            X_val, y_val = eval_set
            if not isinstance(X_val, pd.DataFrame):
                X_val = pd.DataFrame(X_val, columns=self.feature_names_)
            X_val = self._prepare_dataframe(X_val)
            eval_pool = Pool(
                data=X_val,
                label=y_val,
                cat_features=self.cat_col_names_ if self.cat_col_names_ else None,
            )
            self.model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
        else:
            self.model.fit(train_pool)

        logger.info(
            "CatBoost trained on %d samples, %d features (%d categorical)",
            len(X),
            len(self.feature_names_),
            len(self.cat_col_names_),
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)

        # Ensure same columns order
        X = X[self.feature_names_]
        X_proc = self._prepare_dataframe(X)

        return self.model.predict_proba(X_proc)

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Nothing to save")
        self.model.save_model(path)
        meta = {
            "feature_names": self.feature_names_,
            "cat_col_names": self.cat_col_names_,
            "params": self.params,
        }
        joblib.dump(meta, path + ".meta")

    def load(self, path: str) -> "CatBoostMatchModel":
        self.model = CatBoostClassifier()
        self.model.load_model(path)
        meta = joblib.load(path + ".meta")
        self.feature_names_ = meta["feature_names"]
        self.cat_col_names_ = meta["cat_col_names"]
        self.params = meta["params"]
        return self

    def get_feature_importance(self) -> pd.Series:
        if self.model is None or not self.feature_names_:
            return pd.Series(dtype=float)
        return pd.Series(
            self.model.get_feature_importance(),
            index=self.feature_names_,
        ).sort_values(ascending=False)
