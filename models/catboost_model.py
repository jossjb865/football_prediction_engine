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

    def _sanitize_cat_column(self, series: pd.Series) -> pd.Series:
        """Limpia y convierte una columna categórica a string seguro para CatBoost."""
        # Convertir flotantes enteros (ej. 242.0) a enteros primero para evitar '242.0' como string
        s = series.copy()
        
        # Rellenar valores nulos antes de pasar a string
        s = s.fillna("missing")
        
        # Convertir a string y formatear de forma limpia
        def clean_val(val):
            if val == "missing":
                return "missing"
            if isinstance(val, float) and val.is_integer():
                return str(int(val))
            return str(val).strip()

        return s.apply(clean_val)

    def _get_cat_col_names(self, df: pd.DataFrame) -> List[str]:
        """Obtiene la lista de nombres de columnas categóricas."""
        cat_cols = []
        for col in self.cat_features:
            if isinstance(col, int):
                if col < len(df.columns):
                    cat_cols.append(df.columns[col])
            else:
                if col in df.columns:
                    cat_cols.append(col)
        return list(set(cat_cols))

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
        X_processed = X.copy()

        cat_cols = self._get_cat_col_names(X_processed)

        # Aplicar saneamiento a columnas categóricas en Train
        for col in cat_cols:
            X_processed[col] = self._sanitize_cat_column(X_processed[col])

        train_pool = Pool(
            data=X_processed,
            label=y,
            weight=sample_weight,
            cat_features=cat_cols if cat_cols else None,
        )

        if eval_set is not None:
            X_val, y_val = eval_set
            if isinstance(X_val, np.ndarray):
                X_val = pd.DataFrame(X_val, columns=self.feature_names_)

            X_val_processed = X_val.copy()
            for col in cat_cols:
                if col in X_val_processed.columns:
                    X_val_processed[col] = self._sanitize_cat_column(X_val_processed[col])

            eval_pool = Pool(
                data=X_val_processed, 
                label=y_val, 
                cat_features=cat_cols if cat_cols else None
            )
            self.model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
        else:
            self.model.fit(train_pool)

        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_)

        X_processed = X.copy()
        cat_cols = self._get_cat_col_names(X_processed)

        for col in cat_cols:
            if col in X_processed.columns:
                X_processed[col] = self._sanitize_cat_column(X_processed[col])

        return self.model.predict_proba(X_processed)

    def get_feature_importance(self) -> pd.Series:
        if self.feature_names_ is None:
            return pd.Series(dtype=float)
        return pd.Series(
            self.model.get_feature_importance(),
            index=self.feature_names_,
        ).sort_values(ascending=False)
