import logging
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

from .base_model import BaseMatchModel

logger = logging.getLogger(__name__)


class XGBoostMatchModel(BaseMatchModel):
    """Multiclass 1X2 XGBoost classifier with optional expected-goals regression head."""

    def __init__(
        self,
        n_estimators: int = 600,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ):
        self.params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": -1,
        }
        self.n_estimators = n_estimators
        self.model: Optional[xgb.Booster] = None
        self.feature_names: list = []
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None) -> "XGBoostMatchModel":
        self.feature_names = list(X.columns)
        dtrain = xgb.DMatrix(X.values, label=y.values, weight=sample_weight, feature_names=self.feature_names)
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            verbose_eval=False,
        )
        self.feature_importances_ = self.model.get_score(importance_type="gain")
        logger.info("XGBoost trained on %d samples, %d features", len(X), len(self.feature_names))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        dtest = xgb.DMatrix(X[self.feature_names].values, feature_names=self.feature_names)
        return self.model.predict(dtest)

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Nothing to save")
        self.model.save_model(path)
        meta = {"feature_names": self.feature_names, "params": self.params}
        import joblib
        joblib.dump(meta, path + ".meta")

    def load(self, path: str) -> "XGBoostMatchModel":
        self.model = xgb.Booster()
        self.model.load_model(path)
        import joblib
        meta = joblib.load(path + ".meta")
        self.feature_names = meta["feature_names"]
        self.params = meta["params"]
        return self
