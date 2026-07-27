import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from .base_model import BaseMatchModel

logger = logging.getLogger(__name__)


class CatBoostMatchModel(BaseMatchModel):
    """CatBoost multiclass classifier optimised for categorical league / team identifiers."""

    def __init__(
        self,
        iterations: int = 800,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        random_seed: int = 42,
        cat_features: Optional[List[str]] = None,
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.model: Optional[CatBoostClassifier] = None
        self.feature_names: list = []

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None) -> "CatBoostMatchModel":
        self.feature_names = list(X.columns)
        # Detect categorical columns that exist in the frame
        cat_idx = [i for i, c in enumerate(self.feature_names) if c in self.cat_features or c.endswith("_enc")]
        train_pool = Pool(X, label=y, weight=sample_weight, cat_features=cat_idx)

        self.model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            random_seed=self.random_seed,
            verbose=False,
            thread_count=-1,
        )
        self.model.fit(train_pool)
        logger.info("CatBoost trained on %d samples", len(X))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict_proba(X[self.feature_names])

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Nothing to save")
        self.model.save_model(path)
        import joblib
        joblib.dump({"feature_names": self.feature_names, "cat_features": self.cat_features}, path + ".meta")

    def load(self, path: str) -> "CatBoostMatchModel":
        self.model = CatBoostClassifier()
        self.model.load_model(path)
        import joblib
        meta = joblib.load(path + ".meta")
        self.feature_names = meta["feature_names"]
        self.cat_features = meta["cat_features"]
        return self
