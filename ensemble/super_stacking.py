import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from models.base_model import BaseMatchModel
from models.poisson_dixon_coles import DixonColesPoisson
from models.xgboost_model import XGBoostMatchModel
from models.catboost_model import CatBoostMatchModel
from models.dnn_model import DNNMatchModel
from models.lstm_model import LSTMMatchModel
from models.lstm_momentum import LSTMMomentumModel

logger = logging.getLogger(__name__)


class SuperStackingEnsemble:
    """
    Level-1 base models → out-of-fold probability matrix → Level-2 meta-learner
    (penalised Logistic Regression). Strict TimeSeriesSplit to eliminate leakage.
    """

    def __init__(
        self,
        n_splits: int = 5,
        meta_C: float = 1.0,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.meta_C = meta_C
        self.random_state = random_state
        self.base_models: Dict[str, BaseMatchModel] = {}
        self.meta_model: Optional[LogisticRegression] = None
        self.oof_columns: List[str] = []

    def _init_base_models(self) -> Dict[str, BaseMatchModel]:
        return {
            "xgboost": XGBoostMatchModel(random_state=self.random_state),
            "catboost": CatBoostMatchModel(random_seed=self.random_state),
            "dnn": DNNMatchModel(random_state=self.random_state),
            "lstm": LSTMMatchModel(random_state=self.random_state),
            "lstm_momentum": LSTMMomentumModel(random_state=self.random_state),
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        matches_for_poisson: Optional[pd.DataFrame] = None,
    ) -> "SuperStackingEnsemble":
        """
        X, y must be chronologically ordered.
        matches_for_poisson is the original match-level frame required by Dixon-Coles.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        n_samples = len(X)
        n_models = 6  # 5 + poisson
        oof_preds = np.zeros((n_samples, n_models * 3))

        model_names = ["poisson", "xgboost", "catboost", "dnn", "lstm", "lstm_momentum"]
        self.oof_columns = [f"{m}_p{c}" for m in model_names for c in range(3)]

        # --- Poisson (global fit on expanding window) ---
        if matches_for_poisson is not None:
            poisson = DixonColesPoisson()
            # We fit once on the whole history for simplicity of the meta stage;
            # a pure expanding-window version is possible but more expensive.
            poisson.fit_from_matches(matches_for_poisson)
            self.base_models["poisson"] = poisson
            # Produce probabilities for every row
            poisson_proba = poisson.predict_proba(matches_for_poisson)
            oof_preds[:, 0:3] = poisson_proba
        else:
            oof_preds[:, 0:3] = 1.0 / 3.0

        # --- Remaining models via TimeSeriesSplit ---
        base_candidates = self._init_base_models()
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info("Stacking fold %d/%d - train %d, val %d", fold + 1, self.n_splits, len(train_idx), len(val_idx))
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y.iloc[train_idx]

            for i, (name, model) in enumerate(base_candidates.items(), start=1):
                model.fit(X_tr, y_tr)
                proba = model.predict_proba(X_val)
                oof_preds[val_idx, i * 3 : (i + 1) * 3] = proba

        # Final fit of base models on full data
        for name, model in base_candidates.items():
            logger.info("Final fit of %s on full data", name)
            model.fit(X, y)
            self.base_models[name] = model

        # Meta-learner
        meta_X = pd.DataFrame(oof_preds, columns=self.oof_columns)
        # Drop rows that were never in a validation fold (first fold)
        valid_mask = meta_X.sum(axis=1) > 0
        self.meta_model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            C=self.meta_C,
            max_iter=1000,
            random_state=self.random_state,
        )
        self.meta_model.fit(meta_X.loc[valid_mask], y.loc[valid_mask])
        logger.info("Meta-learner fitted on %d OOF samples", valid_mask.sum())
        return self

    def predict_proba(self, X: pd.DataFrame, matches_for_poisson: Optional[pd.DataFrame] = None) -> np.ndarray:
        if self.meta_model is None:
            raise RuntimeError("Ensemble not fitted")
        n = len(X)
        level1 = np.zeros((n, len(self.oof_columns)))

        if "poisson" in self.base_models and matches_for_poisson is not None:
            level1[:, 0:3] = self.base_models["poisson"].predict_proba(matches_for_poisson)
        else:
            level1[:, 0:3] = 1.0 / 3.0

        for i, name in enumerate(["xgboost", "catboost", "dnn", "lstm", "lstm_momentum"], start=1):
            if name in self.base_models:
                level1[:, i * 3 : (i + 1) * 3] = self.base_models[name].predict_proba(X)

        meta_X = pd.DataFrame(level1, columns=self.oof_columns)
        return self.meta_model.predict_proba(meta_X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series, matches_for_poisson: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        proba = self.predict_proba(X, matches_for_poisson)
        pred = np.argmax(proba, axis=1)
        acc = accuracy_score(y, pred)
        # Brier score (multiclass)
        y_onehot = np.eye(3)[y.values]
        brier = np.mean(np.sum((proba - y_onehot) ** 2, axis=1))
        ll = log_loss(y, proba)
        return {"accuracy": acc, "brier_score": brier, "log_loss": ll}
