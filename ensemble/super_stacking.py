import logging
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss

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
    (penalised multinomial Logistic Regression).

    Strict TimeSeriesSplit is used exclusively to eliminate temporal data leakage.
    LSTM-family models receive the original matches DataFrame so they can build
    causal sequences via SequenceBuilder.
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
        X and y must be chronologically ordered.
        matches_for_poisson is the original match-level frame required by
        Dixon-Coles and by the LSTM-family SequenceBuilder.
        """
        if matches_for_poisson is None:
            raise ValueError("matches_for_poisson DataFrame is mandatory for SuperStackingEnsemble")

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        n_samples = len(X)
        model_names = ["poisson", "xgboost", "catboost", "dnn", "lstm", "lstm_momentum"]
        n_models = len(model_names)
        oof_preds = np.zeros((n_samples, n_models * 3), dtype=np.float64)
        self.oof_columns = [f"{m}_p{c}" for m in model_names for c in range(3)]

        # ------------------------------------------------------------------
        # 1. Dixon-Coles Poisson (global fit on the provided history)
        # ------------------------------------------------------------------
        poisson = DixonColesPoisson()
        poisson.fit_from_matches(matches_for_poisson)
        self.base_models["poisson"] = poisson
        poisson_proba = poisson.predict_proba(matches_for_poisson)
        oof_preds[:, 0:3] = poisson_proba

        # ------------------------------------------------------------------
        # 2. Remaining models via expanding TimeSeriesSplit (true OOF)
        # ------------------------------------------------------------------
        base_candidates = self._init_base_models()

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(
                "Stacking fold %d/%d – train=%d  val=%d",
                fold + 1, self.n_splits, len(train_idx), len(val_idx),
            )
            X_tr = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_tr = y.iloc[train_idx]
            matches_tr = matches_for_poisson.iloc[train_idx]
            matches_val = matches_for_poisson.iloc[val_idx]

            for i, (name, model) in enumerate(base_candidates.items(), start=1):
                if name in ("lstm", "lstm_momentum"):
                    model.fit(X_tr, y_tr, matches=matches_tr)
                    proba = model.predict_proba(X_val, matches=matches_val)
                else:
                    model.fit(X_tr, y_tr)
                    proba = model.predict_proba(X_val)
                oof_preds[val_idx, i * 3 : (i + 1) * 3] = proba

        # ------------------------------------------------------------------
        # 3. Final fit of every base model on the full data
        # ------------------------------------------------------------------
        for name, model in base_candidates.items():
            logger.info("Final full-data fit of %s", name)
            if name in ("lstm", "lstm_momentum"):
                model.fit(X, y, matches=matches_for_poisson)
            else:
                model.fit(X, y)
            self.base_models[name] = model

        # ------------------------------------------------------------------
        # 4. Meta-learner on out-of-fold probabilities
        # ------------------------------------------------------------------
        meta_X = pd.DataFrame(oof_preds, columns=self.oof_columns)
        # The first fold never receives OOF predictions → mask them out
        valid_mask = meta_X.sum(axis=1) > 1e-8
        self.meta_model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            C=self.meta_C,
            max_iter=2000,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.meta_model.fit(meta_X.loc[valid_mask], y.loc[valid_mask])
        logger.info("Meta-learner fitted on %d OOF samples", int(valid_mask.sum()))
        return self

    def predict_proba(
        self,
        X: pd.DataFrame,
        matches_for_poisson: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        if self.meta_model is None:
            raise RuntimeError("Ensemble has not been fitted")
        if matches_for_poisson is None:
            raise ValueError("matches_for_poisson is required at prediction time")

        n = len(X)
        level1 = np.zeros((n, len(self.oof_columns)), dtype=np.float64)

        # Poisson
        level1[:, 0:3] = self.base_models["poisson"].predict_proba(matches_for_poisson)

        # Remaining models
        for i, name in enumerate(["xgboost", "catboost", "dnn", "lstm", "lstm_momentum"], start=1):
            model = self.base_models[name]
            if name in ("lstm", "lstm_momentum"):
                proba = model.predict_proba(X, matches=matches_for_poisson)
            else:
                proba = model.predict_proba(X)
            level1[:, i * 3 : (i + 1) * 3] = proba

        meta_X = pd.DataFrame(level1, columns=self.oof_columns)
        return self.meta_model.predict_proba(meta_X)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        matches_for_poisson: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        proba = self.predict_proba(X, matches_for_poisson)
        pred = np.argmax(proba, axis=1)
        acc = accuracy_score(y, pred)
        y_onehot = np.eye(3)[y.values]
        brier = float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))
        ll = log_loss(y, proba)
        return {
            "accuracy": float(acc),
            "brier_score": brier,
            "log_loss": float(ll),
        }

    def save(self, models_dir: str) -> None:
        os.makedirs(models_dir, exist_ok=True)
        for name, model in self.base_models.items():
            path = os.path.join(models_dir, f"{name}.model")
            model.save(path)
            logger.info("Saved base model %s → %s", name, path)
        if self.meta_model is not None:
            joblib.dump(self.meta_model, os.path.join(models_dir, "meta_learner.joblib"))
            joblib.dump(self.oof_columns, os.path.join(models_dir, "oof_columns.joblib"))
            logger.info("Saved meta-learner and OOF column list")
