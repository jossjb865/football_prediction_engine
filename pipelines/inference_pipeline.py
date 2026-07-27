import logging
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from config.settings import settings
from data_loader.sportradar_client import SportradarClient
from data_loader.data_processor import DataProcessor
from features.feature_engineering import FeatureEngineer
from ensemble.super_stacking import SuperStackingEnsemble
from models.poisson_dixon_coles import DixonColesPoisson
from models.xgboost_model import XGBoostMatchModel
from models.catboost_model import CatBoostMatchModel
from models.dnn_model import DNNMatchModel
from models.lstm_model import LSTMMatchModel
from models.lstm_momentum import LSTMMomentumModel

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Loads a previously trained SuperStackingEnsemble and produces
    1X2 probabilities for a list of upcoming (or historical) matches.
    """

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = models_dir or settings.MODELS_DIR
        self.client = SportradarClient()
        self.processor = DataProcessor(self.client)
        self.feature_engineer = FeatureEngineer()
        self.ensemble = SuperStackingEnsemble()
        self._load_models()

    def _load_models(self) -> None:
        mapping = {
            "poisson": (DixonColesPoisson, "poisson.model"),
            "xgboost": (XGBoostMatchModel, "xgboost.model"),
            "catboost": (CatBoostMatchModel, "catboost.model"),
            "dnn": (DNNMatchModel, "dnn.model"),
            "lstm": (LSTMMatchModel, "lstm.model"),
            "lstm_momentum": (LSTMMomentumModel, "lstm_momentum.model"),
        }
        for name, (cls, fname) in mapping.items():
            path = os.path.join(self.models_dir, fname)
            if not os.path.exists(path):
                logger.warning("Model file missing: %s – skipping", path)
                continue
            model = cls()
            model.load(path)
            self.ensemble.base_models[name] = model
            logger.info("Loaded %s", name)

        meta_path = os.path.join(self.models_dir, "meta_learner.joblib")
        if os.path.exists(meta_path):
            self.ensemble.meta_model = joblib.load(meta_path)
            self.ensemble.oof_columns = joblib.load(os.path.join(self.models_dir, "oof_columns.joblib"))
            logger.info("Meta-learner loaded")
        else:
            logger.warning("Meta-learner not found – predictions will be simple average of base models")

    def predict_matches(self, matches: pd.DataFrame) -> pd.DataFrame:
        """
        matches must contain at least: match_id, start_time, home_id, away_id
        and (for historical evaluation) home_score / away_score.
        """
        X = self.feature_engineer.transform(matches)
        # Align
        matches = matches.loc[X.index].reset_index(drop=True)
        X = X.reset_index(drop=True)

        if self.ensemble.meta_model is not None:
            proba = self.ensemble.predict_proba(X, matches_for_poisson=matches)
        else:
            # Fallback: average of available base models
            preds = []
            for name, model in self.ensemble.base_models.items():
                if name in ("lstm", "lstm_momentum"):
                    p = model.predict_proba(X, matches=matches)
                elif name == "poisson":
                    p = model.predict_proba(matches)
                else:
                    p = model.predict_proba(X)
                preds.append(p)
            proba = np.mean(preds, axis=0)

        result = matches[["match_id", "home_id", "away_id", "home_name", "away_name", "start_time"]].copy()
        result["p_home"] = proba[:, 0]
        result["p_draw"] = proba[:, 1]
        result["p_away"] = proba[:, 2]
        result["prediction"] = np.argmax(proba, axis=1)
        result["confidence"] = proba.max(axis=1)
        return result
