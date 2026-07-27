import logging
import os
from typing import List, Optional

import pandas as pd

from config.settings import settings
from data_loader.sportradar_client import SportradarClient
from data_loader.data_processor import DataProcessor
from features.feature_engineering import FeatureEngineer
from features.feature_store import FeatureStore
from ensemble.super_stacking import SuperStackingEnsemble
from pipelines.evaluation import EvaluationReport

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates data ingestion → feature engineering → super-ensemble training → evaluation."""

    def __init__(
        self,
        competition_ids: List[str],
        min_season_year: int = 2018,
    ):
        self.competition_ids = competition_ids
        self.min_season_year = min_season_year
        self.client = SportradarClient()
        self.processor = DataProcessor(self.client)
        self.feature_engineer = FeatureEngineer()
        self.feature_store = FeatureStore()
        self.ensemble: Optional[SuperStackingEnsemble] = None

    def run(self, max_matches: Optional[int] = None) -> EvaluationReport:
        logger.info("=== Starting end-to-end training pipeline ===")

        # 1. Ingest
        matches = self.processor.fetch_historical_matches(
            self.competition_ids,
            min_season_year=self.min_season_year,
            max_matches=max_matches,
        )
        if matches.empty:
            raise RuntimeError("No matches retrieved – check competition IDs and API key.")

        # 2. Feature engineering
        X, y_1x2, y_goals = self.feature_engineer.fit_transform(matches)
        self.feature_store.save_features("train_features", X, y_1x2)

        # Align matches for Poisson (same index order)
        matches_aligned = matches.loc[X.index].reset_index(drop=True)
        X = X.reset_index(drop=True)
        y_1x2 = y_1x2.reset_index(drop=True)

        # 3. Super ensemble
        self.ensemble = SuperStackingEnsemble(
            n_splits=settings.N_TIME_SERIES_SPLITS,
            random_state=settings.RANDOM_SEED,
        )
        self.ensemble.fit(X, y_1x2, matches_for_poisson=matches_aligned)

        # 4. Persist models
        os.makedirs(settings.MODELS_DIR, exist_ok=True)
        for name, model in self.ensemble.base_models.items():
            path = os.path.join(settings.MODELS_DIR, f"{name}.model")
            model.save(path)
            logger.info("Saved %s → %s", name, path)

        # 5. Evaluation on last 20 % (temporal hold-out)
        split = int(len(X) * 0.8)
        X_test = X.iloc[split:]
        y_test = y_1x2.iloc[split:]
        matches_test = matches_aligned.iloc[split:]
        metrics = self.ensemble.evaluate(X_test, y_test, matches_for_poisson=matches_test)
        report = EvaluationReport(metrics)
        report.log()
        return report
