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
from pipelines.roi_simulator import ROISimulator

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    End-to-end orchestration:
        1. Sportradar ingestion (rate-limited + cached)
        2. Leakage-free feature engineering
        3. Super-ensemble training under TimeSeriesSplit
        4. Temporal hold-out evaluation + ROI simulation
        5. Model persistence
    """

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

        # ------------------------------------------------------------------
        # 1. Ingest historical matches
        # ------------------------------------------------------------------
        matches = self.processor.fetch_historical_matches(
            competition_ids=self.competition_ids,
            min_season_year=self.min_season_year,
            max_matches=max_matches,
        )
        if matches.empty:
            raise RuntimeError(
                "No matches retrieved. Verify competition IDs and that "
                "SPORTRADAR_API_KEY is correctly set."
            )
        logger.info("Ingested %d matches from %s", len(matches), self.competition_ids)

        # ------------------------------------------------------------------
        # 2. Feature engineering (strictly causal rolling metrics)
        # ------------------------------------------------------------------
        X, y_1x2, y_goals = self.feature_engineer.fit_transform(matches)
        self.feature_store.save_features("train_features", X, y_1x2)

        # Keep matches aligned with the feature matrix index
        matches_aligned = matches.loc[X.index].reset_index(drop=True)
        X = X.reset_index(drop=True)
        y_1x2 = y_1x2.reset_index(drop=True)

        logger.info(
            "Feature matrix shape: %s  |  target distribution: %s",
            X.shape,
            y_1x2.value_counts(normalize=True).to_dict(),
        )

        # ------------------------------------------------------------------
        # 3. Super-ensemble training
        # ------------------------------------------------------------------
        self.ensemble = SuperStackingEnsemble(
            n_splits=settings.N_TIME_SERIES_SPLITS,
            random_state=settings.RANDOM_SEED,
        )
        self.ensemble.fit(X, y_1x2, matches_for_poisson=matches_aligned)

        # ------------------------------------------------------------------
        # 4. Persist the whole ensemble
        # ------------------------------------------------------------------
        os.makedirs(settings.MODELS_DIR, exist_ok=True)
        self.ensemble.save(settings.MODELS_DIR)
        logger.info("All models persisted under %s", settings.MODELS_DIR)

        # ------------------------------------------------------------------
        # 5. Temporal hold-out evaluation (last 20 %)
        # ------------------------------------------------------------------
        split_idx = int(len(X) * 0.80)
        X_test = X.iloc[split_idx:]
        y_test = y_1x2.iloc[split_idx:]
        matches_test = matches_aligned.iloc[split_idx:]

        metrics = self.ensemble.evaluate(X_test, y_test, matches_for_poisson=matches_test)
        logger.info("===== Out-of-sample metrics =====")
        for k, v in metrics.items():
            logger.info("%-15s : %.4f", k, v)

        # ------------------------------------------------------------------
        # 6. ROI simulation on the same hold-out window
        # ------------------------------------------------------------------
        proba_test = self.ensemble.predict_proba(X_test, matches_for_poisson=matches_test)
        roi_sim = ROISimulator(bankroll=10_000.0, flat_stake=50.0, kelly_fraction=0.25)
        roi_metrics = roi_sim.simulate(y_test.values, proba_test)

        logger.info("===== ROI Simulation (hold-out) =====")
        for k, v in roi_metrics.items():
            logger.info("%-22s : %s", k, v)

        # Merge metrics for the final report
        full_metrics = {**metrics, **{f"roi_{k}": v for k, v in roi_metrics.items()}}
        report = EvaluationReport(full_metrics)
        report.log()
        return report
