import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config.settings import settings
from .rolling_metrics import RollingMetricsCalculator

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """End-to-end feature pipeline that produces model-ready matrices with strict temporal ordering."""

    def __init__(self, windows: Optional[List[int]] = None):
        self.windows = windows or settings.ROLLING_WINDOWS
        self.rolling = RollingMetricsCalculator(self.windows)
        self.label_encoders: dict = {}

    def fit_transform(self, matches: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Returns:
            X: feature matrix
            y_1x2: multiclass target (0=home, 1=draw, 2=away)
            y_goals: total goals (for regression heads)
        """
        df = self.rolling.compute(matches)
        df = self._add_static_features(df)
        df = self._encode_categoricals(df, fit=True)
        df = df.dropna(subset=[c for c in df.columns if c.startswith(("home_roll_", "away_roll_"))])
        feature_cols = self._select_feature_columns(df)
        X = df[feature_cols].astype(np.float32)
        y_1x2 = df["result_1x2"].astype(int)
        y_goals = df["total_goals"].astype(np.float32)
        return X, y_1x2, y_goals

    def transform(self, matches: pd.DataFrame) -> pd.DataFrame:
        df = self.rolling.compute(matches)
        df = self._add_static_features(df)
        df = self._encode_categoricals(df, fit=False)
        feature_cols = self._select_feature_columns(df)
        return df[feature_cols].astype(np.float32)

    def _add_static_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["day_of_week"] = df["start_time"].dt.dayofweek
        df["month"] = df["start_time"].dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        for col in ["competition_id", "home_id", "away_id"]:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is None:
                    df[f"{col}_enc"] = 0
                else:
                    # unseen labels -> -1
                    known = set(le.classes_)
                    df[f"{col}_enc"] = df[col].astype(str).apply(lambda x: le.transform([x])[0] if x in known else -1)
        return df

    def _select_feature_columns(self, df: pd.DataFrame) -> List[str]:
        exclude = {
            "match_id", "season_id", "competition_id", "competition_name", "season_name",
            "start_time", "home_id", "away_id", "home_name", "away_name",
            "home_score", "away_score", "status", "match_status", "venue_id", "venue_name",
            "referee_id", "result_1x2", "total_goals",
        }
        return [c for c in df.columns if c not in exclude and not c.endswith("_name")]
