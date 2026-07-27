import os
import joblib
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import settings


class FeatureStore:
    """Simple on-disk feature store for train/inference consistency."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or settings.FEATURES_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_features(self, name: str, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        path = self.base_dir / f"{name}.joblib"
        payload = {"X": X, "y": y}
        joblib.dump(payload, path)

    def load_features(self, name: str):
        path = self.base_dir / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(path)
        return joblib.load(path)
