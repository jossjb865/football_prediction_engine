from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class BaseMatchModel(ABC):
    """Common interface for all 1X2 probabilistic models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None) -> "BaseMatchModel":
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return (n_samples, 3) array of probabilities for [home, draw, away]."""
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> "BaseMatchModel":
        pass
