import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SequenceBuilder:
    """
    Constructs fixed-length, strictly causal sequences of team-level features
    for LSTM / LSTM-Momentum models.

    Guarantees zero future leakage:
    - Each sequence for match t only contains information available before match t.
    - Sequences are built from the chronological history of each team independently.
    """

    def __init__(self, seq_len: int = 10, feature_cols: Optional[List[str]] = None):
        self.seq_len = seq_len
        self.feature_cols = feature_cols or []
        self.team_histories: Dict[str, pd.DataFrame] = {}

    def fit(self, matches: pd.DataFrame, feature_cols: List[str]) -> "SequenceBuilder":
        """
        Build per-team chronological histories from the full (already sorted) match frame.
        Call this once on the training set before any transform.
        """
        self.feature_cols = feature_cols
        df = matches.sort_values("start_time").reset_index(drop=True).copy()

        # Long format: one row per team-participation
        home = df[["match_id", "start_time", "home_id"] + [c for c in feature_cols if c.startswith("home_")]].copy()
        home = home.rename(columns={"home_id": "team_id"})
        home.columns = [c.replace("home_", "") if c.startswith("home_") else c for c in home.columns]

        away = df[["match_id", "start_time", "away_id"] + [c for c in feature_cols if c.startswith("away_")]].copy()
        away = away.rename(columns={"away_id": "team_id"})
        away.columns = [c.replace("away_", "") if c.startswith("away_") else c for c in away.columns]

        long = pd.concat([home, away], ignore_index=True)
        long = long.sort_values(["team_id", "start_time"]).reset_index(drop=True)

        self.team_histories = {
            tid: grp.reset_index(drop=True)
            for tid, grp in long.groupby("team_id")
        }
        logger.info("SequenceBuilder fitted on %d teams", len(self.team_histories))
        return self

    def transform(self, matches: pd.DataFrame) -> np.ndarray:
        """
        Returns array of shape (n_matches, seq_len, n_features * 2)
        where the last dimension concatenates home history + away history.
        Missing history is zero-padded on the left (oldest positions).
        """
        n = len(matches)
        n_feat = len(self.feature_cols) // 2  # because we stripped home_/away_ prefixes
        # We will use the generic feature set that appears in both home and away
        generic_cols = [c for c in self.feature_cols if c.startswith("home_")]
        generic_cols = [c.replace("home_", "") for c in generic_cols]
        n_feat = len(generic_cols)

        sequences = np.zeros((n, self.seq_len, n_feat * 2), dtype=np.float32)

        for i, row in matches.iterrows():
            home_id = row["home_id"]
            away_id = row["away_id"]
            match_time = row["start_time"]

            home_seq = self._get_team_sequence(home_id, match_time, generic_cols)
            away_seq = self._get_team_sequence(away_id, match_time, generic_cols)

            sequences[i] = np.concatenate([home_seq, away_seq], axis=1)

        return sequences

    def _get_team_sequence(self, team_id: str, before_time: pd.Timestamp, cols: List[str]) -> np.ndarray:
        hist = self.team_histories.get(team_id)
        if hist is None or hist.empty:
            return np.zeros((self.seq_len, len(cols)), dtype=np.float32)

        # Strictly before the current match
        past = hist[hist["start_time"] < before_time].tail(self.seq_len)
        if past.empty:
            return np.zeros((self.seq_len, len(cols)), dtype=np.float32)

        arr = past[cols].values.astype(np.float32)
        if len(arr) < self.seq_len:
            pad = np.zeros((self.seq_len - len(arr), len(cols)), dtype=np.float32)
            arr = np.vstack([pad, arr])
        return arr
