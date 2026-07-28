import logging
from typing import Dict, List, Optional

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
    - Features are computed on-the-fly from raw match results (no dependence on X).
    """

    def __init__(self, seq_len: int = 10):
        self.seq_len = seq_len
        self.team_histories: Dict[str, pd.DataFrame] = {}
        # Features that will be available in every sequence step
        self.seq_feature_names = [
            "gf", "ga", "gd", "pts", "is_home", "result"  # result: 1=win, 0=draw, -1=loss
        ]

    def fit(self, matches: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> "SequenceBuilder":
        """
        Build per-team chronological histories from the raw matches DataFrame.
        feature_cols is ignored (kept only for API compatibility).
        """
        df = matches.sort_values("start_time").reset_index(drop=True).copy()

        # Ensure required columns exist
        required = {"match_id", "start_time", "home_id", "away_id", "home_score", "away_score"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"SequenceBuilder requires columns: {missing}")

        records = []

        for _, row in df.iterrows():
            home_id = str(row["home_id"])
            away_id = str(row["away_id"])
            hg = float(row["home_score"]) if pd.notna(row["home_score"]) else 0.0
            ag = float(row["away_score"]) if pd.notna(row["away_score"]) else 0.0

            # Home perspective
            home_pts = 3.0 if hg > ag else (1.0 if hg == ag else 0.0)
            home_result = 1.0 if hg > ag else (0.0 if hg == ag else -1.0)
            records.append({
                "match_id": row["match_id"],
                "start_time": row["start_time"],
                "team_id": home_id,
                "gf": hg,
                "ga": ag,
                "gd": hg - ag,
                "pts": home_pts,
                "is_home": 1.0,
                "result": home_result,
            })

            # Away perspective
            away_pts = 3.0 if ag > hg else (1.0 if ag == hg else 0.0)
            away_result = 1.0 if ag > hg else (0.0 if ag == hg else -1.0)
            records.append({
                "match_id": row["match_id"],
                "start_time": row["start_time"],
                "team_id": away_id,
                "gf": ag,
                "ga": hg,
                "gd": ag - hg,
                "pts": away_pts,
                "is_home": 0.0,
                "result": away_result,
            })

        long = pd.DataFrame(records)
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
        where the last dimension concatenates [home history | away history].
        Missing history is zero-padded on the left (oldest positions).
        """
        n = len(matches)
        n_feat = len(self.seq_feature_names)
        sequences = np.zeros((n, self.seq_len, n_feat * 2), dtype=np.float32)

        for i, (_, row) in enumerate(matches.iterrows()):
            home_id = str(row["home_id"])
            away_id = str(row["away_id"])
            match_time = row["start_time"]

            home_seq = self._get_team_sequence(home_id, match_time)
            away_seq = self._get_team_sequence(away_id, match_time)

            sequences[i] = np.concatenate([home_seq, away_seq], axis=1)

        return sequences

    def _get_team_sequence(self, team_id: str, before_time) -> np.ndarray:
        hist = self.team_histories.get(team_id)
        n_feat = len(self.seq_feature_names)

        if hist is None or hist.empty:
            return np.zeros((self.seq_len, n_feat), dtype=np.float32)

        # Strictly causal: only matches before the current one
        past = hist[hist["start_time"] < before_time].tail(self.seq_len)

        if past.empty:
            return np.zeros((self.seq_len, n_feat), dtype=np.float32)

        arr = past[self.seq_feature_names].values.astype(np.float32)

        if len(arr) < self.seq_len:
            pad = np.zeros((self.seq_len - len(arr), n_feat), dtype=np.float32)
            arr = np.vstack([pad, arr])

        return arr
