import numpy as np
import pandas as pd
from typing import List


class RollingMetricsCalculator:
    """Compute rolling attack/defence strengths, form, and momentum features without leakage."""

    def __init__(self, windows: List[int] = None):
        self.windows = windows or [3, 5, 10, 20]

    def compute(self, matches: pd.DataFrame) -> pd.DataFrame:
        """
        Expects a chronologically sorted DataFrame with columns:
        match_id, start_time, home_id, away_id, home_score, away_score, result_1x2
        Returns the same DataFrame with additional rolling columns.
        """
        df = matches.copy().sort_values("start_time").reset_index(drop=True)
        team_ids = pd.unique(df[["home_id", "away_id"]].values.ravel("K"))

        # Per-team long-format history
        home_df = df[["match_id", "start_time", "home_id", "home_score", "away_score", "result_1x2"]].rename(
            columns={
                "home_id": "team_id",
                "home_score": "goals_for",
                "away_score": "goals_against",
            }
        )
        home_df["is_home"] = 1
        home_df["points"] = home_df["result_1x2"].map({0: 3, 1: 1, 2: 0})

        away_df = df[["match_id", "start_time", "away_id", "away_score", "home_score", "result_1x2"]].rename(
            columns={
                "away_id": "team_id",
                "away_score": "goals_for",
                "home_score": "goals_against",
            }
        )
        away_df["is_home"] = 0
        away_df["points"] = away_df["result_1x2"].map({0: 0, 1: 1, 2: 3})

        long = pd.concat([home_df, away_df], ignore_index=True)
        long = long.sort_values(["team_id", "start_time"]).reset_index(drop=True)

        for w in self.windows:
            long[f"roll_gf_{w}"] = (
                long.groupby("team_id")["goals_for"]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )
            long[f"roll_ga_{w}"] = (
                long.groupby("team_id")["goals_against"]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )
            long[f"roll_pts_{w}"] = (
                long.groupby("team_id")["points"]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )
            long[f"roll_form_{w}"] = (
                long.groupby("team_id")["points"]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).sum())
            )

        # Pivot back to match level
        home_feats = long[long["is_home"] == 1].drop(columns=["is_home", "goals_for", "goals_against", "points", "result_1x2"])
        away_feats = long[long["is_home"] == 0].drop(columns=["is_home", "goals_for", "goals_against", "points", "result_1x2"])

        home_feats = home_feats.add_prefix("home_").rename(columns={"home_match_id": "match_id", "home_team_id": "home_id", "home_start_time": "start_time"})
        away_feats = away_feats.add_prefix("away_").rename(columns={"away_match_id": "match_id", "away_team_id": "away_id", "away_start_time": "start_time"})

        # Clean rename
        home_feats = home_feats.rename(columns={c: c.replace("home_match_id", "match_id") for c in home_feats.columns})
        home_feats.columns = [c if c != "home_match_id" else "match_id" for c in home_feats.columns]
        # Safer approach
        home_feats = long[long["is_home"] == 1].copy()
        home_feats = home_feats.drop(columns=["is_home", "goals_for", "goals_against", "points", "result_1x2", "team_id"])
        home_feats = home_feats.rename(columns={c: f"home_{c}" if c not in ("match_id", "start_time") else c for c in home_feats.columns})

        away_feats = long[long["is_home"] == 0].copy()
        away_feats = away_feats.drop(columns=["is_home", "goals_for", "goals_against", "points", "result_1x2", "team_id"])
        away_feats = away_feats.rename(columns={c: f"away_{c}" if c not in ("match_id", "start_time") else c for c in away_feats.columns})

        merged = df.merge(home_feats, on=["match_id", "start_time"], how="left")
        merged = merged.merge(away_feats, on=["match_id", "start_time"], how="left")

        # Goal difference features
        for w in self.windows:
            merged[f"home_gd_{w}"] = merged[f"home_roll_gf_{w}"] - merged[f"home_roll_ga_{w}"]
            merged[f"away_gd_{w}"] = merged[f"away_roll_gf_{w}"] - merged[f"away_roll_ga_{w}"]
            merged[f"rel_strength_{w}"] = merged[f"home_gd_{w}"] - merged[f"away_gd_{w}"]

        return merged
