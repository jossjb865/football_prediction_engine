import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .sportradar_client import SportradarClient

logger = logging.getLogger(__name__)


class DataProcessor:
    """Transforms raw Sportradar payloads into clean tabular datasets ready for feature engineering."""

    def __init__(self, client: Optional[SportradarClient] = None):
        self.client = client or SportradarClient()

    def fetch_historical_matches(
        self,
        competition_ids: List[str],
        min_season_year: int = 2018,
        max_matches: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Pull season schedules for the given competitions and flatten into a match-level DataFrame.
        Columns produced:
            match_id, season_id, competition_id, start_time, home_id, away_id,
            home_name, away_name, home_score, away_score, status, venue_id, ...
        """
        records: List[Dict[str, Any]] = []
        for comp_id in competition_ids:
            seasons = self.client.get_competition_seasons(comp_id)
            for season in seasons:
                year = int(str(season.get("start_date", "2000"))[:4])
                if year < min_season_year:
                    continue
                season_id = season["id"]
                logger.info("Fetching schedule for %s (%s)", season.get("name"), season_id)
                schedules = self.client.get_season_schedule(season_id)
                for item in schedules:
                    se = item.get("sport_event", {})
                    status = item.get("sport_event_status", {})
                    competitors = {c["qualifier"]: c for c in se.get("competitors", [])}
                    home = competitors.get("home", {})
                    away = competitors.get("away", {})
                    home_score = status.get("home_score")
                    away_score = status.get("away_score")
                    if home_score is None or away_score is None:
                        continue  # unfinished or cancelled
                    records.append(
                        {
                            "match_id": se.get("id"),
                            "season_id": season_id,
                            "competition_id": comp_id,
                            "competition_name": season.get("competition", {}).get("name"),
                            "season_name": season.get("name"),
                            "start_time": se.get("start_time"),
                            "home_id": home.get("id"),
                            "away_id": away.get("id"),
                            "home_name": home.get("name"),
                            "away_name": away.get("name"),
                            "home_score": int(home_score),
                            "away_score": int(away_score),
                            "status": status.get("status"),
                            "match_status": status.get("match_status"),
                            "venue_id": se.get("venue", {}).get("id"),
                            "venue_name": se.get("venue", {}).get("name"),
                            "referee_id": None,  # filled later if available
                        }
                    )
                    if max_matches and len(records) >= max_matches:
                        break
                if max_matches and len(records) >= max_matches:
                    break
            if max_matches and len(records) >= max_matches:
                break

        df = pd.DataFrame(records)
        if df.empty:
            logger.warning("No historical matches retrieved.")
            return df

        df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
        df = df.sort_values("start_time").reset_index(drop=True)
        df["result_1x2"] = df.apply(self._encode_1x2, axis=1)
        df["total_goals"] = df["home_score"] + df["away_score"]
        logger.info("Loaded %d historical matches", len(df))
        return df

    @staticmethod
    def _encode_1x2(row: pd.Series) -> int:
        if row["home_score"] > row["away_score"]:
            return 0  # home win
        if row["home_score"] < row["away_score"]:
            return 2  # away win
        return 1  # draw

    def enrich_with_match_statistics(self, matches_df: pd.DataFrame, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        For a subset of matches, pull detailed summary statistics (shots, possession, cards, etc.).
        This is expensive; use sample_size for experimentation.
        """
        if matches_df.empty:
            return matches_df

        target = matches_df if sample_size is None else matches_df.sample(n=min(sample_size, len(matches_df)), random_state=42)
        extra_rows = []
        for _, row in target.iterrows():
            try:
                summary = self.client.get_sport_event_summary(row["match_id"])
                stats = summary.get("statistics", {}).get("totals", {}).get("competitors", [])
                home_stats = next((s for s in stats if s.get("qualifier") == "home"), {})
                away_stats = next((s for s in stats if s.get("qualifier") == "away"), {})
                extra = {
                    "match_id": row["match_id"],
                    "home_shots_on_target": home_stats.get("statistics", {}).get("shots_on_target"),
                    "away_shots_on_target": away_stats.get("statistics", {}).get("shots_on_target"),
                    "home_possession": home_stats.get("statistics", {}).get("ball_possession"),
                    "away_possession": away_stats.get("statistics", {}).get("ball_possession"),
                    "home_yellow_cards": home_stats.get("statistics", {}).get("yellow_cards"),
                    "away_yellow_cards": away_stats.get("statistics", {}).get("yellow_cards"),
                    "home_corners": home_stats.get("statistics", {}).get("corner_kicks"),
                    "away_corners": away_stats.get("statistics", {}).get("corner_kicks"),
                }
                extra_rows.append(extra)
            except Exception as exc:
                logger.debug("Could not enrich %s: %s", row["match_id"], exc)
                continue

        if not extra_rows:
            return matches_df
        extra_df = pd.DataFrame(extra_rows)
        return matches_df.merge(extra_df, on="match_id", how="left")
