#!/usr/bin/env python3
"""
CLI for live / batch inference.
Example:
    python predict.py --date 2026-07-28 --competitions sr:competition:17
"""

import argparse
import logging
from datetime import datetime

import pandas as pd

from config.logging_config import setup_logging
from config.settings import settings
from data_loader.sportradar_client import SportradarClient
from data_loader.data_processor import DataProcessor
from pipelines.inference_pipeline import InferencePipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Football Prediction Engine – Inference")
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (default = today)")
    parser.add_argument("--competitions", nargs="+", default=["sr:competition:17"])
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("predict")

    settings.validate()
    date_str = args.date or datetime.utcnow().strftime("%Y-%m-%d")

    client = SportradarClient()
    processor = DataProcessor(client)

    # Pull today’s schedule
    schedules = client.get_daily_schedules(date_str)
    records = []
    for item in schedules:
        se = item.get("sport_event", {})
        competitors = {c["qualifier"]: c for c in se.get("competitors", [])}
        home = competitors.get("home", {})
        away = competitors.get("away", {})
        records.append(
            {
                "match_id": se.get("id"),
                "start_time": se.get("start_time"),
                "home_id": home.get("id"),
                "away_id": away.get("id"),
                "home_name": home.get("name"),
                "away_name": away.get("name"),
                "competition_id": se.get("sport_event_context", {}).get("competition", {}).get("id"),
            }
        )

    if not records:
        logger.warning("No matches found for %s", date_str)
        return

    matches = pd.DataFrame(records)
    matches["start_time"] = pd.to_datetime(matches["start_time"], utc=True)

    # Filter by requested competitions if possible
    if "competition_id" in matches.columns:
        matches = matches[matches["competition_id"].isin(args.competitions)]

    if matches.empty:
        logger.warning("No matches left after competition filter")
        return

    pipeline = InferencePipeline(models_dir=args.models_dir)
    preds = pipeline.predict_matches(matches)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print("\n=== Predictions ===")
    print(preds.to_string(index=False))
    preds.to_csv(f"predictions_{date_str}.csv", index=False)
    logger.info("Saved predictions_%s.csv", date_str)


if __name__ == "__main__":
    main()
