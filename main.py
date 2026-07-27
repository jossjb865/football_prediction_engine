#!/usr/bin/env python3
"""
Entry point for the Football Prediction Engine.
Usage:
    export SPORTRADAR_API_KEY=your_key
    python main.py --competitions sr:competition:17 sr:competition:8
"""

import argparse
import logging
import sys

from config.logging_config import setup_logging
from config.settings import settings
from pipelines.training_pipeline import TrainingPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Football Prediction Engine – Super Ensemble Trainer")
    parser.add_argument(
        "--competitions",
        nargs="+",
        default=["sr:competition:17"],  # Premier League example
        help="Sportradar competition IDs (e.g. sr:competition:17 for EPL)",
    )
    parser.add_argument("--min-year", type=int, default=2018, help="Earliest season year to include")
    parser.add_argument("--max-matches", type=int, default=None, help="Limit number of matches (debug)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("main")

    try:
        settings.validate()
    except EnvironmentError as exc:
        logger.error(str(exc))
        sys.exit(1)

    pipeline = TrainingPipeline(
        competition_ids=args.competitions,
        min_season_year=args.min_year,
    )
    report = pipeline.run(max_matches=args.max_matches)
    logger.info("Pipeline finished. Metrics: %s", report.to_dict())


if __name__ == "__main__":
    main()
