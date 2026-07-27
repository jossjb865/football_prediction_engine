import logging
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    metrics: Dict[str, float]

    def log(self) -> None:
        logger.info("===== Out-of-sample Evaluation =====")
        for k, v in self.metrics.items():
            logger.info("%-15s : %.4f", k, v)
        acc = self.metrics.get("accuracy", 0.0)
        if acc >= 0.70:
            logger.info("TARGET ACHIEVED: accuracy >= 70 %%")
        else:
            logger.info("Accuracy below 70 %% target – consider more data or feature enrichment.")

    def to_dict(self) -> Dict[str, float]:
        return dict(self.metrics)
