import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ROISimulator:
    """
    Simulates betting ROI under three staking strategies:
    1. Flat stake
    2. Fractional Kelly
    3. Confidence-threshold filter
    Assumes decimal odds are available (or a synthetic edge is supplied).
    """

    def __init__(self, bankroll: float = 10000.0, flat_stake: float = 50.0, kelly_fraction: float = 0.25):
        self.bankroll = bankroll
        self.flat_stake = flat_stake
        self.kelly_fraction = kelly_fraction

    def simulate(
        self,
        y_true: np.ndarray,
        proba: np.ndarray,
        odds: Optional[np.ndarray] = None,
        min_edge: float = 0.05,
        min_prob: float = 0.45,
    ) -> Dict[str, float]:
        """
        y_true : (n,) integer labels 0/1/2
        proba  : (n, 3) model probabilities
        odds   : (n, 3) decimal odds. If None, synthetic odds = 1 / (true_prob + noise) are generated.
        """
        n = len(y_true)
        if odds is None:
            # Synthetic market that is slightly inefficient
            true_p = np.eye(3)[y_true]
            noise = np.random.normal(0, 0.03, size=true_p.shape)
            market_p = np.clip(true_p + noise, 0.05, 0.90)
            market_p /= market_p.sum(axis=1, keepdims=True)
            odds = 1.0 / market_p

        # Edge = model_prob * odds - 1
        edge = proba * odds - 1.0
        best_outcome = np.argmax(edge, axis=1)
        best_edge = edge[np.arange(n), best_outcome]
        best_prob = proba[np.arange(n), best_outcome]
        best_odds = odds[np.arange(n), best_outcome]

        # Filter bets
        mask = (best_edge >= min_edge) & (best_prob >= min_prob)
        if mask.sum() == 0:
            logger.warning("No bets passed the edge/probability filter")
            return {"n_bets": 0, "roi_flat": 0.0, "roi_kelly": 0.0, "final_bankroll_flat": self.bankroll}

        y_sel = y_true[mask]
        out_sel = best_outcome[mask]
        odds_sel = best_odds[mask]
        prob_sel = best_prob[mask]
        edge_sel = best_edge[mask]

        # Flat staking
        stakes_flat = np.full(mask.sum(), self.flat_stake)
        wins = (out_sel == y_sel).astype(float)
        pnl_flat = stakes_flat * (wins * (odds_sel - 1) - (1 - wins))
        roi_flat = pnl_flat.sum() / stakes_flat.sum()

        # Fractional Kelly
        kelly_f = self.kelly_fraction * (prob_sel * odds_sel - 1) / (odds_sel - 1)
        kelly_f = np.clip(kelly_f, 0.0, 0.1)  # hard cap 10 % of bankroll
        stakes_kelly = kelly_f * self.bankroll
        pnl_kelly = stakes_kelly * (wins * (odds_sel - 1) - (1 - wins))
        roi_kelly = pnl_kelly.sum() / (stakes_kelly.sum() + 1e-9)

        return {
            "n_bets": int(mask.sum()),
            "hit_rate": float(wins.mean()),
            "avg_odds": float(odds_sel.mean()),
            "avg_edge": float(edge_sel.mean()),
            "roi_flat": float(roi_flat),
            "roi_kelly": float(roi_kelly),
            "pnl_flat": float(pnl_flat.sum()),
            "pnl_kelly": float(pnl_kelly.sum()),
            "final_bankroll_flat": float(self.bankroll + pnl_flat.sum()),
            "final_bankroll_kelly": float(self.bankroll + pnl_kelly.sum()),
        }
