import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from .base_model import BaseMatchModel

logger = logging.getLogger(__name__)


class DixonColesPoisson(BaseMatchModel):
    """
    Classic Dixon-Coles bivariate Poisson model with time-decay weighting
    and low-score correlation correction (rho parameter).
    """

    def __init__(self, xi: float = 0.0018, max_goals: int = 10):
        self.xi = xi  # time-decay half-life parameter
        self.max_goals = max_goals
        self.teams: list = []
        self.attack: Dict[str, float] = {}
        self.defence: Dict[str, float] = {}
        self.home_advantage: float = 0.0
        self.rho: float = 0.0
        self.team_index: Dict[str, int] = {}

    def _dc_probability(self, x: int, y: int, lambda_home: float, lambda_away: float, rho: float) -> float:
        """Dixon-Coles correction for low-scoring outcomes."""
        p = poisson.pmf(x, lambda_home) * poisson.pmf(y, lambda_away)
        if x == 0 and y == 0:
            return p * (1 - lambda_home * lambda_away * rho)
        if x == 0 and y == 1:
            return p * (1 + lambda_home * rho)
        if x == 1 and y == 0:
            return p * (1 + lambda_away * rho)
        if x == 1 and y == 1:
            return p * (1 - rho)
        return p

    def _log_likelihood(self, params: np.ndarray, matches: pd.DataFrame, n_teams: int) -> float:
        attack = params[:n_teams]
        defence = params[n_teams: 2 * n_teams]
        home_adv = params[2 * n_teams]
        rho = params[2 * n_teams + 1]

        ll = 0.0
        for _, row in matches.iterrows():
            i = self.team_index[row["home_id"]]
            j = self.team_index[row["away_id"]]
            lambda_h = np.exp(attack[i] + defence[j] + home_adv)
            lambda_a = np.exp(attack[j] + defence[i])
            # time weight
            days = row.get("days_ago", 0)
            weight = np.exp(-self.xi * days)
            p = self._dc_probability(int(row["home_score"]), int(row["away_score"]), lambda_h, lambda_a, rho)
            ll += weight * np.log(max(p, 1e-12))
        return -ll  # minimize negative log-likelihood

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None) -> "DixonColesPoisson":
        # X is expected to contain home_id, away_id, home_score, away_score, start_time
        # For compatibility with the pipeline we reconstruct from the original matches
        # In production the training pipeline passes a dedicated matches frame.
        # Here we assume X already contains the necessary columns or we receive them via kwargs.
        raise NotImplementedError(
            "DixonColesPoisson expects a dedicated matches DataFrame. "
            "Use fit_from_matches() instead of the generic fit()."
        )

    def fit_from_matches(self, matches: pd.DataFrame) -> "DixonColesPoisson":
        df = matches.copy()
        df["days_ago"] = (df["start_time"].max() - df["start_time"]).dt.total_seconds() / 86400.0
        self.teams = sorted(pd.unique(df[["home_id", "away_id"]].values.ravel("K")))
        self.team_index = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        # Initial parameters: attack=0, defence=0, home_adv=0.3, rho=-0.1
        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = 0.25
        x0[2 * n + 1] = -0.05

        bounds = [(-3, 3)] * (2 * n) + [(0.0, 1.0), (-0.3, 0.3)]

        logger.info("Optimising Dixon-Coles parameters for %d teams ...", n)
        res = minimize(
            fun=self._log_likelihood,
            x0=x0,
            args=(df, n),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 400, "disp": False},
        )
        if not res.success:
            logger.warning("Dixon-Coles optimisation did not fully converge: %s", res.message)

        params = res.x
        self.attack = {t: params[i] for t, i in self.team_index.items()}
        self.defence = {t: params[n + i] for t, i in self.team_index.items()}
        self.home_advantage = float(params[2 * n])
        self.rho = float(params[2 * n + 1])
        logger.info("Dixon-Coles fitted. Home advantage=%.3f, rho=%.3f", self.home_advantage, self.rho)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """X must contain home_id and away_id columns."""
        n = len(X)
        probs = np.zeros((n, 3))
        for idx, row in X.iterrows():
            home = row["home_id"]
            away = row["away_id"]
            if home not in self.attack or away not in self.attack:
                probs[idx] = [0.45, 0.27, 0.28]  # fallback prior
                continue
            lambda_h = np.exp(self.attack[home] + self.defence[away] + self.home_advantage)
            lambda_a = np.exp(self.attack[away] + self.defence[home])
            score_matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))
            for i in range(self.max_goals + 1):
                for j in range(self.max_goals + 1):
                    score_matrix[i, j] = self._dc_probability(i, j, lambda_h, lambda_a, self.rho)
            score_matrix /= score_matrix.sum()
            home_win = np.tril(score_matrix, -1).sum()
            draw = np.trace(score_matrix)
            away_win = np.triu(score_matrix, 1).sum()
            probs[idx] = [home_win, draw, away_win]
        return probs

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(
            {
                "teams": self.teams,
                "attack": self.attack,
                "defence": self.defence,
                "home_advantage": self.home_advantage,
                "rho": self.rho,
                "team_index": self.team_index,
                "xi": self.xi,
                "max_goals": self.max_goals,
            },
            path,
        )

    def load(self, path: str) -> "DixonColesPoisson":
        import joblib
        data = joblib.load(path)
        self.teams = data["teams"]
        self.attack = data["attack"]
        self.defence = data["defence"]
        self.home_advantage = data["home_advantage"]
        self.rho = data["rho"]
        self.team_index = data["team_index"]
        self.xi = data["xi"]
        self.max_goals = data["max_goals"]
        return self
