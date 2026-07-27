import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseMatchModel

logger = logging.getLogger(__name__)


class _LSTMMomentum(nn.Module):
    """
    LSTM that receives an additional momentum channel (recent goal difference / form streak)
    concatenated at every time step.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


class LSTMMomentumModel(BaseMatchModel):
    """
    LSTM-Momentum variant.
    Explicitly injects rolling form / goal-difference streaks as extra input channels
    so the network can learn momentum dynamics.
    """

    def __init__(
        self,
        seq_len: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 30,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.random_state = random_state
        self.model: Optional[_LSTMMomentum] = None
        self.feature_dim: int = 0
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.momentum_cols = [
            "home_roll_form_5", "away_roll_form_5",
            "home_gd_5", "away_gd_5",
            "home_roll_pts_5", "away_roll_pts_5",
        ]

    def _build_sequences(self, matches: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        base_cols = [c for c in feature_cols if c not in self.momentum_cols]
        # Ensure momentum columns exist
        for c in self.momentum_cols:
            if c not in matches.columns:
                matches[c] = 0.0
        X_base = matches[base_cols].values.astype(np.float32)
        X_mom = matches[self.momentum_cols].values.astype(np.float32)
        # Concatenate momentum channels
        X = np.concatenate([X_base, X_mom], axis=1)
        X_seq = np.repeat(X[:, np.newaxis, :], self.seq_len, axis=1)
        # Add a simple linear momentum ramp across the sequence dimension
        ramp = np.linspace(0.5, 1.5, self.seq_len, dtype=np.float32).reshape(1, self.seq_len, 1)
        X_seq = X_seq * ramp
        y = matches["result_1x2"].values.astype(np.int64)
        return X_seq, y

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None) -> "LSTMMomentumModel":
        torch.manual_seed(self.random_state)
        feature_cols = [c for c in X.columns if c != "result_1x2"]
        matches = X.copy()
        matches["result_1x2"] = y.values

        X_seq, y_np = self._build_sequences(matches, feature_cols)
        self.feature_dim = X_seq.shape[2]

        flat = X_seq.reshape(-1, self.feature_dim)
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0) + 1e-8
        X_seq = (X_seq - self.mean_) / self.std_

        dataset = TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_np))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = _LSTMMomentum(self.feature_dim, self.hidden_dim, self.num_layers, self.dropout).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)
            if (epoch + 1) % 10 == 0:
                logger.info("LSTM-Momentum epoch %d/%d - loss %.4f", epoch + 1, self.epochs, total_loss / len(dataset))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        feature_cols = [c for c in X.columns if c != "result_1x2"]
        matches = X.copy()
        if "result_1x2" not in matches.columns:
            matches["result_1x2"] = 0
        X_seq, _ = self._build_sequences(matches, feature_cols)
        X_seq = (X_seq - self.mean_) / self.std_
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X_seq).to(self.device))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "feature_dim": self.feature_dim,
                "mean": self.mean_,
                "std": self.std_,
                "seq_len": self.seq_len,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
            path,
        )

    def load(self, path: str) -> "LSTMMomentumModel":
        ckpt = torch.load(path, map_location=self.device)
        self.feature_dim = ckpt["feature_dim"]
        self.mean_ = ckpt["mean"]
        self.std_ = ckpt["std"]
        self.seq_len = ckpt["seq_len"]
        self.hidden_dim = ckpt["hidden_dim"]
        self.num_layers = ckpt["num_layers"]
        self.dropout = ckpt["dropout"]
        self.model = _LSTMMomentum(self.feature_dim, self.hidden_dim, self.num_layers, self.dropout).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        return self
