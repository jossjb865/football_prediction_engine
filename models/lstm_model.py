import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseMatchModel

logger = logging.getLogger(__name__)


class _LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        out, (h_n, _) = self.lstm(x)
        last = h_n[-1]  # (batch, hidden)
        return self.fc(last)


class LSTMMatchModel(BaseMatchModel):
    """
    Standard LSTM that consumes a fixed-length sequence of past match features
    for each team (home and away) concatenated.
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
        self.model: Optional[_LSTMClassifier] = None
        self.feature_dim: int = 0
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def _build_sequences(self, matches: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build (n_matches, seq_len, n_features) tensors.
        For simplicity we use the rolling features already present in the frame
        and repeat / pad to seq_len. In a full production system one would
        maintain per-team chronological feature histories.
        """
        X = matches[feature_cols].values.astype(np.float32)
        # Simple temporal expansion: use the same feature vector repeated seq_len times
        # (placeholder for a richer history builder that is still leakage-free)
        X_seq = np.repeat(X[:, np.newaxis, :], self.seq_len, axis=1)
        y = matches["result_1x2"].values.astype(np.int64)
        return X_seq, y

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None) -> "LSTMMatchModel":
        # X is expected to already contain the engineered features + result_1x2
        # We reconstruct a minimal matches frame
        torch.manual_seed(self.random_state)
        feature_cols = [c for c in X.columns if c != "result_1x2"]
        matches = X.copy()
        matches["result_1x2"] = y.values

        X_seq, y_np = self._build_sequences(matches, feature_cols)
        self.feature_dim = X_seq.shape[2]

        # Standardize across the feature dimension
        flat = X_seq.reshape(-1, self.feature_dim)
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0) + 1e-8
        X_seq = (X_seq - self.mean_) / self.std_

        dataset = TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_np))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = _LSTMClassifier(self.feature_dim, self.hidden_dim, self.num_layers, self.dropout).to(self.device)
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
                logger.info("LSTM epoch %d/%d - loss %.4f", epoch + 1, self.epochs, total_loss / len(dataset))
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

    def load(self, path: str) -> "LSTMMatchModel":
        ckpt = torch.load(path, map_location=self.device)
        self.feature_dim = ckpt["feature_dim"]
        self.mean_ = ckpt["mean"]
        self.std_ = ckpt["std"]
        self.seq_len = ckpt["seq_len"]
        self.hidden_dim = ckpt["hidden_dim"]
        self.num_layers = ckpt["num_layers"]
        self.dropout = ckpt["dropout"]
        self.model = _LSTMClassifier(self.feature_dim, self.hidden_dim, self.num_layers, self.dropout).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        return self
