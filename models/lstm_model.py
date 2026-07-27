import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from features.sequence_builder import SequenceBuilder
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
        out, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


class LSTMMatchModel(BaseMatchModel):
    """
    Standard LSTM consuming leakage-free per-team historical sequences
    produced by SequenceBuilder.
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
        self.sequence_builder = SequenceBuilder(seq_len=seq_len)
        self.feature_cols: List[str] = []
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.input_dim: int = 0

    def fit(self, X: pd.DataFrame, y: pd.Series, matches: Optional[pd.DataFrame] = None, sample_weight: Optional[np.ndarray] = None) -> "LSTMMatchModel":
        if matches is None:
            raise ValueError("LSTMMatchModel.fit requires the original matches DataFrame for sequence construction")

        torch.manual_seed(self.random_state)
        self.feature_cols = list(X.columns)
        self.sequence_builder.fit(matches, self.feature_cols)

        X_seq = self.sequence_builder.transform(matches)
        self.input_dim = X_seq.shape[2]

        flat = X_seq.reshape(-1, self.input_dim)
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0) + 1e-8
        X_seq = (X_seq - self.mean_) / self.std_

        y_np = y.values.astype(np.int64)
        dataset = TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_np))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = _LSTMClassifier(self.input_dim, self.hidden_dim, self.num_layers, self.dropout).to(self.device)
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
                logger.info("LSTM epoch %d/%d – loss %.4f", epoch + 1, self.epochs, total_loss / len(dataset))
        return self

    def predict_proba(self, X: pd.DataFrame, matches: Optional[pd.DataFrame] = None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        if matches is None:
            raise ValueError("predict_proba requires matches DataFrame")

        X_seq = self.sequence_builder.transform(matches)
        X_seq = (X_seq - self.mean_) / self.std_
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X_seq).to(self.device))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
                "mean": self.mean_,
                "std": self.std_,
                "seq_len": self.seq_len,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "feature_cols": self.feature_cols,
            },
            path,
        )

    def load(self, path: str) -> "LSTMMatchModel":
        ckpt = torch.load(path, map_location=self.device)
        self.input_dim = ckpt["input_dim"]
        self.mean_ = ckpt["mean"]
        self.std_ = ckpt["std"]
        self.seq_len = ckpt["seq_len"]
        self.hidden_dim = ckpt["hidden_dim"]
        self.num_layers = ckpt["num_layers"]
        self.dropout = ckpt["dropout"]
        self.feature_cols = ckpt["feature_cols"]
        self.sequence_builder = SequenceBuilder(seq_len=self.seq_len)
        self.model = _LSTMClassifier(self.input_dim, self.hidden_dim, self.num_layers, self.dropout).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        return self
