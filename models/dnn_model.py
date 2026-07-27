import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseMatchModel

logger = logging.getLogger(__name__)


class _DNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (256, 128, 64), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DNNMatchModel(BaseMatchModel):
    """Deep neural network for 1X2 classification implemented in pure PyTorch."""

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3,
        lr: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 40,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.random_state = random_state
        self.model: Optional[_DNN] = None
        self.feature_names: list = []
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def _standardize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0) + 1e-8
        return (X - self.mean_) / self.std_

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None) -> "DNNMatchModel":
        torch.manual_seed(self.random_state)
        self.feature_names = list(X.columns)
        X_np = X.values.astype(np.float32)
        y_np = y.values.astype(np.int64)
        X_np = self._standardize(X_np, fit=True)

        dataset = TensorDataset(torch.from_numpy(X_np), torch.from_numpy(y_np))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.model = _DNN(X_np.shape[1], self.hidden_dims, self.dropout).to(self.device)
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
                logger.info("DNN epoch %d/%d - loss %.4f", epoch + 1, self.epochs, total_loss / len(dataset))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        self.model.eval()
        X_np = self._standardize(X[self.feature_names].values.astype(np.float32), fit=False)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X_np).to(self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "feature_names": self.feature_names,
                "mean": self.mean_,
                "std": self.std_,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
            },
            path,
        )

    def load(self, path: str) -> "DNNMatchModel":
        ckpt = torch.load(path, map_location=self.device)
        self.feature_names = ckpt["feature_names"]
        self.mean_ = ckpt["mean"]
        self.std_ = ckpt["std"]
        self.hidden_dims = ckpt["hidden_dims"]
        self.dropout = ckpt["dropout"]
        self.model = _DNN(len(self.feature_names), self.hidden_dims, self.dropout).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        return self
