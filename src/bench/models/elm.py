import os, numpy as np
from typing import Optional
from .base import ModelAPI
from ..registry import register_model
def relu(x): return np.maximum(0, x)

@register_model("elm")
class ELM(ModelAPI):
    name = "elm"
    def __init__(self, input_dim: int=None, hidden_width: int=256, activation="relu", alpha=1e-3):
        self.input_dim = input_dim; self.hidden = hidden_width
        self.act = relu if activation=="relu" else np.tanh
        self.alpha = alpha
        self.W = None; self.b = None; self.beta = None
    def _hidden_map(self, X):
        H = self.act(X @ self.W + self.b)
        return np.hstack([H, np.ones((H.shape[0],1), dtype=H.dtype)])
    def fit(self, X: np.ndarray, y: Optional[np.ndarray], **kwargs):
        n, d = X.shape; self.input_dim = d
        rng = np.random.default_rng(42)
        self.W = rng.normal(0, 1/np.sqrt(d), size=(d, self.hidden)).astype("float32")
        self.b = rng.normal(0, 1, size=(1, self.hidden)).astype("float32")
        H = self._hidden_map(X)
        lam = self.alpha * np.eye(H.shape[1], dtype=H.dtype)
        self.beta = np.linalg.pinv(H.T @ H + lam) @ (H.T @ y.astype("float32"))
        return self
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self._hidden_map(X) @ self.beta
        return (1/(1+np.exp(-z))).reshape(-1)
    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype("int64")
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        np.savez(os.path.join(path, "elm.npz"), W=self.W, b=self.b, beta=self.beta,
                 input_dim=self.input_dim, hidden=self.hidden, alpha=self.alpha)
    @classmethod
    def load(cls, path: str):
        d = np.load(os.path.join(path, "elm.npz"), allow_pickle=True)
        m = cls(int(d["input_dim"]), int(d["hidden"]), "relu", float(d["alpha"]))
        m.W = d["W"]; m.b = d["b"]; m.beta = d["beta"]; return m
