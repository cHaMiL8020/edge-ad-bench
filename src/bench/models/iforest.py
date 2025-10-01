import numpy as np
from sklearn.ensemble import IsolationForest
from .base import ModelAPI
from ..registry import register_model

@register_model("iforest")
class IForest(ModelAPI):
    name = "iforest"
    def __init__(self, input_dim: int=None, n_estimators: int=200, contamination: float="auto", random_state: int=42):
        self.input_dim = input_dim
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X: np.ndarray, y=None, **kwargs):
        # IsolationForest is unsupervised; ignore y
        self.model.fit(X)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # decision_function: higher = normal. Convert to anomaly score in [0,1]
        s = self.model.decision_function(X)  # typically ~[-0.5, +0.5], higher normal
        # map to [0,1] where 1 = anomaly (positive class)
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-9)
        anomaly = 1.0 - s_norm
        return anomaly.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype("int64")

    def save(self, path: str) -> None:
        import os, joblib
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model, os.path.join(path, "iforest.joblib"))

    @classmethod
    def load(cls, path: str):
        import os, joblib
        m = cls()
        m.model = joblib.load(os.path.join(path, "iforest.joblib"))
        return m
