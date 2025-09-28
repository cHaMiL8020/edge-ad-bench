import os, numpy as np
from .base import DatasetAdapter
from ..registry import register_dataset

@register_dataset("ims")
class IMSDataset(DatasetAdapter):
    name = "ims"; modality = "timeseries"; task = "binary"

    def __init__(self, cfg):
        self.cfg = cfg
        self.proc = cfg.processed_dir
        os.makedirs(self.proc, exist_ok=True)

    def prepare(self, cfg):
        n_train, n_val, n_test = 800, 200, 200
        w = cfg.window.length
        def make(n):
            X0 = np.random.normal(0,   1, size=(n//2, w)).astype("float32")
            X1 = np.random.normal(0.6, 1, size=(n//2, w)).astype("float32")
            X  = np.vstack([X0, X1])
            y  = np.array([0]*(n//2) + [1]*(n//2), dtype="int64")
            idx = np.random.permutation(len(y))
            return X[idx], y[idx]
        for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
            X, y = make(n)
            np.savez(os.path.join(self.proc, f"{split}.npz"), X=X, y=y)

    def load(self, split):
        d = np.load(os.path.join(self.proc, f"{split}.npz"))
        return d["X"], d["y"]
