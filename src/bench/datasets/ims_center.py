import os, glob
import numpy as np
from typing import Tuple, List
from .base import DatasetAdapter
from ..registry import register_dataset

def _read_ascii_series(path: str) -> np.ndarray:
    x = np.loadtxt(path, dtype=np.float32)
    return np.ravel(x).astype(np.float32)

def _zscore(x: np.ndarray, eps: float=1e-8) -> np.ndarray:
    m, s = x.mean(), x.std()
    return (x - m) / (s + eps)

def _window_1d(x: np.ndarray, w: int, s: int) -> np.ndarray:
    if len(x) < w: return np.empty((0, w), dtype=np.float32)
    starts = np.arange(0, len(x) - w + 1, s, dtype=int)
    return np.stack([x[i:i+w] for i in starts], axis=0).astype(np.float32)

def _channel_map(set_no: int, bearing_id: int) -> List[int]:
    if set_no == 1:
        base = (bearing_id - 1) * 2 + 1
        return [base, base + 1]         # 8 ch total
    elif set_no in (2, 3):
        return [bearing_id]             # 4 ch total
    else:
        raise ValueError("set_no must be 1, 2, or 3")

@register_dataset("ims_center")
class IMSCenterDataset(DatasetAdapter):
    name = "ims_center"
    modality = "timeseries"
    task = "binary"

    def __init__(self, cfg):
        self.cfg = cfg
        self.raw_root = cfg.raw_root
        self.proc = cfg.processed_dir
        os.makedirs(self.proc, exist_ok=True)

    def _list_files(self):
        set_dir = os.path.join(self.raw_root, f"set{self.cfg.set_no}")
        files = sorted(glob.glob(os.path.join(set_dir, "*")))
        files = [f for f in files if os.path.isfile(f)]
        if not files:
            raise FileNotFoundError(f"No ASCII files found in {set_dir}")
        return files

    def _label_indices(self, n_files: int) -> np.ndarray:
        mode = self.cfg.labeling.mode
        y = np.zeros((n_files,), dtype=np.int64)
        if mode == "cut_index":
            cut = int(self.cfg.labeling.cut_index)
            cut = max(0, min(cut, n_files))
            y[cut:] = 1
        elif mode == "last_k":
            k = int(self.cfg.labeling.last_k)
            k = max(0, min(k, n_files))
            if k > 0: y[-k:] = 1
        else:
            raise ValueError("labeling.mode must be 'cut_index' or 'last_k'")
        return y

    def prepare(self, cfg):
        files = self._list_files()
        y_files = self._label_indices(len(files))

        Xs, ys = [], []
        for i, f in enumerate(files):
            x = _read_ascii_series(f)
            if cfg.preprocess.zscore: x = _zscore(x)
            if cfg.preprocess.clip_value:
                v = float(cfg.preprocess.clip_value); x = np.clip(x, -v, v)

            if cfg.window.enabled:
                Xw = _window_1d(x, int(cfg.window.length), int(cfg.window.stride))
                yrep = np.full((len(Xw),), y_files[i], dtype=np.int64)
                if len(Xw):
                    Xs.append(Xw); ys.append(yrep)
            else:
                Xs.append(x.reshape(1, -1))
                ys.append(np.array([y_files[i]], dtype=np.int64))

        if not Xs:
            raise RuntimeError("No samples produced. Check window length/stride/raw files.")
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)

        # time-preserving split (we'll keep the end as test)
        n = len(y)
        n_tr = int(cfg.splits.train * n)
        n_va = int(cfg.splits.val   * n)
        n_te = n - n_tr - n_va

        Xtr, ytr = X[:n_tr], y[:n_tr]
        Xva, yva = X[n_tr:n_tr+n_va], y[n_tr:n_tr+n_va]
        Xte, yte = X[-n_te:], y[-n_te:]

        np.savez(os.path.join(self.proc, "train.npz"), X=Xtr, y=ytr)
        np.savez(os.path.join(self.proc, "val.npz"),   X=Xva, y=yva)
        np.savez(os.path.join(self.proc, "test.npz"),  X=Xte, y=yte)
        
        for split in ("train","val","test"):
            y = np.load(os.path.join(self.proc, f"{split}.npz"))["y"]
            print(f"{split:5s}  n={len(y):4d}  pos={(y==1).sum():4d}  neg={(y==0).sum():4d}")


    def load(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        path = os.path.join(self.proc, f"{split}.npz")
        with np.load(path) as d:
            return d["X"], d["y"]
