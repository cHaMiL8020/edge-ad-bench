import os, json, numpy as np, time, mlflow
from omegaconf import DictConfig
from .utils_io import ensure_dir, start_mlflow
from .registry import DATASETS, MODELS
from .utils_metrics import macro_f1, precision, recall, youden_threshold

def _safe_auroc(y_true, y_score):
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(map(int, y_true))) < 2:
            return 0.5  # undefined ROC if one class only
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")

def _fixed_fpr_threshold(y_true, y_score, target_fpr=0.05):
    """Pick threshold as the (1 - target_fpr) quantile of scores on negatives."""
    neg = y_score[(np.asarray(y_true)==0)]
    if len(neg) == 0:
        return 0.5  # fallback if no negatives (shouldn't happen for val in our plan)
    q = 1.0 - float(target_fpr)
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(neg, q))

def _pick_threshold(select_cfg: str, yv, pv):
    if select_cfg == "youden_j":
        return float(youden_threshold(yv, pv))
    if select_cfg.startswith("fixed_fpr:"):
        try:
            fpr = float(select_cfg.split(":",1)[1])
        except Exception:
            fpr = 0.05
        return _fixed_fpr_threshold(yv, pv, fpr)
    # default
    return 0.5

def run_eval(cfg: DictConfig):
    ds = DATASETS[cfg.dataset.name](cfg.dataset)
    Xv, yv = ds.load("val")
    Xt, yt = ds.load("test")

    model = MODELS[cfg.model.name].load(os.path.join(cfg.output_dir, "model"))
    with start_mlflow(cfg.tracking.mlflow_uri, cfg.tracking.run_name + "_eval"):
        pv = model.predict_proba(Xv)
        pt = model.predict_proba(Xt)

        t = _pick_threshold(cfg.eval.select_threshold, yv, pv)
        yhat = (pt >= t).astype(int)

        m = {
            "threshold":           float(t),
            "macro_f1":            float(macro_f1(yt, yhat)),
            "auroc":               _safe_auroc(yt, pt),
            "precision":           float(precision(yt, yhat)),
            "recall":              float(recall(yt, yhat)),
        }

        # latency on single sample
        Xs = Xt[:1]; ts=[]
        for _ in range(cfg.eval.profiling.n_warmup): _ = model.predict_proba(Xs)
        for _ in range(cfg.eval.profiling.n_iters):
            t0=time.perf_counter(); _=model.predict_proba(Xs); ts.append(time.perf_counter()-t0)
        m["latency_median_ms"]=float(np.median(ts)*1e3)
        m["latency_p95_ms"]=float(np.percentile(ts,95)*1e3)

        ensure_dir(cfg.output_dir)
        with open(os.path.join(cfg.output_dir,"report.json"),"w") as f: json.dump(m,f,indent=2)
        for k,v in m.items():
            if isinstance(v, (int, float)): mlflow.log_metric(k, v)
        return m

