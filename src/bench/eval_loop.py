import os, json, numpy as np, time, mlflow
from omegaconf import DictConfig
from .utils_io import ensure_dir, start_mlflow
from .registry import DATASETS, MODELS
from .utils_metrics import macro_f1, auroc, precision, recall, youden_threshold

def run_eval(cfg: DictConfig):
    ds = DATASETS[cfg.dataset.name](cfg.dataset)
    Xte, yte = ds.load("test")
    model = MODELS[cfg.model.name].load(os.path.join(cfg.output_dir, "model"))
    with start_mlflow(cfg.tracking.mlflow_uri, cfg.tracking.run_name + "_eval"):
        p = model.predict_proba(Xte)
        t = youden_threshold(yte, p) if cfg.eval.select_threshold == "youden_j" else 0.5
        yhat = (p>=t).astype(int)
        m = {
            "macro_f1": float(macro_f1(yte, yhat)),
            "auroc":    float(auroc(yte, p)),
            "precision":float(precision(yte, yhat)),
            "recall":   float(recall(yte, yhat)),
            "threshold":float(t),
        }
        # simple latency profile
        Xs = Xte[:1]; ts=[]
        for _ in range(cfg.eval.profiling.n_warmup): _ = model.predict_proba(Xs)
        for _ in range(cfg.eval.profiling.n_iters):
            t0=time.perf_counter(); _=model.predict_proba(Xs); ts.append(time.perf_counter()-t0)
        m["latency_median_ms"]=float(np.median(ts)*1e3)
        m["latency_p95_ms"]=float(np.percentile(ts,95)*1e3)
        ensure_dir(cfg.output_dir)
        with open(os.path.join(cfg.output_dir,"report.json"),"w") as f: json.dump(m,f,indent=2)
        for k,v in m.items(): mlflow.log_metric(k, v if isinstance(v,(int,float)) else 0.0)
        return m
