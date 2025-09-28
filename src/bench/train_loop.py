import os, mlflow
from omegaconf import DictConfig
from .utils_io import set_seeds, ensure_dir, start_mlflow
from .registry import DATASETS, MODELS
from .calibrate import apply_calibration
from .utils_metrics import macro_f1, youden_threshold

def run_train(cfg: DictConfig):
    set_seeds(cfg.seed)
    ds = DATASETS[cfg.dataset.name](cfg.dataset)
    ds.prepare(cfg.dataset)
    Xtr, ytr = ds.load("train"); Xv, yv = ds.load("val")
    model = MODELS[cfg.model.name](input_dim=Xtr.shape[1],
                                   hidden_width=cfg.model.hidden_width,
                                   activation=getattr(cfg.model, "activation", "relu"),
                                   alpha=getattr(cfg.model, "alpha", 1e-3))
    with start_mlflow(cfg.tracking.mlflow_uri, cfg.tracking.run_name):
        model.fit(Xtr, ytr)
        model, _ = apply_calibration(cfg, model, Xv, yv)
        outdir = cfg.output_dir; ensure_dir(outdir)
        model.save(os.path.join(outdir, "model"))
        p = model.predict_proba(Xv); t = youden_threshold(yv, p)
        mf1 = macro_f1(yv, (p>=t).astype(int))
        mlflow.log_metric("val_macro_f1", mf1); mlflow.log_param("threshold", t)
        return outdir
