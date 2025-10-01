import os, mlflow
from omegaconf import DictConfig, OmegaConf
from .utils_io import set_seeds, ensure_dir, start_mlflow
from .registry import DATASETS, MODELS
from .calibrate import apply_calibration
from .utils_metrics import macro_f1, youden_threshold

def _make_model(cfg: DictConfig, input_dim: int):
    name = cfg.model.name
    params = OmegaConf.to_container(cfg.model, resolve=True)
    params = {k:v for k,v in params.items() if k != "name"}

    ModelCls = MODELS[name]

    if name == "elm":
        # ELM expects these params
        return ModelCls(
            input_dim=input_dim,
            hidden_width=int(params.get("hidden_width", 256)),
            activation=params.get("activation", "relu"),
            alpha=float(params.get("alpha", 1e-3)),
        )

    if name == "iforest":
        # IsolationForest is unsupervised; ignore labels in fit
        return ModelCls(
            input_dim=input_dim,
            n_estimators=int(params.get("n_estimators", 200)),
            contamination=params.get("contamination", "auto"),
            random_state=int(params.get("random_state", cfg.seed)),
        )

    # Generic fallback: pass everything except 'name'
    return ModelCls(input_dim=input_dim, **params)

def run_train(cfg: DictConfig):
    set_seeds(cfg.seed)
    ds = DATASETS[cfg.dataset.name](cfg.dataset)

    # Only prepare if processed files are missing (prevents accidental relabel)
    proc = cfg.dataset.processed_dir
    needed = [os.path.join(proc, f"{s}.npz") for s in ("train","val","test")]
    if not all(os.path.exists(p) for p in needed):
        ds.prepare(cfg.dataset)

    Xtr, ytr = ds.load("train")
    Xv,  yv  = ds.load("val")

    model = _make_model(cfg, input_dim=Xtr.shape[1])

    with start_mlflow(cfg.tracking.mlflow_uri, cfg.tracking.run_name):
        model.fit(Xtr, ytr)  # IForest ignores y
        # calibration stays off unless you enable it in config
        model, _ = apply_calibration(cfg, model, Xv, yv)

        outdir = cfg.output_dir
        ensure_dir(outdir)
        model.save(os.path.join(outdir, "model"))

        # quick val check (for logging/debug)
        pv = model.predict_proba(Xv)
        t = youden_threshold(yv, pv) if len(set(map(int, yv))) > 1 else 0.5
        vf1 = macro_f1(yv, (pv >= t).astype(int))
        mlflow.log_metric("val_macro_f1", vf1)
        mlflow.log_param("threshold_preview", t)

        return outdir

