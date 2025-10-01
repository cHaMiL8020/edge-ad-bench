# Edge-AD-Bench: Current Implementation Status (Sep 2025)

## 1. Overview

This repository provides a **config-driven, dataset-agnostic benchmarking framework** for lightweight anomaly detection models on edge devices.  

- **Framework**: Ubuntu 20.04, Python 3.8, Hydra configs, MLflow tracking.  
- **Datasets supported (planned)**:  
  - ✅ IMS Bearing Dataset (Set 2, ASCII, rolling element failures)  
  - ⏳ NASA C-MAPSS (engine prognostics)  
  - ⏳ UCI HAR (human activity recognition)  
  - ⏳ MIT-BIH Arrhythmia (ECG anomalies)  
  - ⏳ UCSD Ped2 (video anomalies)  

- **Models supported (planned)**:  
  - ✅ ELM (Extreme Learning Machine, supervised baseline)  
  - ⏳ Isolation Forest (unsupervised baseline)  
  - ⏳ One-Class SVM (unsupervised baseline)  
  - ⏳ Tiny CNN (lightweight supervised deep model)  
  - ⏳ dCeNN + ASP (neuro-symbolic edge forecasting)  

---

## 2. IMS Dataset Integration

- Raw IMS ASCII files placed under:
Each file contains vibration signals from the bearing center.

- **Preprocessing pipeline** (`action=prepare`):
1. Reads ASCII files.  
2. Applies windowing:  
   - `length=2048`  
   - `stride=1024`  
3. Labels each window using **configurable strategy**:
   - `cut_index`: mark all after index *cut* as faulty.  
   - `last_k`: mark last *k* files as faulty.  

- Outputs cached splits:
Each `.npz` contains:
- `X`: numpy array of shape `[n_samples, window_len]`  
- `y`: labels (0 = healthy, 1 = faulty)  

Example check:
```bash
python - <<'PY'
import numpy as np, os
root="data/processed/ims_center"
for split in ["train","val","test"]:
  y=np.load(os.path.join(root,f"{split}.npz"))["y"]
  print(f"{split:5s} n={len(y):5d} pos={(y==1).sum():5d} neg={(y==0).sum():5d}")
PY
# Prepare dataset
python -m bench.cli action=prepare dataset=ims_center dataset.labeling.mode=last_k +dataset.labeling.last_k=120

# Train ELM baseline
python -m bench.cli action=train dataset=ims_center model=elm model.hidden_width=512

# Evaluate on test set
python -m bench.cli action=eval dataset=ims_center model=elm
{
  "threshold": 0.5,
  "macro_f1": 0.446,
  "auroc": 0.50,
  "precision": 0.805,
  "recall": 1.0,
  "latency_median_ms": 15.4,
  "latency_p95_ms": 18.7
}
make mlflow

---

### After running this:
1. Check file:
   ```bash
   less docs/implementation_status.md

---

### After running this:
1. Check file:
   ```bash
   less docs/implementation_status.md
