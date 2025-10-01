# edge-ad-bench

Config-driven, dataset-agnostic benchmarking for lightweight anomaly detection (AD) on edge targets.  
**Stage:** IMS Center (Set 2) end-to-end working with **Isolation Forest (unsupervised)** and **ELM (supervised baseline)**.  
Features: Hydra configs, CLI, MLflow tracking, ONNX/TFLite hooks (coming next), clean Makefile targets, Raspberry Pi / Jetson deploy stubs (next stage).

---

## Contents
- [Quick Start](#quick-start)
- [Project Layout](#project-layout)
- [Datasets (current stage)](#datasets-current-stage)
- [Run the Pipeline](#run-the-pipeline)
  - [A. Non-windowed (1 sample per file)](#a-non-windowed-1-sample-per-file)
  - [B. Windowed (many samples per file)](#b-windowed-many-samples-per-file)
- [Models](#models)
- [Thresholding & Metrics](#thresholding--metrics)
- [MLflow](#mlflow)
- [Makefile Shortcuts](#makefile-shortcuts)
- [Repro Tips](#repro-tips)
- [Roadmap (next)](#roadmap-next)

---

## Quick Start

> Tested on **Ubuntu 20.04**, **Python 3.8.10** (venv). Works CPU-only.

```bash
# system deps
sudo apt update && sudo apt install -y build-essential cmake pkg-config \
  libopenblas-dev liblapack-dev ffmpeg git python3-venv python3-pip

# clone (or use your existing local repo)
git clone https://github.com/cHaMiL8020/edge-ad-bench.git
cd edge-ad-bench

# venv
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip wheel setuptools

# install package (editable)
pip install -e .

# (optional) start MLflow UI in another terminal
# mlflow ui
