# edge-ad-bench

Config-driven, dataset-agnostic benchmarking for lightweight anomaly detection:
**ELM, dCeNN, Tiny CNN, Neuro-Symbolic**. CLI first, notebooks for exploration.  
Exports **ONNX** and (Tiny CNN) **TFLite int8**. Tracks with **MLflow**. Profiles
latency (median/P95), RAM, model size. Ubuntu 20.04, Python 3.8.

## Quick start

```bash
# system deps
sudo apt update && sudo apt install -y build-essential cmake pkg-config \
  libopenblas-dev liblapack-dev ffmpeg git python3-venv python3-pip
# optional for ASP
# sudo apt install -y clingo  

# setup
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip wheel setuptools
pip install -e .

# run MLflow UI (optional)
mlflow ui  

# smoke test (synthetic IMS + ELM)
make smoke
