import os, json, random
import numpy as np
import mlflow

def set_seeds(seed: int=42):
    random.seed(seed); np.random.seed(seed)

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)

def save_json(obj, path):
    with open(path, "w") as f: f.write(json.dumps(obj, indent=2))

def start_mlflow(uri: str, run_name: str):
    mlflow.set_tracking_uri(uri)
    return mlflow.start_run(run_name=run_name)
