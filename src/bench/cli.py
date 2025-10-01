import hydra
from omegaconf import DictConfig
from .registry import DATASETS, MODELS
from .datasets.ims import IMSDataset   # synthetic
from .datasets.ims_center import IMSCenterDataset # real set1/2/3
from .models.elm import ELM            # ensure registration
from .train_loop import run_train
from .eval_loop import run_eval
from .models.iforest import IForest


@hydra.main(version_base=None, config_path="../../configs", config_name="experiment")
def main(cfg: DictConfig):
    if cfg.action == "prepare":
        ds = DATASETS[cfg.dataset.name](cfg.dataset); ds.prepare(cfg.dataset)
        print("Prepared:", cfg.dataset.name)
    elif cfg.action == "train":
        out = run_train(cfg); print("Trained. Artifacts ->", out)
    elif cfg.action == "eval":
        m = run_eval(cfg); print("Eval:", m)
    elif cfg.action == "export":
        print("Export: TODO")
    elif cfg.action == "profile":
        print("Profile: TODO")
    else:
        print("Unknown action")
if __name__ == "__main__":
    main()
