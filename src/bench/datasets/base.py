from typing import Tuple, Optional
import numpy as np

class DatasetAdapter:
    name: str; modality: str; task: str
    def prepare(self, cfg) -> None: raise NotImplementedError
    def load(self, split: str) -> Tuple[np.ndarray, Optional[np.ndarray]]: raise NotImplementedError
