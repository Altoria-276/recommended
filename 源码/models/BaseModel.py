import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from tqdm import trange


class BaseModel:
    def __init__(self):
        self.model_name = "BaseModel"
        self.rmse_history: List[float] = []
        self.val_rmse_history: List[float] = []
        self.test_rmse_history: List[float] = []

    def fit(self, train_data, val_data=None, test_data=None):
        raise NotImplementedError("Subclasses must implement fit method")

    def predict(self, data):
        raise NotImplementedError("Subclasses must implement predict method")

    def _evaluate(self, data):
        raise NotImplementedError("Subclasses must implement _evaluate method")
