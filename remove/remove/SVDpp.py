from .BaseModel import BaseModel
import numpy as np
import math
import time
import os
import psutil
from tqdm import trange
from typing import Dict, List, Tuple, Optional
import pandas as pd

class SVDpp(BaseModel):
    def __init__(
        self,
        n_factors: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        n_epochs: int = 50,
        verbose: bool = True,
        grad_clip: float = 100.0,
        rating_scale: Tuple[float, float] = (0, 100),
    ):
        super().__init__()
        self.model_name = "SVDpp"
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.grad_clip = grad_clip
        self.rating_scale = rating_scale

        self.user_map: Dict[str, int] = {}
        self.item_map: Dict[str, int] = {}
        self.inv_user_map: Dict[int, str] = {}
        self.inv_item_map: Dict[int, str] = {}

        self.global_mean: float = 0.0
        self.user_biases: np.ndarray = np.array([])
        self.item_biases: np.ndarray = np.array([])
        self.user_factors: np.ndarray = np.array([])
        self.item_factors: np.ndarray = np.array([])
        self.y_factors: np.ndarray = np.array([])

        self.rmse_history: List[float] = []
        self.val_rmse_history: List[float] = []
        self.test_rmse_history: List[float] = []
        self.training_time: float = 0.0
        self.mem_usage: float = 0.0

    def get_process_memory(self) -> float:
        try:
            p = psutil.Process(os.getpid())
            return p.memory_info().rss / (1024 ** 2)
        except Exception:
            return 0.0

    def _predict_pair_idx(self, u: int, i: int, Ru: List[int]) -> float:
        sqrt_N = np.sqrt(len(Ru)) if Ru else 1.0
        y_sum = np.sum(self.y_factors[Ru], axis=0) / sqrt_N if Ru else 0.0
        pu = self.user_factors[u] + y_sum
        qi = self.item_factors[i]
        return self.global_mean + self.user_biases[u] + self.item_biases[i] + pu.dot(qi)

    def _evaluate(self, data: pd.DataFrame) -> float:
        se, cnt = 0.0, 0
        for _, row in data.iterrows():
            u_id, i_id, r = row['user'], row['item'], row.get('rating', None)
            u = self.user_map.get(u_id)
            i = self.item_map.get(i_id)
            if u is not None and i is not None:
                Ru = self.user_ratings.get(u, [])
                pred = self._predict_pair_idx(u, i, Ru)
            else:
                pred = self.global_mean
            pred = max(self.rating_scale[0], min(self.rating_scale[1], pred))
            if r is not None:
                se += (r - pred) ** 2
                cnt += 1
        return math.sqrt(se / cnt) if cnt > 0 else float('nan')

    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        test_data: Optional[pd.DataFrame] = None,
    ) -> None:
        # 构建映射
        self.user_map = {u: idx for idx, u in enumerate(train_data['user'].unique())}
        self.item_map = {i: idx for idx, i in enumerate(train_data['item'].unique())}
        self.inv_user_map = {v: k for k, v in self.user_map.items()}
        self.inv_item_map = {v: k for k, v in self.item_map.items()}
        n_users, n_items = len(self.user_map), len(self.item_map)
        self.global_mean = train_data['rating'].mean()

        # 初始化参数
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        init = 0.01
        self.user_factors = np.random.normal(0, init, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, init, (n_items, self.n_factors))
        self.y_factors = np.random.normal(0, init, (n_items, self.n_factors))

        # 构建用户隐式反馈集合
        self.user_ratings: Dict[int, List[int]] = {}
        for _, row in train_data.iterrows():
            u = self.user_map[row['user']]
            i = self.item_map[row['item']]
            self.user_ratings.setdefault(u, []).append(i)

        start = time.time()
        mem0 = self.get_process_memory()

        epoch_bar = trange(self.n_epochs, desc="训练进度", disable=not self.verbose)
        for epoch in epoch_bar:
            # 随机打乱训练样本
            df = train_data.sample(frac=1).reset_index(drop=True)
            for _, row in df.iterrows():
                u = self.user_map[row['user']]
                i = self.item_map[row['item']]
                r = row['rating']
                Ru = self.user_ratings[u]
                pred = self._predict_pair_idx(u, i, Ru)
                e = r - pred
                # 更新偏置
                self.user_biases[u] += self.lr * (e - self.reg * self.user_biases[u])
                self.item_biases[i] += self.lr * (e - self.reg * self.item_biases[i])
                # 更新隐式和显式因子
                sqrt_N = np.sqrt(len(Ru)) if Ru else 1.0
                y_sum = np.sum(self.y_factors[Ru], axis=0) / sqrt_N if Ru else np.zeros(self.n_factors)
                pu = self.user_factors[u] + y_sum

                grad_qi = e * pu - self.reg * self.item_factors[i]
                grad_pu = e * self.item_factors[i] - self.reg * self.user_factors[u]
                self.item_factors[i] += self.lr * np.clip(grad_qi, -self.grad_clip, self.grad_clip)
                self.user_factors[u] += self.lr * np.clip(grad_pu, -self.grad_clip, self.grad_clip)

                if Ru:
                    grad_y = (e * self.item_factors[i] / sqrt_N)
                    for j in Ru:
                        self.y_factors[j] += self.lr * np.clip(grad_y - self.reg * self.y_factors[j], -self.grad_clip, self.grad_clip)

            # 记录并输出
            train_rmse = self._evaluate(train_data)
            self.rmse_history.append(train_rmse)
            postfix = {"TrainRMSE": f"{train_rmse:.4f}"}
            if val_data is not None:
                val_rmse = self._evaluate(val_data)
                self.val_rmse_history.append(val_rmse)
                postfix["ValRMSE"] = f"{val_rmse:.4f}"
            if test_data is not None:
                test_rmse = self._evaluate(test_data)
                self.test_rmse_history.append(test_rmse)
                postfix["TestRMSE"] = f"{test_rmse:.4f}"
            epoch_bar.set_postfix(postfix)

        self.training_time = time.time() - start
        self.mem_usage = self.get_process_memory() - mem0

    def predict(self, data: pd.DataFrame) -> List[float]:
        preds: List[float] = []
        # 构建测试用户隐式反馈
        test_ratings = {u: [self.item_map[i] for i in data[data['user'] == uid]['item'] if i in self.item_map]
                        for uid, u in self.user_map.items()}
        for _, row in data.iterrows():
            u = self.user_map.get(row['user'], None)
            i = self.item_map.get(row['item'], None)
            if u is not None and i is not None:
                Ru = test_ratings.get(u, [])
                p = self._predict_pair_idx(u, i, Ru)
            else:
                p = self.global_mean
            preds.append(max(self.rating_scale[0], min(self.rating_scale[1], p)))
        return preds
