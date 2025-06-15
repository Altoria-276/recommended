from .BaseModel import BaseModel
import numpy as np
import math
import time
import os
import psutil
from tqdm import trange
from typing import Dict, List, Tuple, Optional
import pandas as pd

class NeuralMF(BaseModel):
    def __init__(
        self,
        n_factors: int = 20,
        hidden_dim: int = 16,
        lr: float = 0.001,
        reg: float = 0.1,
        n_epochs: int = 50,
        verbose: bool = True,
        grad_clip: float = 100.0,
        rating_scale: Tuple[float, float] = (0, 100),
    ):
        super().__init__()
        self.model_name = "NeuralMF"
        # 超参数
        self.n_factors = n_factors
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.grad_clip = grad_clip
        self.rating_scale = rating_scale

        # 模型参数
        self.global_mean = 0.0
        self.user_biases = np.array([])
        self.item_biases = np.array([])
        self.user_factors = np.array([])
        self.item_factors = np.array([])
        self.W1 = np.array([])
        self.b1 = np.array([])
        self.W2 = np.array([])
        self.b2 = 0.0

        # id 映射
        self.user_map: Dict[str, int] = {}
        self.item_map: Dict[str, int] = {}
        self.inv_user_map: Dict[int, str] = {}
        self.inv_item_map: Dict[int, str] = {}

        # 训练历史和资源记录
        self.rmse_history: List[float] = []
        self.val_rmse_history: List[float] = []
        self.test_rmse_history: List[float] = []
        self.training_time: float = 0.0
        self.mem_usage: float = 0.0

    def get_process_memory(self) -> float:
        try:
            p = psutil.Process(os.getpid())
            return p.memory_info().rss / (1024 ** 2)
        except:
            return 0.0

    def _mlp_forward(self, u_idx: int, i_idx: int) -> float:
        x = np.concatenate([self.user_factors[u_idx], self.item_factors[i_idx]])
        h = np.tanh(self.W1 @ x + self.b1)
        return self.W2 @ h + self.b2

    def _predict_pair_idx(self, u_idx: int, i_idx: int) -> float:
        return (
            self.global_mean
            + self.user_biases[u_idx]
            + self.item_biases[i_idx]
            + self._mlp_forward(u_idx, i_idx)
        )

    def _evaluate(self, data: pd.DataFrame) -> float:
        se, cnt = 0.0, 0
        for _, row in data.iterrows():
            u, i, r = row['user'], row['item'], row.get('rating', None)
            if u in self.user_map and i in self.item_map:
                pred = self._predict_pair_idx(self.user_map[u], self.item_map[i])
            elif u in self.user_map:
                pred = self.global_mean + self.user_biases[self.user_map[u]]
            elif i in self.item_map:
                pred = self.global_mean + self.item_biases[self.item_map[i]]
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
        start_time = time.time()
        start_mem = self.get_process_memory()

        # 构建映射并初始化参数
        self.user_map = {u: idx for idx, u in enumerate(train_data['user'].unique())}
        self.item_map = {i: idx for idx, i in enumerate(train_data['item'].unique())}
        self.inv_user_map = {v: k for k, v in self.user_map.items()}
        self.inv_item_map = {v: k for k, v in self.item_map.items()}
        n_users, n_items = len(self.user_map), len(self.item_map)

        self.global_mean = train_data['rating'].mean()
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        init_std = 0.01
        self.user_factors = np.random.normal(0, init_std, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, init_std, (n_items, self.n_factors))
        concat_dim = 2 * self.n_factors
        self.W1 = np.random.normal(0, init_std, (self.hidden_dim, concat_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.normal(0, init_std, self.hidden_dim)
        self.b2 = 0.0

        # 准备训练数据索引
        train = train_data.copy()
        train['u_idx'] = train['user'].map(self.user_map)
        train['i_idx'] = train['item'].map(self.item_map)

        # 训练循环
        epoch_bar = trange(self.n_epochs, desc='训练进度', disable=not self.verbose)
        for epoch in epoch_bar:
            df = train.sample(frac=1).reset_index(drop=True)
            for _, row in df.iterrows():
                u, i, r = int(row['u_idx']), int(row['i_idx']), row['rating']
                pred = self._predict_pair_idx(u, i)
                e = r - pred
                # 更新偏置
                self.user_biases[u] += self.lr * np.clip(e - self.reg * self.user_biases[u], -self.grad_clip, self.grad_clip)
                self.item_biases[i] += self.lr * np.clip(e - self.reg * self.item_biases[i], -self.grad_clip, self.grad_clip)
                # 更新MLP参数
                x = np.concatenate([self.user_factors[u], self.item_factors[i]])
                h = np.tanh(self.W1 @ x + self.b1)
                grad_out = -2 * e
                grad_h = grad_out * self.W2 * (1 - h**2)
                self.W2 -= self.lr * (grad_out * h + self.reg * self.W2)
                self.b2 -= self.lr * grad_out
                self.W1 -= self.lr * (grad_h[:, None] * x[None, :] + self.reg * self.W1)
                self.b1 -= self.lr * grad_h
                grad_x = self.W1.T @ grad_h
                self.user_factors[u] -= self.lr * (grad_x[:self.n_factors] - self.reg * self.user_factors[u])
                self.item_factors[i] -= self.lr * (grad_x[self.n_factors:] - self.reg * self.item_factors[i])

            # 记录RMSE
            self.rmse_history.append(self._evaluate(train_data))
            if val_data is not None:
                self.val_rmse_history.append(self._evaluate(val_data))
            if test_data is not None:
                self.test_rmse_history.append(self._evaluate(test_data))
            postfix = {'TrainRMSE': f'{self.rmse_history[-1]:.4f}'}
            if val_data is not None:
                postfix['ValRMSE'] = f'{self.val_rmse_history[-1]:.4f}'
            if test_data is not None:
                postfix['TestRMSE'] = f'{self.test_rmse_history[-1]:.4f}'
            epoch_bar.set_postfix(postfix)

        # 记录训练资源
        self.training_time = time.time() - start_time
        self.mem_usage = self.get_process_memory() - start_mem

    def predict(self, data: pd.DataFrame) -> List[float]:
        preds: List[float] = []
        for _, row in data.iterrows():
            u, i = row['user'], row['item']
            if u in self.user_map and i in self.item_map:
                p = self._predict_pair_idx(self.user_map[u], self.item_map[i])
            elif u in self.user_map:
                p = self.global_mean + self.user_biases[self.user_map[u]]
            elif i in self.item_map:
                p = self.global_mean + self.item_biases[self.item_map[i]]
            else:
                p = self.global_mean
            preds.append(max(self.rating_scale[0], min(self.rating_scale[1], p)))
        return preds
