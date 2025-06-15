from .BaseModel import BaseModel
import numpy as np
import math
import time
import os
import psutil
from tqdm import trange
from typing import Dict, List, Tuple, Optional
import pandas as pd

class NeuMF(BaseModel):
    def __init__(
        self,
        n_factors: int = 32,
        mlp_hidden_dims: Tuple[int, ...] = (64, 32, 16, 8),
        lr: float = 0.001,
        reg: float = 0.0,
        n_epochs: int = 50,
        verbose: bool = True,
        grad_clip: float = 100.0,
        rating_scale: Tuple[float, float] = (0, 100),
    ):
        super().__init__()
        self.model_name = "NeuMF"
        # 超参数
        self.n_factors = n_factors
        self.mlp_hidden_dims = mlp_hidden_dims
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.grad_clip = grad_clip
        self.rating_scale = rating_scale

        # id 映射
        self.user_map: Dict[str, int] = {}
        self.item_map: Dict[str, int] = {}
        self.inv_user_map: Dict[int, str] = {}
        self.inv_item_map: Dict[int, str] = {}

        # 全局统计
        self.global_mean: float = 0.0
        # 嵌入
        self.user_embed_gmf: np.ndarray = np.array([])
        self.item_embed_gmf: np.ndarray = np.array([])
        self.user_embed_mlp: np.ndarray = np.array([])
        self.item_embed_mlp: np.ndarray = np.array([])
        # MLP 权重
        self.mlp_weights: List[np.ndarray] = []
        self.mlp_biases: List[np.ndarray] = []
        # 输出层参数
        self.hybrid_weight: np.ndarray = np.array([])
        self.hybrid_bias: float = 0.0

        # 训练历史
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

    def _mlp_forward(self, u_idx: int, i_idx: int) -> np.ndarray:
        x = np.concatenate([
            self.user_embed_mlp[u_idx],
            self.item_embed_mlp[i_idx]
        ])
        for W, b in zip(self.mlp_weights, self.mlp_biases):
            x = np.tanh(W @ x + b)
        return x

    def _predict_pair_idx(self, u_idx: int, i_idx: int) -> float:
        # GMF part
        gmf_out = self.user_embed_gmf[u_idx] * self.item_embed_gmf[i_idx]
        # MLP part
        mlp_out = self._mlp_forward(u_idx, i_idx)
        # concat and output
        vector = np.concatenate([gmf_out, mlp_out])
        return self.global_mean + self.hybrid_weight @ vector + self.hybrid_bias

    def _evaluate(self, data: pd.DataFrame) -> float:
        se, cnt = 0.0, 0
        for _, row in data.iterrows():
            u_id, i_id, r = row['user'], row['item'], row.get('rating', None)
            u = self.user_map.get(u_id)
            i = self.item_map.get(i_id)
            if u is not None and i is not None:
                pred = self._predict_pair_idx(u, i)
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

        # 构建映射
        self.user_map = {u: idx for idx, u in enumerate(train_data['user'].unique())}
        self.item_map = {i: idx for idx, i in enumerate(train_data['item'].unique())}
        self.inv_user_map = {v: k for k, v in self.user_map.items()}
        self.inv_item_map = {v: k for k, v in self.item_map.items()}
        n_users, n_items = len(self.user_map), len(self.item_map)
        self.global_mean = train_data['rating'].mean()

        # 初始化嵌入和MLP参数
        init_std = 0.01
        self.user_embed_gmf = np.random.normal(0, init_std, (n_users, self.n_factors))
        self.item_embed_gmf = np.random.normal(0, init_std, (n_items, self.n_factors))
        self.user_embed_mlp = np.random.normal(0, init_std, (n_users, self.n_factors))
        self.item_embed_mlp = np.random.normal(0, init_std, (n_items, self.n_factors))
        input_dim = 2 * self.n_factors
        for dim in self.mlp_hidden_dims:
            W = np.random.normal(0, init_std, (dim, input_dim))
            b = np.zeros(dim)
            self.mlp_weights.append(W)
            self.mlp_biases.append(b)
            input_dim = dim
        # 输出层参数
        self.hybrid_weight = np.random.normal(0, init_std, (self.n_factors + self.mlp_hidden_dims[-1]))
        self.hybrid_bias = 0.0

        epoch_bar = trange(self.n_epochs, desc='训练进度', disable=not self.verbose)
        for epoch in epoch_bar:
            for _, row in train_data.iterrows():
                u = self.user_map[row['user']]
                i = self.item_map[row['item']]
                r = row['rating']
                pred = self._predict_pair_idx(u, i)
                e = r - pred
                # 输出层梯度
                vector = np.concatenate([
                    self.user_embed_gmf[u] * self.item_embed_gmf[i],
                    self._mlp_forward(u, i)
                ])
                grad_h = -e + self.reg * self.hybrid_weight
                self.hybrid_weight -= self.lr * (grad_h * vector)
                self.hybrid_bias -= self.lr * grad_h
                # GMF梯度
                grad_gmf = -e * self.hybrid_weight[:self.n_factors] * (self.item_embed_gmf[i]) + self.reg * self.user_embed_gmf[u]
                self.user_embed_gmf[u] -= self.lr * grad_gmf
                self.item_embed_gmf[i] -= self.lr * (-e * self.hybrid_weight[:self.n_factors] * self.user_embed_gmf[u] + self.reg * self.item_embed_gmf[i])
                # MLP梯度
                mlp_grad = -e * self.hybrid_weight[self.n_factors:] 
                # 反向传播MLP
                grad = mlp_grad
                for idx in reversed(range(len(self.mlp_weights))):
                    W, b = self.mlp_weights[idx], self.mlp_biases[idx]
                    # x 输入来自前一层或嵌入拼接
                    if idx == 0:
                        x = np.concatenate([self.user_embed_mlp[u], self.item_embed_mlp[i]])
                    else:
                        x = np.tanh(self.mlp_weights[idx-1] @ x + self.mlp_biases[idx-1])
                    grad_W = np.outer(grad * (1 - x**2), x) + self.reg * W
                    grad_b = grad * (1 - x**2)
                    grad = W.T @ (grad * (1 - x**2))
                    self.mlp_weights[idx] -= self.lr * grad_W
                    self.mlp_biases[idx] -= self.lr * grad_b
                # 嵌入梯度
                self.user_embed_mlp[u] -= self.lr * (grad[:self.n_factors] + self.reg * self.user_embed_mlp[u])
                self.item_embed_mlp[i] -= self.lr * (grad[self.n_factors:] + self.reg * self.item_embed_mlp[i])

            # 记录指标
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

        self.training_time = time.time() - start_time
        self.mem_usage = self.get_process_memory() - start_mem

    def predict(self, data: pd.DataFrame) -> List[float]:
        preds: List[float] = []
        for _, row in data.iterrows():
            u_id, i_id = row['user'], row['item']
            u = self.user_map.get(u_id)
            i = self.item_map.get(i_id)
            if u is not None and i is not None:
                p = self._predict_pair_idx(u, i)
            else:
                p = self.global_mean
            preds.append(max(self.rating_scale[0], min(self.rating_scale[1], p)))
        return preds
