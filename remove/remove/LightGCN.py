from .BaseModel import BaseModel
import numpy as np
import math
import time
import os
import psutil
from tqdm import trange
from typing import Dict, List, Tuple, Optional
import pandas as pd

class LightGCN(BaseModel):
    def __init__(
        self,
        n_factors: int = 64,
        n_layers: int = 3,
        lr: float = 0.01,
        reg: float = 1e-4,
        n_epochs: int = 100,
        verbose: bool = True,
        grad_clip: float = 1.0,
        rating_scale: Tuple[float, float] = (0, 100),
    ):
        super().__init__()
        self.model_name = "LightGCN"
        # 超参数
        self.n_factors = n_factors
        self.n_layers = n_layers
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
        self.global_mean = 0.0
        self.user_embeddings: np.ndarray = np.array([])
        self.item_embeddings: np.ndarray = np.array([])

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

    def _evaluate(self, data: pd.DataFrame) -> float:
        se, cnt = 0.0, 0
        for _, row in data.iterrows():
            u, i, r = row['user'], row['item'], row.get('rating', None)
            if u in self.user_map and i in self.item_map:
                pred = np.dot(
                    self.user_embeddings[self.user_map[u]],
                    self.item_embeddings[self.item_map[i]]
                )
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

        # 初始化嵌入
        self.user_embeddings = np.random.normal(0, 0.01, (n_users, self.n_factors))
        self.item_embeddings = np.random.normal(0, 0.01, (n_items, self.n_factors))

        # 构建邻接表
        adj: List[List[int]] = [[] for _ in range(n_users + n_items)]
        for _, row in train_data.iterrows():
            u_idx = self.user_map[row['user']]
            i_idx = self.item_map[row['item']]
            adj[u_idx].append(n_users + i_idx)
            adj[n_users + i_idx].append(u_idx)

        epoch_bar = trange(self.n_epochs, desc="训练进度", disable=not self.verbose)
        # 训练循环
        for epoch in epoch_bar:
            # 图卷积传播
            emb = np.concatenate([self.user_embeddings, self.item_embeddings], axis=0)
            emb_layers = [emb]
            for _ in range(self.n_layers):
                next_emb = np.zeros_like(emb)
                for idx, nbrs in enumerate(adj):
                    if nbrs:
                        next_emb[idx] = np.mean(emb[nbrs], axis=0)
                emb = next_emb
                emb_layers.append(emb)
            final_emb = np.mean(emb_layers, axis=0)
            self.user_embeddings = final_emb[:n_users]
            self.item_embeddings = final_emb[n_users:]

            # 评分回归梯度
            grad_u = np.zeros_like(self.user_embeddings)
            grad_i = np.zeros_like(self.item_embeddings)
            for _, row in train_data.iterrows():
                u_idx = self.user_map[row['user']]
                i_idx = self.item_map[row['item']]
                r = row['rating']
                pred = np.dot(self.user_embeddings[u_idx], self.item_embeddings[i_idx])
                err = pred - r
                grad_u[u_idx] += err * self.item_embeddings[i_idx] + self.reg * self.user_embeddings[u_idx]
                grad_i[i_idx] += err * self.user_embeddings[u_idx] + self.reg * self.item_embeddings[i_idx]

            # 梯度裁剪与更新
            grad_u = np.clip(grad_u, -self.grad_clip, self.grad_clip)
            grad_i = np.clip(grad_i, -self.grad_clip, self.grad_clip)
            self.user_embeddings -= self.lr * grad_u
            self.item_embeddings -= self.lr * grad_i

            # 每轮评估
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

        # 记录资源消耗
        self.training_time = time.time() - start_time
        self.mem_usage = self.get_process_memory() - start_mem

    def predict(self, data: pd.DataFrame) -> List[float]:
        preds: List[float] = []
        for _, row in data.iterrows():
            u, i = row['user'], row['item']
            if u in self.user_map and i in self.item_map:
                p = np.dot(
                    self.user_embeddings[self.user_map[u]],
                    self.item_embeddings[self.item_map[i]]
                )
            else:
                p = self.global_mean
            preds.append(max(self.rating_scale[0], min(self.rating_scale[1], p)))
        return preds
