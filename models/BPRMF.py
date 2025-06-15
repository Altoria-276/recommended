from .BaseModel import BaseModel
import numpy as np
import math
import time
import os
import psutil
from tqdm import trange
from typing import Dict, List, Tuple, Optional
import pandas as pd

class BPRMF(BaseModel):
    def __init__(
        self,
        n_factors: int = 20,
        lr: float = 0.01,
        reg: float = 0.01,
        n_epochs: int = 50,
        verbose: bool = True,
        grad_clip: float = 1.0, 
        rating_scale = None, 
    ):
        super().__init__()
        self.model_name = "BPRMF"
        # 超参数
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.grad_clip = grad_clip

        # id 映射
        self.user_map: Dict[str, int] = {}
        self.item_map: Dict[str, int] = {}
        self.inv_user_map: Dict[int, str] = {}
        self.inv_item_map: Dict[int, str] = {}

        # 嵌入矩阵
        self.user_embeddings: np.ndarray = np.array([])
        self.item_embeddings: np.ndarray = np.array([])

        # 训练历史和资源
        self.loss_history: List[float] = []
        self.training_time: float = 0.0
        self.mem_usage: float = 0.0

    def get_process_memory(self) -> float:
        try:
            p = psutil.Process(os.getpid())
            return p.memory_info().rss / (1024 ** 2)
        except:
            return 0.0

    def _sample_negative(self, user: int, user_items: Dict[int, set]) -> int:
        # 随机采样一个未交互的负样本
        n_items = self.item_embeddings.shape[0]
        while True:
            j = np.random.randint(n_items)
            if j not in user_items[user]:
                return j

    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        test_data: Optional[pd.DataFrame] = None,
    ) -> None:
        start = time.time()
        mem0 = self.get_process_memory()

        # 构建映射
        self.user_map = {u: idx for idx, u in enumerate(train_data['user'].unique())}
        self.item_map = {i: idx for idx, i in enumerate(train_data['item'].unique())}
        self.inv_user_map = {v: k for k, v in self.user_map.items()}
        self.inv_item_map = {v: k for k, v in self.item_map.items()}
        n_users = len(self.user_map)
        n_items = len(self.item_map)

        # 初始化嵌入
        init_std = 0.01
        self.user_embeddings = np.random.normal(0, init_std, (n_users, self.n_factors))
        self.item_embeddings = np.random.normal(0, init_std, (n_items, self.n_factors))

        # 用户的正样本集合
        user_items: Dict[int, set] = {}
        for _, row in train_data.iterrows():
            u = self.user_map[row['user']]
            i = self.item_map[row['item']]
            user_items.setdefault(u, set()).add(i)

            total_loss = 0.0
            for _, row in train_data.iterrows():
                u = self.user_map[row['user']]
                i = self.item_map[row['item']]
                j = self._sample_negative(u, user_items)
                # 预测评分差
                x_ui = np.dot(self.user_embeddings[u], self.item_embeddings[i])
                x_uj = np.dot(self.user_embeddings[u], self.item_embeddings[j])
                xuij = x_ui - x_uj
                # sigmoid loss gradient
                sigmoid = 1 / (1 + np.exp(-xuij))
                grad_u = (sigmoid - 1) * (self.item_embeddings[i] - self.item_embeddings[j]) + self.reg * self.user_embeddings[u]
                grad_i = (sigmoid - 1) * self.user_embeddings[u] + self.reg * self.item_embeddings[i]
                grad_j = -(sigmoid - 1) * self.user_embeddings[u] + self.reg * self.item_embeddings[j]
                # 梯度裁剪
                grad_u = np.clip(grad_u, -self.grad_clip, self.grad_clip)
                grad_i = np.clip(grad_i, -self.grad_clip, self.grad_clip)
                grad_j = np.clip(grad_j, -self.grad_clip, self.grad_clip)
                # 更新参数
                self.user_embeddings[u] -= self.lr * grad_u
                self.item_embeddings[i] -= self.lr * grad_i
                self.item_embeddings[j] -= self.lr * grad_j
                # 累积损失
                total_loss += -np.log(sigmoid) 
            
            # 每轮评估
            train_rmse = self._evaluate(train_data)
            self.rmse_history.append(train_rmse)
            postfix = {"TrainRMSE": f"{train_rmse:.4f}"}
            if val_data is not None:
                val_rmse = self._evaluate(val_data)
                self.val_rmse_history.append(val_rmse)
                postfix["ValRMSE"] = f"{val_rmse:.4f}"
            print("TrainRMSE:"+train_rmse) 
            print("ValRMSE:"+val_rmse) 
            self.loss_history.append(total_loss)
        self.training_time = time.time() - start
        self.mem_usage = self.get_process_memory() - mem0

    def predict(self, data: pd.DataFrame) -> List[float]:
        preds: List[float] = []
        for _, row in data.iterrows():
            u = self.user_map.get(row['user'], None)
            i = self.item_map.get(row['item'], None)
            if u is not None and i is not None:
                preds.append(np.dot(self.user_embeddings[u], self.item_embeddings[i]))
            else:
                preds.append(0.0)
        return preds
