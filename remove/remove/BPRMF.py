from .BaseModel import BaseModel
import numpy as np
import time
import os
import psutil
from tqdm import trange
from typing import Dict, List, Optional
import pandas as pd
import math
from sklearn.metrics import roc_auc_score

class BPRMF(BaseModel):
    def __init__(
        self,
        n_factors: int = 20,
        lr: float = 0.01,
        reg: float = 0.01,
        n_epochs: int = 50,
        verbose: bool = True,
        grad_clip: float = 1.0,
        rating_scale: Optional[tuple] = None,
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
        # 可选评分范围，用于预测截断
        self.rating_scale = rating_scale

        # id 映射
        self.user_map: Dict = {}
        self.item_map: Dict = {}
        self.inv_user_map: Dict = {}
        self.inv_item_map: Dict = {}

        # 嵌入矩阵
        self.user_embeddings: np.ndarray = np.array([])
        self.item_embeddings: np.ndarray = np.array([])

        # 训练历史和资源
        self.loss_history: List[float] = []
        self.rmse_history: List[float] = []
        self.val_rmse_history: List[float] = []
        self.val_auc_history: List[float] = []
        self.test_rmse_history: List[float] = []
        self.training_time: float = 0.0
        self.mem_usage: float = 0.0

    def get_process_memory(self) -> float:
        try:
            p = psutil.Process(os.getpid())
            return p.memory_info().rss / (1024 ** 2)
        except Exception:
            return 0.0

    def _sample_negative(self, u: int, user_items: Dict[int, set]) -> int:
        n_items = self.item_embeddings.shape[0]
        while True:
            j = np.random.randint(n_items)
            if j not in user_items[u]:
                return j

    def _evaluate(self, data: pd.DataFrame) -> float:
        """
        计算 RMSE：对每个 (user,item,rating) 预测评分并与真实值比较
        """
        se, cnt = 0.0, 0
        for _, row in data.iterrows():
            u_id, i_id, r = row['user'], row['item'], row.get('rating', None)
            u = self.user_map.get(u_id)
            i = self.item_map.get(i_id)
            if u is not None and i is not None:
                pred = self.user_embeddings[u].dot(self.item_embeddings[i])
                if self.rating_scale is not None:
                    pred = max(self.rating_scale[0], min(self.rating_scale[1], pred))
            else:
                pred = 0.0
            if r is not None:
                se += (r - pred) ** 2
                cnt += 1
        return math.sqrt(se / cnt) if cnt > 0 else float('nan')

    def _evaluate_auc(self, data: pd.DataFrame) -> float:
        """
        计算 AUC：为每个正样本对采一个负样本来衡量排序能力
        """
        y_true, y_score = [], []
        # 构造训练集正样本集合用于采 negative
        user_items = {u: set() for u in range(len(self.user_map))}
        for _, row in data.iterrows():
            u_id, i_id = row['user'], row['item']
            u = self.user_map.get(u_id)
            i = self.item_map.get(i_id)
            if u is not None and i is not None:
                user_items[u].add(i)
        # 采样并打分
        for _, row in data.iterrows():
            u_id, i_id = row['user'], row['item']
            u = self.user_map.get(u_id)
            pos = self.item_map.get(i_id)
            if u is None or pos is None:
                continue
            # 正样本
            y_true.append(1)
            y_score.append(self.user_embeddings[u].dot(self.item_embeddings[pos]))
            # 负样本
            neg = self._sample_negative(u, user_items)
            y_true.append(0)
            y_score.append(self.user_embeddings[u].dot(self.item_embeddings[neg]))
        if len(y_true) < 2 or len(set(y_true)) < 2:
            return float('nan')
        return roc_auc_score(y_true, y_score)

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
        n_users, n_items = len(self.user_map), len(self.item_map)

        # 初始化嵌入
        init_std = 0.01
        self.user_embeddings = np.random.normal(0, init_std, (n_users, self.n_factors))
        self.item_embeddings = np.random.normal(0, init_std, (n_items, self.n_factors))

        # 构建用户正样本集合
        user_items: Dict[int, set] = {}
        for _, row in train_data.iterrows():
            u = self.user_map[row['user']]
            i = self.item_map[row['item']]
            user_items.setdefault(u, set()).add(i)

        # 训练循环
        for epoch in range(1, self.n_epochs + 1):
            total_loss = 0.0
            df = train_data.sample(frac=1).reset_index(drop=True)
            for _, row in df.iterrows():
                u = self.user_map[row['user']]
                i = self.item_map[row['item']]
                j = self._sample_negative(u, user_items)
                x_ui = self.user_embeddings[u].dot(self.item_embeddings[i])
                x_uj = self.user_embeddings[u].dot(self.item_embeddings[j])
                xuij = x_ui - x_uj
                sigmoid = 1.0 / (1.0 + np.exp(-xuij))
                grad_u = (sigmoid - 1) * (self.item_embeddings[i] - self.item_embeddings[j]) + self.reg * self.user_embeddings[u]
                grad_i = (sigmoid - 1) * self.user_embeddings[u] + self.reg * self.item_embeddings[i]
                grad_j = -(sigmoid - 1) * self.user_embeddings[u] + self.reg * self.item_embeddings[j]
                self.user_embeddings[u] -= self.lr * np.clip(grad_u, -self.grad_clip, self.grad_clip)
                self.item_embeddings[i] -= self.lr * np.clip(grad_i, -self.grad_clip, self.grad_clip)
                self.item_embeddings[j] -= self.lr * np.clip(grad_j, -self.grad_clip, self.grad_clip)
                total_loss += -np.log(sigmoid)

            # 记录并输出
            self.loss_history.append(total_loss)
            train_rmse = self._evaluate(train_data)
            self.rmse_history.append(train_rmse)
            msg = f"Epoch {epoch}/{self.n_epochs} - loss: {total_loss:.4f} - TrainRMSE: {train_rmse:.4f}"
            if val_data is not None:
                val_rmse = self._evaluate(val_data)
                self.val_rmse_history.append(val_rmse)
                msg += f" - ValRMSE: {val_rmse:.4f}"
                val_auc = self._evaluate_auc(val_data)
                self.val_auc_history.append(val_auc)
                msg += f" - ValAUC: {val_auc:.4f}"
            if test_data is not None:
                test_rmse = self._evaluate(test_data)
                self.test_rmse_history.append(test_rmse)
                msg += f" - TestRMSE: {test_rmse:.4f}"
            if self.verbose:
                print(msg)

        # 记录资源消耗
        self.training_time = time.time() - start
        self.mem_usage = self.get_process_memory() - mem0

    def predict(self, data: pd.DataFrame) -> List[float]:
        preds: List[float] = []
        for _, row in data.iterrows():
            u_id, i_id = row['user'], row['item']
            u = self.user_map.get(u_id)
            i = self.item_map.get(i_id)
            if u is not None and i is not None:
                p = self.user_embeddings[u].dot(self.item_embeddings[i])
                if self.rating_scale is not None:
                    p = max(self.rating_scale[0], min(self.rating_scale[1], p))
            else:
                p = 0.0
            preds.append(p)
        return preds
