import numpy as np
import math
from typing import Dict, List, Tuple
from tqdm import trange

from .BaseModel import BaseModel


class SVD(BaseModel):
    """
    经典矩阵分解模型（SVD）实现：
    \hat{r}_{ui} = U_u^T V_i
    若遇冷启动（用户或物品未知），使用以下策略：
    - 仅有用户：返回该用户训练集平均评分
    - 仅有物品：返回该物品训练集平均评分
    - 均无：返回训练集全局平均评分
    """
    def __init__(
        self,
        n_factors: int = 20,
        lr: float = 0.0005,
        reg: float = 0.1,
        n_epochs: int = 50,
        verbose: bool = True,
        grad_clip: float = 100.0,
        rating_scale: Tuple[float, float] = (0.0, 100.0),
    ):
        super().__init__()
        self.model_name = "SVD"
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.grad_clip = grad_clip
        self.rating_scale = rating_scale

        # 因子矩阵
        self.user_factors: np.ndarray = np.array([])
        self.item_factors: np.ndarray = np.array([])

        # 用户/物品映射
        self.user_map: Dict[str, int] = {}
        self.item_map: Dict[str, int] = {}
        self.inv_user_map: Dict[int, str] = {}
        self.inv_item_map: Dict[int, str] = {}

        # 用户/物品均值
        self.user_mean: Dict[str, float] = {}
        self.item_mean: Dict[str, float] = {}
        self.global_mean: float = 0.0

        # 学习率策略
        self.lr_schedule = "constant"
        self.warmup_epochs = 5
        self.decay_ratio = 0.1
        self.step_drop = 0.5
        self.step_every = 20

    def get_current_lr(self, epoch: int) -> float:
        if self.lr_schedule == "constant":
            return self.lr
        elif self.lr_schedule == "warmup_decay":
            if epoch < self.warmup_epochs:
                return self.lr * (epoch + 1) / self.warmup_epochs
            total_decay = self.n_epochs - self.warmup_epochs
            progress = (epoch - self.warmup_epochs) / max(1, total_decay)
            return self.lr * ((1 - progress) + self.decay_ratio * progress)
        elif self.lr_schedule == "step_decay":
            return self.lr * (self.step_drop ** (epoch // self.step_every))
        else:
            raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")

    def _predict_pair_idx(self, u_idx: int, i_idx: int) -> float:
        return float(np.dot(self.user_factors[u_idx], self.item_factors[i_idx].T))

    def _evaluate(self, data) -> float:
        preds = []
        for _, row in data.iterrows(): 
            u, i = row["user"], row["item"]
            if u in self.user_map and i in self.item_map: 
                pred = self._predict_pair_idx(self.user_map[u], self.item_map[i])
            elif u in self.user_mean: 
                pred = self.user_mean[u]
            elif i in self.item_mean:
                pred = self.item_mean[i]
            else:
                pred = self.global_mean
            pred = np.clip(pred, *self.rating_scale)
            preds.append(pred)
        ratings = data["rating"].values
        se = np.square(ratings - np.array(preds)).sum()
        return math.sqrt(se / len(data))

    def fit(self, train_data, val_data=None, test_data=None) -> None:
        self.user_map = {u: idx for idx, u in enumerate(train_data["user"].unique())}
        self.item_map = {i: idx for idx, i in enumerate(train_data["item"].unique())}
        self.inv_user_map = {v: k for k, v in self.user_map.items()}
        self.inv_item_map = {v: k for k, v in self.item_map.items()}

        n_users = len(self.user_map)
        n_items = len(self.item_map)

        # 初始化因子
        init_std = 0.01 

        self.user_factors = np.random.normal(0, init_std, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, init_std, (n_items, self.n_factors))

        # 构建均值
        self.global_mean = train_data["rating"].mean()
        self.user_mean = train_data.groupby("user")["rating"].mean().to_dict()
        self.item_mean = train_data.groupby("item")["rating"].mean().to_dict()

        df = train_data.copy()
        df["u_idx"] = df["user"].map(self.user_map).astype(int)
        df["i_idx"] = df["item"].map(self.item_map).astype(int)

        epoch_iter = trange(self.n_epochs, desc="训练进度", disable=not self.verbose)
        for epoch in epoch_iter:
            cur_lr = self.get_current_lr(epoch)
            shuffled = df.sample(frac=1).reset_index(drop=True)
            for _, row in shuffled.iterrows():
                u = int(row["u_idx"])
                i = int(row["i_idx"])
                r = row["rating"]
                pred = np.dot(self.user_factors[u], self.item_factors[i].T)
                e = r - pred
                grad_u = np.clip(2 * e * self.item_factors[i] - self.reg * self.user_factors[u], -self.grad_clip, self.grad_clip)
                grad_v = np.clip(2 * e * self.user_factors[u] - self.reg * self.item_factors[i], -self.grad_clip, self.grad_clip)
                self.user_factors[u] += cur_lr * grad_u
                self.item_factors[i] += cur_lr * grad_v

            train_rmse = self._evaluate(train_data)
            self.rmse_history.append(train_rmse)
            postfix = {"TrainRMSE": f"{train_rmse:.4f}", "lr": f"{cur_lr:.4f}"}
            if val_data is not None:
                val_rmse = self._evaluate(val_data)
                self.val_rmse_history.append(val_rmse)
                postfix["ValRMSE"] = f"{val_rmse:.4f}"
            if test_data is not None:
                test_rmse = self._evaluate(test_data)
                self.test_rmse_history.append(test_rmse)
                postfix["TestRMSE"] = f"{test_rmse:.4f}"
            epoch_iter.set_postfix(postfix)

    def predict(self, data) -> List[float]:
        preds = []
        for _, row in data.iterrows():
            u, i = row["user"], row["item"]
            if u in self.user_map and i in self.item_map:
                pred = self._predict_pair_idx(self.user_map[u], self.item_map[i])
            elif u in self.user_mean:
                pred = self.user_mean[u]
            elif i in self.item_mean:
                pred = self.item_mean[i]
            else:
                pred = self.global_mean
            preds.append(np.clip(pred, *self.rating_scale))
        return preds
