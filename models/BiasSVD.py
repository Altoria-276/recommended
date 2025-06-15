import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from tqdm import trange

from .BaseModel import BaseModel


class BiasSVD(BaseModel):
    def __init__(
        self,
        n_factors: int = 20,
        lr: float = 0.0005,
        reg: float = 0.1,
        n_epochs: int = 50,
        verbose: bool = True,
        grad_clip: float = 100.0,
        rating_scale: Tuple[float, float] = (0, 100),
    ):
        super().__init__()
        self.model_name = "BiasSVD"
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.grad_clip = grad_clip
        self.rating_scale = rating_scale

        self.global_mean = 0.0
        self.user_biases = np.array([])
        self.item_biases = np.array([])
        self.user_factors = np.array([])
        self.item_factors = np.array([])

        self.user_map: Dict[str, int] = {}
        self.item_map: Dict[str, int] = {}
        self.inv_user_map: Dict[int, str] = {}
        self.inv_item_map: Dict[int, str] = {}

    def _predict_pair_idx(self, u_idx: int, i_idx: int) -> float:
        return (
            self.global_mean
            + self.user_biases[u_idx]
            + self.item_biases[i_idx]
            + np.dot(self.user_factors[u_idx], self.item_factors[i_idx])
        )

    def _evaluate(self, data):
        se, cnt = 0.0, 0
        for _, row in data.iterrows():
            u, i, r = row["user"], row["item"], row["rating"]
            if u in self.user_map and i in self.item_map:
                pred = self._predict_pair_idx(self.user_map[u], self.item_map[i])
            elif u in self.user_map:
                pred = self.global_mean + self.user_biases[self.user_map[u]]
            elif i in self.item_map:
                pred = self.global_mean + self.item_biases[self.item_map[i]]
            else:
                pred = self.global_mean
            pred = max(self.rating_scale[0], min(self.rating_scale[1], pred))
            se += (r - pred) ** 2
            cnt += 1
        return math.sqrt(se / cnt) if cnt > 0 else float("nan")

    def fit(
        self,
        train_data,
        val_data=None,
        test_data=None,
    ) -> None:
        # 构建映射
        self.user_map = {u: idx for idx, u in enumerate(train_data["user"].unique())}
        self.item_map = {i: idx for idx, i in enumerate(train_data["item"].unique())}
        self.inv_user_map = {v: k for k, v in self.user_map.items()}
        self.inv_item_map = {v: k for k, v in self.item_map.items()}
        n_users, n_items = len(self.user_map), len(self.item_map)

        # 初始化参数
        self.global_mean = train_data["rating"].mean()
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        init_std = 0.01
        self.user_factors = np.random.normal(0, init_std, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, init_std, (n_items, self.n_factors))

        # 内部ID
        train = train_data.copy()
        train["u_idx"] = train["user"].map(self.user_map)
        train["i_idx"] = train["item"].map(self.item_map)

        epoch_bar = trange(self.n_epochs, desc="训练进度", disable=not self.verbose)
        for epoch in epoch_bar:
            df = train.sample(frac=1).reset_index(drop=True)
            for _, row in df.iterrows():
                u, i, r = int(row["u_idx"]), int(row["i_idx"]), row["rating"]
                pred = self._predict_pair_idx(u, i)
                e = r - pred
                # 更新偏置
                ub = np.clip(e - self.reg * self.user_biases[u], -self.grad_clip, self.grad_clip)
                ib = np.clip(e - self.reg * self.item_biases[i], -self.grad_clip, self.grad_clip)
                self.user_biases[u] += self.lr * ub
                self.item_biases[i] += self.lr * ib
                # 更新因子
                uf, vf = self.user_factors[u], self.item_factors[i]
                ug = np.clip(e * vf - self.reg * uf, -self.grad_clip, self.grad_clip)
                vg = np.clip(e * uf - self.reg * vf, -self.grad_clip, self.grad_clip)
                self.user_factors[u] += self.lr * ug
                self.item_factors[i] += self.lr * vg

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

    def predict(self, data):
        preds: List[float] = []
        for _, row in data.iterrows():
            u, i = row["user"], row["item"]
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
