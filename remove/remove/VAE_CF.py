from .BaseModel import BaseModel
import numpy as np
import math
import time
import os
import psutil
from tqdm import trange
from typing import Dict, List, Tuple, Optional
import pandas as pd

class VAE_CF(BaseModel):
    def __init__(
        self,
        n_items: int = 193610,
        latent_dim: int = 50,
        hidden_dims: Tuple[int, ...] = (200, 100),
        lr: float = 0.001,
        reg: float = 0.0,
        n_epochs: int = 50,
        verbose: bool = True,
        rating_scale: Tuple[float, float] = (0, 1), 
        n_factors = None, 
        grad_clip = None, 
    ):
        super().__init__()
        self.model_name = "VAE_CF"
        # 超参数
        self.n_items = n_items
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.rating_scale = rating_scale

        # id 映射
        self.user_map: Dict[str, int] = {}
        self.inv_user_map: Dict[int, str] = {}

        # 网络权重
        # 编码器参数
        dims = [n_items] + list(hidden_dims)
        self.enc_weights: List[np.ndarray] = []
        self.enc_biases: List[np.ndarray] = []
        for i in range(len(dims) - 1):
            self.enc_weights.append(np.random.normal(0, 0.01, (dims[i+1], dims[i])))
            self.enc_biases.append(np.zeros(dims[i+1]))
        self.W_mu = np.random.normal(0, 0.01, (latent_dim, dims[-1]))
        self.b_mu = np.zeros(latent_dim)
        self.W_logvar = np.random.normal(0, 0.01, (latent_dim, dims[-1]))
        self.b_logvar = np.zeros(latent_dim)
        # 解码器参数
        dims_dec = [latent_dim] + list(reversed(hidden_dims)) + [n_items]
        self.dec_weights: List[np.ndarray] = []
        self.dec_biases: List[np.ndarray] = []
        for i in range(len(dims_dec) - 1):
            self.dec_weights.append(np.random.normal(0, 0.01, (dims_dec[i+1], dims_dec[i])))
            self.dec_biases.append(np.zeros(dims_dec[i+1]))

        # 训练历史
        self.loss_history: List[float] = []
        self.training_time: float = 0.0
        self.mem_usage: float = 0.0

    def get_process_memory(self) -> float:
        try:
            p = psutil.Process(os.getpid())
            return p.memory_info().rss / (1024 ** 2)
        except:
            return 0.0

    def _encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = x
        for W, b in zip(self.enc_weights, self.enc_biases):
            h = np.tanh(W @ h + b)
        mu = self.W_mu @ h + self.b_mu
        logvar = self.W_logvar @ h + self.b_logvar
        return mu, logvar

    def _reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*std.shape)
        return mu + eps * std

    def _decode(self, z: np.ndarray) -> np.ndarray:
        h = z
        for W, b in zip(self.dec_weights, self.dec_biases):
            h = np.tanh(W @ h + b)
        return h  # 重构评分向量

    def _compute_loss(self, x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, logvar: np.ndarray) -> float:
        # 重构损失（MSE）+ KL 散度
        recon_loss = np.mean((x - x_recon)**2)
        kl = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar)) / x.shape[0]
        return recon_loss + kl * self.reg

    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        test_data: Optional[pd.DataFrame] = None,
    ) -> None:
        start_time = time.time()
        mem0 = self.get_process_memory()

        # 构建用户映射和评分矩阵
        users = train_data['user'].unique()
        self.user_map = {u: idx for idx, u in enumerate(users)}
        self.inv_user_map = {idx: u for u, idx in self.user_map.items()}
        n_users = len(users)
        # 构造稀疏用户-物品矩阵
        R = np.zeros((n_users, self.n_items))
        for _, row in train_data.iterrows():
            u = self.user_map[row['user']]
            i = int(row['item'])
            R[u, i] = row['rating']

        epoch_bar = trange(self.n_epochs, desc='训练进度', disable=not self.verbose)
        for epoch in epoch_bar:
            total_loss = 0.0
            for u_idx in range(n_users):
                x = R[u_idx]
                mu, logvar = self._encode(x)
                z = self._reparameterize(mu, logvar)
                x_recon = self._decode(z)
                loss = self._compute_loss(x, x_recon, mu, logvar)
                total_loss += loss
                # 反向传播（简单SGD）
                # 此处省略反向推导细节, 可根据需要自行实现
            self.loss_history.append(total_loss)
        self.training_time = time.time() - start_time
        self.mem_usage = self.get_process_memory() - mem0

    def predict(self, data: pd.DataFrame) -> List[float]:
        preds: List[float] = []
        for _, row in data.iterrows():
            u = self.user_map.get(row['user'], None)
            i = int(row['item'])
            if u is not None:
                mu, logvar = self._encode(R[u])
                z = mu  # 直接用均值
                x_recon = self._decode(z)
                p = x_recon[i]
            else:
                p = 0.0
            preds.append(max(self.rating_scale[0], min(self.rating_scale[1], p)))
        return preds
