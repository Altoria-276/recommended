import numpy as np
import pandas as pd
import re
import math
import time
import os
import psutil
from tqdm import trange
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split


def parse_data(file_path: Union[str, Path], has_ratings: bool = True) -> pd.DataFrame:
    users: List[str] = []
    items: List[str] = []
    ratings: Optional[List[float]] = [] if has_ratings else None

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if '|' in lines[i]:
            user_id, cnt = lines[i].strip().split('|')
            cnt = int(cnt)
            for j in range(1, cnt + 1):
                if i + j >= len(lines): break
                parts = re.split(r"\s+", lines[i + j].strip())
                item_id = parts[0]
                users.append(user_id)
                items.append(item_id)
                if has_ratings and ratings is not None:
                    ratings.append(float(parts[1]) if len(parts) > 1 else 0.0)
            i += cnt + 1
        else:
            i += 1
    df = pd.DataFrame({'user': users, 'item': items})
    if has_ratings and ratings is not None:
        df['rating'] = ratings  # type: ignore
    return df


class NeuralMF:
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
        self.n_factors = n_factors
        self.hidden_dim = hidden_dim
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

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        self.user_map: Dict[str, int] = {}
        self.item_map: Dict[str, int] = {}
        self.inv_user_map: Dict[int, str] = {}
        self.inv_item_map: Dict[int, str] = {}

        self.rmse_history: List[float] = []
        self.val_rmse_history: List[float] = []
        self.test_rmse_history: List[float] = []
        self.training_time = 0.0
        self.mem_usage = 0.0

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
        self.W1 = np.random.normal(0, 0.01, (self.hidden_dim, concat_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.normal(0, 0.01, self.hidden_dim)
        self.b2 = 0.0

        train = train_data.copy()
        train['u_idx'] = train['user'].map(self.user_map)
        train['i_idx'] = train['item'].map(self.item_map)

        epoch_bar = trange(self.n_epochs, desc='训练进度', disable=not self.verbose) 
        for epoch in epoch_bar:
            df = train.sample(frac=1).reset_index(drop=True)
            for _, row in df.iterrows():
                u, i, r = int(row['u_idx']), int(row['i_idx']), row['rating']
                # forward pass
                x = np.concatenate([self.user_factors[u], self.item_factors[i]])
                h = np.tanh(self.W1 @ x + self.b1)
                mlp_out = self.W2 @ h + self.b2
                pred = self.global_mean + self.user_biases[u] + self.item_biases[i] + mlp_out
                e = r - pred
                # gradients
                grad_out = -2 * e
                # bias updates
                self.user_biases[u] -= self.lr * (grad_out + self.reg * self.user_biases[u])
                self.item_biases[i] -= self.lr * (grad_out + self.reg * self.item_biases[i])
                # MLP parameter updates
                grad_h = grad_out * self.W2
                grad_h = grad_h * (1 - h ** 2)
                self.W2 -= self.lr * (grad_out * h + self.reg * self.W2)
                self.b2 -= self.lr * grad_out
                self.W1 -= self.lr * (grad_h[:, None] * x[None, :] + self.reg * self.W1)
                self.b1 -= self.lr * grad_h
                # factor updates via gradient to x
                grad_x = self.W1.T @ grad_h
                grad_uf = grad_x[:self.n_factors] - self.reg * self.user_factors[u]
                grad_if = grad_x[self.n_factors:] - self.reg * self.item_factors[i]
                self.user_factors[u] -= self.lr * grad_uf
                self.item_factors[i] -= self.lr * grad_if
            train_rmse = self._evaluate(train_data)
            self.rmse_history.append(train_rmse)
            postfix = {'TrainRMSE': f'{train_rmse:.4f}'}
            if val_data is not None:
                val_rmse = self._evaluate(val_data)
                self.val_rmse_history.append(val_rmse)
                postfix['ValRMSE'] = f'{val_rmse:.4f}'
            if test_data is not None:
                test_rmse = self._evaluate(test_data)
                self.test_rmse_history.append(test_rmse)
                postfix['TestRMSE'] = f'{test_rmse:.4f}'
            epoch_bar.set_postfix(postfix)

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

# save_training_stats, save_predictions, main remain unchanged (omitted for brevity)

def save_training_stats(stats_path: Path, model: NeuralMF, args: argparse.Namespace) -> None:
    with stats_path.open('w', encoding='utf-8') as f:
        f.write("===== 训练统计信息 =====\n\n")
        f.write(f"模型: NeuralMF\n")
        f.write(f"评分范围: {model.rating_scale[0]}-{model.rating_scale[1]}\n")
        f.write(f"训练文件: {args.train}\n")
        f.write(f"测试文件: {args.test}\n")
        f.write(f"输出文件: {args.output}\n")
        f.write(f"统计文件: {stats_path}\n\n")

        f.write("===== 模型参数 =====\n")
        f.write(f"隐因子维度: {args.factors}\n")
        f.write(f"隐藏层维度: {args.hidden_dim}\n")
        f.write(f"训练轮数: {args.epochs}\n")
        f.write(f"学习率: {args.lr}\n")
        f.write(f"正则化系数: {args.reg}\n")
        f.write(f"梯度剪裁阈值: {args.grad_clip}\n")

        f.write("===== 性能指标 =====\n")
        f.write(f"总训练时间: {model.training_time:.2f} 秒\n")
        f.write(f"平均每轮时间: {model.training_time/args.epochs:.2f} 秒\n")
        f.write(f"内存消耗: {model.mem_usage:.2f} MB\n")

        if model.rmse_history:
            f.write(f"最终 Train RMSE: {model.rmse_history[-1]:.4f}\n")
        if model.val_rmse_history:
            f.write(f"最终 Val RMSE: {model.val_rmse_history[-1]:.4f}\n")
        if model.test_rmse_history:
            f.write(f"最终 Test RMSE: {model.test_rmse_history[-1]:.4f}\n")

        f.write("\n=====系统信息=====\n")
        try:
            import platform
            f.write(f"操作系统: {platform.system()} {platform.release()}\n")
            f.write(f"处理器: {platform.processor()}\n")
            f.write(f"Python版本: {platform.python_version()}\n")
            f.write(f"NumPy版本: {np.__version__}\n")
            f.write(f"Pandas版本: {pd.__version__}\n")
        except:
            f.write("系统信息获取失败\n")


def save_predictions(df: pd.DataFrame, output_path: Path) -> None:
    with output_path.open('w', encoding='utf-8') as f:
        grouped = df.groupby('user')
        for user, group in grouped:
            f.write(f"{user}|{len(group)}\n")
            for _, row in group.iterrows():
                f.write(f"{row['item']}\t{row['prediction']:.4f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="基于矩阵分解+神经网络的推荐系统")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--output", type=str, default="Predictions.txt")
    parser.add_argument("--stats", type=str, default="TrainingStats.txt")
    parser.add_argument("--factors", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--reg", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=100.0)
    parser.add_argument("--min_rating", type=float, default=0.0)
    parser.add_argument("--max_rating", type=float, default=100.0)
    args = parser.parse_args()

    print(f"正在加载训练数据: {args.train}")
    train_df = parse_data(args.train, has_ratings=True)
    print(f"训练数据加载完成，共 {len(train_df)} 条记录，评分范围 {train_df['rating'].min():.1f}-{train_df['rating'].max():.1f}")

    print(f"正在加载测试数据: {args.test}")
    test_df = parse_data(args.test, has_ratings=False)
    print(f"测试数据加载完成，共 {len(test_df)} 条记录需要预测")

    train_part, val_part = train_test_split(train_df, test_size=0.2, random_state=42)
    print(f"划分训练集 {len(train_part)} 条，验证集 {len(val_part)} 条")

    try:
        _ = psutil.Process()
        print("内存监控: 已启用(psutil 已安装)")
    except:
        print("内存监控: 未启用(请安装 psutil)")

    model = NeuralMF(
        n_factors=args.factors,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        reg=args.reg,
        n_epochs=args.epochs,
        grad_clip=args.grad_clip,
        rating_scale=(args.min_rating, args.max_rating),
        verbose=True
    )
    model.fit(train_part, val_data=val_part)
    print("模型训练完成")

    full_train_rmse = model._evaluate(train_df)
    print(f"完整训练集 RMSE: {full_train_rmse:.4f}")

    print("开始生成预测结果...")
    preds = model.predict(test_df)
    test_df['prediction'] = preds
    save_predictions(test_df, Path(args.output))
    print(f"预测结果已保存至 {args.output}")

    print(f"开始保存训练统计信息至 {args.stats}")
    save_training_stats(Path(args.stats), model, args)
    print(f"统计信息已保存至 {args.stats}")

    print("===== 训练摘要 =====")
    print(f"总训练时间: {model.training_time:.2f} 秒")
    print(f"平均每轮时间: {model.training_time/args.epochs:.2f} 秒")
    print(f"内存消耗: {model.mem_usage:.2f} MB")
    if model.rmse_history:
        print(f"最终 Train RMSE: {model.rmse_history[-1]:.4f}, 最小 Train RMSE: {min(model.rmse_history):.4f}")
    if model.val_rmse_history:
        print(f"最终 Val RMSE: {model.val_rmse_history[-1]:.4f}, 最小 Val RMSE: {min(model.val_rmse_history):.4f}")

    print("任务完成！")

if __name__ == '__main__':
    main()
