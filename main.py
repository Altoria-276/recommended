"""基于矩阵分解的评分预测系统

该模块实现了带偏置的矩阵分解算法(BiasSVD)，
用于预测用户对物品的评分。支持自定义评分范围，
并提供了详细的训练统计信息输出。

主要功能:
1. 解析特殊格式的训练和测试数据
2. 使用BiasSVD模型进行评分预测
3. 生成预测结果文件
4. 输出训练统计信息(时间、内存、RMSE等)

使用示例:
    python 推荐系统.py --train data/Train.txt --test data/Test.txt
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import re
import math
import time
import os
import psutil
from tqdm import tqdm, trange
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional, Union, Any, DefaultDict


def parse_data(file_path: Union[str, Path], has_ratings: bool = True) -> pd.DataFrame:
    """解析特殊格式的数据文件

    支持格式:
        <user id>|<numbers of rating items>
        <item id>   <score>

    Args:
        file_path: 文件路径
        has_ratings: 文件是否包含评分数据

    Returns:
        解析后的数据框，包含'user'和'item'列，
        如果has_ratings为True则包含'rating'列

    Raises:
        FileNotFoundError: 如果文件不存在
    """
    users: List[str] = []
    items: List[str] = []
    ratings: Optional[List[float]] = [] if has_ratings else None

    with open(file_path, "r") as f:
        lines: List[str] = f.readlines()
        i: int = 0
        while i < len(lines):
            # 解析用户行（格式：<user id>|<number of ratings>）
            if "|" in lines[i]:
                parts: List[str] = lines[i].strip().split("|")
                user_id: str = parts[0]
                num_ratings: int = int(parts[1])

                # 解析后续的评分行
                for j in range(1, num_ratings + 1):
                    if i + j >= len(lines):
                        break

                    # 解析物品和评分（使用正则处理可能的多空格）
                    item_parts: List[str] = re.split(r"\s+", lines[i + j].strip())
                    item_id: str = item_parts[0]

                    if has_ratings and ratings is not None:
                        rating: float = float(item_parts[1]) if len(item_parts) > 1 else 0.0
                        ratings.append(rating)

                    users.append(user_id)
                    items.append(item_id)

                i += num_ratings + 1
            else:
                i += 1

    if has_ratings and ratings is not None:
        return pd.DataFrame({"user": users, "item": items, "rating": ratings})
    else:
        return pd.DataFrame({"user": users, "item": items})


class BiasSVD:
    """带偏置的矩阵分解模型（支持自定义评分范围）

    实现基于矩阵分解的评分预测算法，包含偏置项和正则化。
    支持自定义评分范围，并包含梯度裁剪防止数值溢出。

    Attributes:
        n_factors: 隐因子维度
        lr: 学习率
        reg: 正则化系数
        n_epochs: 训练轮数
        verbose: 是否显示训练进度
        grad_clip: 梯度裁剪阈值
        rating_scale: 评分范围(min, max)
        global_mean: 全局平均评分
        user_biases: 用户偏置向量
        item_biases: 物品偏置向量
        user_factors: 用户隐因子矩阵
        item_factors: 物品隐因子矩阵
        rmse_history: 每个epoch的RMSE记录
        training_time: 总训练时间(秒)
        mem_usage: 内存消耗(MB)
    """

    def __init__(
        self,
        n_factors: int = 20,
        lr: float = 0.0005,
        reg: float = 0.1,
        n_epochs: int = 50,
        verbose: bool = True,
        grad_clip: float = 100.0,
        rating_scale: Tuple[float, float] = (0, 100),
    ) -> None:
        """初始化BiasSVD模型

        Args:
            n_factors: 隐因子维度，默认20
            lr: 学习率，默认0.0005
            reg: 正则化系数，默认0.1
            n_epochs: 训练轮数，默认50
            verbose: 是否显示训练进度，默认True
            grad_clip: 梯度裁剪阈值，默认100.0
            rating_scale: 评分范围(min, max)，默认(0, 100)
        """
        self.n_factors: int = n_factors
        self.lr: float = lr
        self.reg: float = reg
        self.n_epochs: int = n_epochs
        self.verbose: bool = verbose
        self.grad_clip: float = grad_clip
        self.rating_scale: Tuple[float, float] = rating_scale

        # 初始化模型参数
        self.global_mean: float = 0.0
        self.user_biases: np.ndarray = np.array([])
        self.item_biases: np.ndarray = np.array([])
        self.user_factors: np.ndarray = np.array([])
        self.item_factors: np.ndarray = np.array([])

        # 映射字典
        self.user_map: Dict[str, int] = {}
        self.item_map: Dict[str, int] = {}
        self.inv_user_map: Dict[int, str] = {}
        self.inv_item_map: Dict[int, str] = {}

        # 训练统计
        self.rmse_history: List[float] = []
        self.training_time: float = 0.0
        self.mem_usage: float = 0.0

    def fit(self, train_data: pd.DataFrame) -> List[float]:
        """训练BiasSVD模型

        使用随机梯度下降(SGD)优化模型参数，包含梯度裁剪和进度显示。

        Args:
            train_data: 训练数据，包含'user','item','rating'列

        Returns:
            每个epoch的RMSE历史记录

        Raises:
            ValueError: 如果训练数据格式不正确
        """
        # 记录开始时间和初始内存
        start_time: float = time.time()
        start_mem: float = self.get_process_memory()

        # 创建用户和物品的映射
        self.user_map = {u: i for i, u in enumerate(train_data["user"].unique())}
        self.item_map = {i: idx for idx, i in enumerate(train_data["item"].unique())}
        self.inv_user_map = {i: u for u, i in self.user_map.items()}
        self.inv_item_map = {i: u for u, i in self.item_map.items()}

        n_users: int = len(self.user_map)
        n_items: int = len(self.item_map)

        # 初始化模型参数
        self.global_mean = train_data["rating"].mean()
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)

        # 使用更小的初始化标准差
        init_std: float = 0.01
        self.user_factors = np.random.normal(0, init_std, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, init_std, (n_items, self.n_factors))

        # 转换为内部ID
        train_data = train_data.copy()
        train_data["user_id"] = train_data["user"].map(self.user_map)
        train_data["item_id"] = train_data["item"].map(self.item_map)

        # 创建外层进度条
        epoch_bar = trange(self.n_epochs, desc="训练进度", disable=not self.verbose, position=0)

        # 训练模型
        for epoch in epoch_bar:
            epoch_start_time: float = time.time()
            total_loss: float = 0.0
            nan_count: int = 0  # 跟踪NaN值出现次数
            valid_samples: int = 0

            # 打乱数据顺序
            shuffled_data: pd.DataFrame = train_data.sample(frac=1).reset_index(drop=True)

            # 创建内层进度条
            inner_bar = tqdm(
                shuffled_data.iterrows(),
                total=len(shuffled_data),
                desc=f"第 {epoch+1} 轮",
                disable=not self.verbose or len(shuffled_data) > 10000,
                position=1,
                leave=False,
            )

            for _, row in inner_bar:
                u: int = int(row["user_id"])
                i: int = int(row["item_id"])
                r: float = row["rating"]

                # 计算预测值和误差
                pred: float = self._predict_pair(u, i)
                e: float = r - pred

                # 检查数值稳定性
                if math.isnan(e) or math.isinf(e):
                    nan_count += 1
                    continue

                total_loss += e**2
                valid_samples += 1

                # 更新偏置 - 添加梯度裁剪
                user_bias_grad: float = e - self.reg * self.user_biases[u]
                item_bias_grad: float = e - self.reg * self.item_biases[i]

                # 梯度裁剪
                user_bias_grad = np.clip(user_bias_grad, -self.grad_clip, self.grad_clip)
                item_bias_grad = np.clip(item_bias_grad, -self.grad_clip, self.grad_clip)

                self.user_biases[u] += self.lr * user_bias_grad
                self.item_biases[i] += self.lr * item_bias_grad

                # 更新隐向量 - 添加梯度裁剪
                u_factor: np.ndarray = self.user_factors[u]
                i_factor: np.ndarray = self.item_factors[i]

                # 计算梯度
                user_grad: np.ndarray = e * i_factor - self.reg * u_factor
                item_grad: np.ndarray = e * u_factor - self.reg * i_factor

                # 梯度裁剪
                user_grad = np.clip(user_grad, -self.grad_clip, self.grad_clip)
                item_grad = np.clip(item_grad, -self.grad_clip, self.grad_clip)

                self.user_factors[u] += self.lr * user_grad
                self.item_factors[i] += self.lr * item_grad

            # 关闭内层进度条
            inner_bar.close()

            # 计算并更新RMSE显示在外层进度条
            if valid_samples > 0:
                rmse: float = np.sqrt(total_loss / valid_samples)
                self.rmse_history.append(rmse)
                epoch_time: float = time.time() - epoch_start_time
                epoch_bar.set_postfix({"RMSE": f"{rmse:.2f}", "无效样本": nan_count, "时间": f"{epoch_time:.1f}s"})
            else:
                self.rmse_history.append(float("nan"))
                epoch_bar.set_postfix({"RMSE": "NaN", "无效样本": nan_count, "时间": f"{time.time() - epoch_start_time:.1f}s"})
                print(f"\n警告: 第 {epoch+1}/{self.n_epochs} 轮训练所有样本无效! 请调整学习率或正则化参数")

        # 记录总训练时间
        self.training_time = time.time() - start_time

        # 记录最终内存使用
        self.end_mem: float = self.get_process_memory()
        self.mem_usage = self.end_mem - start_mem

        return self.rmse_history

    def get_process_memory(self) -> float:
        """获取当前进程的内存使用量

        使用psutil获取当前进程的常驻内存集(RSS)大小。

        Returns:
            内存使用量(MB)，如果psutil不可用则返回0.0
        """
        try:
            process: psutil.Process = psutil.Process(os.getpid())
            mem_info: Any = process.memory_info()
            return mem_info.rss / (1024**2)  # 转换为MB
        except:
            return 0.0

    def _predict_pair(self, u_idx: int, i_idx: int) -> float:
        """使用内部ID进行评分预测

        内部方法，用于预测指定用户和物品的评分。

        Args:
            u_idx: 用户内部ID
            i_idx: 物品内部ID

        Returns:
            预测评分值
        """
        prediction: float = (
            self.global_mean
            + self.user_biases[u_idx]
            + self.item_biases[i_idx]
            + np.dot(self.user_factors[u_idx], self.item_factors[i_idx])
        )
        return prediction

    def predict(self, test_data: pd.DataFrame) -> List[float]:
        """预测测试集评分

        为测试集中的每个用户-物品对生成预测评分，
        自动处理冷启动问题（新用户/新物品）。

        Args:
            test_data: 测试数据，包含'user'和'item'列

        Returns:
            预测评分列表，顺序与test_data相同
        """
        predictions: List[float] = []

        # 创建预测进度条
        pred_bar = tqdm(test_data.iterrows(), total=len(test_data), desc="生成预测", position=0)

        for _, row in pred_bar:
            user: str = row["user"]
            item: str = row["item"]

            # 处理冷启动情况
            if user in self.user_map and item in self.item_map:
                u_idx: int = self.user_map[user]
                i_idx: int = self.item_map[item]
                pred: float = self._predict_pair(u_idx, i_idx)
            elif user in self.user_map:
                # 新物品：使用用户平均分
                u_idx = self.user_map[user]
                pred = self.global_mean + self.user_biases[u_idx]
            elif item in self.item_map:
                # 新用户：使用物品平均分
                i_idx = self.item_map[item]
                pred = self.global_mean + self.item_biases[i_idx]
            else:
                # 全新用户和物品：使用全局平均分
                pred = self.global_mean

            # 确保评分在合理范围内
            min_score, max_score = self.rating_scale
            predictions.append(max(min_score, min(max_score, pred)))

        # 关闭预测进度条
        pred_bar.close()
        return predictions


def save_training_stats(stats_path: Path, model: BiasSVD, args: argparse.Namespace) -> None:
    """保存训练统计信息到文件

    生成包含模型参数、性能指标和系统信息的详细报告。

    Args:
        stats_path: 统计文件路径
        model: 训练好的模型实例
        args: 命令行参数
    """
    with stats_path.open("w", encoding="utf-8") as f:
        f.write("===== 训练统计信息 =====\n\n")
        f.write(f"模型: BiasSVD\n")
        f.write(f"评分范围: {model.rating_scale[0]}-{model.rating_scale[1]}\n")
        f.write(f"训练文件: {args.train}\n")
        f.write(f"测试文件: {args.test}\n")
        f.write(f"输出文件: {args.output}\n")
        f.write(f"统计文件: {stats_path}\n\n")

        f.write("===== 模型参数 =====\n")
        f.write(f"隐因子维度: {args.factors}\n")
        f.write(f"训练轮数: {args.epochs}\n")
        f.write(f"学习率: {args.lr}\n")
        f.write(f"正则化系数: {args.reg}\n")
        f.write(f"梯度裁剪阈值: {args.grad_clip}\n")
        f.write(f"最小评分: {args.min_rating}\n")
        f.write(f"最大评分: {args.max_rating}\n\n")

        f.write("===== 性能指标 =====\n")
        f.write(f"总训练时间: {model.training_time:.2f} 秒\n")
        f.write(f"平均每轮时间: {model.training_time/args.epochs:.2f} 秒\n")
        f.write(f"内存消耗: {model.mem_usage:.2f} MB\n")

        if model.rmse_history:
            final_rmse: float = model.rmse_history[-1]
            min_rmse: float = min(model.rmse_history)
            f.write(f"最终RMSE: {final_rmse:.2f}\n")
            f.write(f"最小RMSE: {min_rmse:.2f}\n")
        else:
            f.write("RMSE: 无有效数据\n")

        f.write("\n===== RMSE历史记录 =====\n")
        if model.rmse_history:
            for i, rmse in enumerate(model.rmse_history, 1):
                f.write(f"Epoch {i:2d}: {rmse:.2f}\n")
        else:
            f.write("无有效RMSE记录\n")

        f.write("\n===== 系统信息 =====\n")
        try:
            import platform

            f.write(f"操作系统: {platform.system()} {platform.release()}\n")
            f.write(f"处理器: {platform.processor()}\n")
            f.write(f"Python版本: {platform.python_version()}\n")
            f.write(f"NumPy版本: {np.__version__}\n")
            f.write(f"Pandas版本: {pd.__version__}\n")
        except:
            f.write("系统信息获取失败\n")


def main() -> None:
    """主程序入口

    解析命令行参数，执行训练和预测流程。
    """
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="基于矩阵分解的评分预测系统")
    parser.add_argument("--train", type=str, required=True, help="训练数据文件路径 (Train.txt)")
    parser.add_argument("--test", type=str, required=True, help="测试数据文件路径 (Test.txt)")
    parser.add_argument("--output", type=str, default="Predictions.txt", help="预测结果输出路径 (默认: Predictions.txt)")
    parser.add_argument("--stats", type=str, default="TrainingStats.txt", help="统计信息输出路径 (默认: TrainingStats.txt)")
    parser.add_argument("--factors", type=int, default=20, help="隐因子维度 (默认: 20)")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数 (默认: 50)")
    parser.add_argument("--lr", type=float, default=0.0005, help="学习率 (默认: 0.0005)")
    parser.add_argument("--reg", type=float, default=0.1, help="正则化系数 (默认: 0.1)")
    parser.add_argument("--grad_clip", type=float, default=100.0, help="梯度裁剪阈值 (默认: 100.0)")
    parser.add_argument("--min_rating", type=float, default=0.0, help="最小评分值 (默认: 0.0)")
    parser.add_argument("--max_rating", type=float, default=100.0, help="最大评分值 (默认: 100.0)")

    args: argparse.Namespace = parser.parse_args()

    # 使用 pathlib 处理路径
    train_path: Path = Path(args.train)
    test_path: Path = Path(args.test)
    output_path: Path = Path(args.output)
    stats_path: Path = Path(args.stats)

    # 验证文件路径
    if not train_path.exists():
        print(f"错误: 训练文件不存在 - {train_path}")
        exit(1)

    if not test_path.exists():
        print(f"错误: 测试文件不存在 - {test_path}")
        exit(1)

    # 1. 读取并解析数据
    print(f"正在加载训练数据: {train_path}")
    train_df: pd.DataFrame = parse_data(train_path, has_ratings=True)
    print(f"训练数据加载完成，共 {len(train_df)} 条评分记录")

    # 验证评分范围
    actual_min: float = train_df["rating"].min()
    actual_max: float = train_df["rating"].max()
    print(f"评分范围: {actual_min:.1f} - {actual_max:.1f}")

    print(f"\n正在加载测试数据: {test_path}")
    test_df: pd.DataFrame = parse_data(test_path, has_ratings=False)
    print(f"测试数据加载完成，共需预测 {len(test_df)} 个评分")

    # 2. 训练模型
    print("\n开始训练BiasSVD模型...")
    print(f"模型参数: 隐因子={args.factors}, 学习率={args.lr}, 正则化={args.reg}, 轮数={args.epochs}, 梯度裁剪={args.grad_clip}")
    print(f"评分范围: {args.min_rating} - {args.max_rating}")

    # 检查psutil是否可用
    try:
        import psutil

        print("内存监控: 已启用 (psutil 已安装)")
    except ImportError:
        print("内存监控: 未启用 (请安装 psutil: pip install psutil)")

    model: BiasSVD = BiasSVD(
        n_factors=args.factors,
        lr=args.lr,
        reg=args.reg,
        n_epochs=args.epochs,
        verbose=True,
        grad_clip=args.grad_clip,
        rating_scale=(args.min_rating, args.max_rating),
    )
    model.fit(train_df)
    print("\n模型训练完成！")

    # 3. 进行预测
    print("\n开始生成预测结果...")
    predictions: List[float] = model.predict(test_df)
    test_df["prediction"] = predictions

    # 分析预测结果分布
    pred_min: float = test_df["prediction"].min()
    pred_max: float = test_df["prediction"].max()
    pred_mean: float = test_df["prediction"].mean()
    print(f"\n预测结果分析: 最小值={pred_min:.1f}, 最大值={pred_max:.1f}, 平均值={pred_mean:.1f}")

    # 4. 保存预测结果
    print(f"\n正在保存预测结果到: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 按用户分组输出
    with output_path.open("w", encoding="utf-8") as f:
        grouped = test_df.groupby("user")
        for user, group in grouped:
            num_items: int = len(group)
            f.write(f"{user}|{num_items}\n")
            for _, row in group.iterrows():
                f.write(f"{row['item']}\t{row['prediction']:.1f}\n")

    print(f"预测结果已成功保存至 {output_path}")

    # 5. 保存统计信息
    print(f"\n正在保存训练统计信息到: {stats_path}")
    save_training_stats(stats_path, model, args)
    print(f"统计信息已成功保存至 {stats_path}")

    # 6. 控制台显示关键统计信息
    print("\n===== 训练摘要 =====")
    print(f"总训练时间: {model.training_time:.2f} 秒")
    print(f"平均每轮时间: {model.training_time/args.epochs:.2f} 秒")
    if hasattr(model, "mem_usage"):
        print(f"内存消耗: {model.mem_usage:.2f} MB")
    if model.rmse_history:
        final_rmse: float = model.rmse_history[-1]
        min_rmse: float = min(model.rmse_history)
        print(f"最终RMSE: {final_rmse:.2f}")
        print(f"最小RMSE: {min_rmse:.2f}")

    print("\n任务完成！")


if __name__ == "__main__":
    main()
