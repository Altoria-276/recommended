import numpy as np
import pandas as pd
from collections import defaultdict
import re
import sys
import math
from tqdm import tqdm, trange
from pathlib import Path


def parse_data(file_path, has_ratings=True):
    """解析特殊格式的数据文件"""
    users = []
    items = []
    ratings = [] if has_ratings else None

    with open(file_path, "r") as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            # 解析用户行（格式：<user id>|<number of ratings>）
            if "|" in lines[i]:
                user_id, num_ratings = lines[i].strip().split("|")
                num_ratings = int(num_ratings)

                # 解析后续的评分行
                for j in range(1, num_ratings + 1):
                    if i + j >= len(lines):
                        break

                    # 解析物品和评分（使用正则处理可能的多空格）
                    parts = re.split(r"\s+", lines[i + j].strip())
                    item_id = parts[0]

                    if has_ratings:
                        rating = float(parts[1]) if len(parts) > 1 else None
                        ratings.append(rating)

                    users.append(user_id)
                    items.append(item_id)

                i += num_ratings + 1
            else:
                i += 1

    if has_ratings:
        return pd.DataFrame({"user": users, "item": items, "rating": ratings})
    else:
        return pd.DataFrame({"user": users, "item": items})


class BiasSVD:
    """带偏置的矩阵分解模型"""

    def __init__(self, n_factors=20, lr=0.005, reg=0.02, n_epochs=50, verbose=True, grad_clip=5.0):
        self.n_factors = n_factors  # 隐因子维度
        self.lr = lr  # 学习率
        self.reg = reg  # 正则化系数
        self.n_epochs = n_epochs  # 迭代次数
        self.verbose = verbose  # 是否打印训练过程
        self.grad_clip = grad_clip  # 梯度裁剪阈值
        self.global_mean = 0  # 全局平均评分
        self.user_biases = None  # 用户偏置
        self.item_biases = None  # 物品偏置
        self.user_factors = None  # 用户隐向量
        self.item_factors = None  # 物品隐向量

    def fit(self, train_data):
        # 创建用户和物品的映射
        self.user_map = {u: i for i, u in enumerate(train_data["user"].unique())}
        self.item_map = {i: idx for idx, i in enumerate(train_data["item"].unique())}
        self.inv_user_map = {i: u for u, i in self.user_map.items()}
        self.inv_item_map = {i: u for u, i in self.item_map.items()}

        n_users = len(self.user_map)
        n_items = len(self.item_map)

        # 初始化模型参数
        self.global_mean = train_data["rating"].mean()
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # 转换为内部ID
        train_data["user_id"] = train_data["user"].map(self.user_map)
        train_data["item_id"] = train_data["item"].map(self.item_map)

        # 创建外层进度条
        epoch_bar = trange(self.n_epochs, desc="训练进度", disable=not self.verbose, position=0)  # 确保外层进度条在位置0

        # 训练模型
        for epoch in epoch_bar:
            total_loss = 0
            nan_count = 0  # 跟踪NaN值出现次数

            # 打乱数据顺序
            shuffled_data = train_data.sample(frac=1).reset_index(drop=True)

            # 创建内层进度条
            inner_bar = tqdm(
                shuffled_data.iterrows(),
                total=len(shuffled_data),
                desc=f"第 {epoch+1} 轮",
                disable=not self.verbose or len(shuffled_data) > 10000,
                position=1,  # 内层进度条在位置1
                leave=False,  # 内层完成后不留痕迹
            )

            for _, row in inner_bar:
                u = int(row["user_id"])
                i = int(row["item_id"])
                r = row["rating"]

                # 计算预测值和误差
                pred = self._predict_pair(u, i)
                e = r - pred

                # 检查数值稳定性
                if math.isnan(e) or math.isinf(e):
                    nan_count += 1
                    continue

                total_loss += e**2

                # 更新偏置 - 添加梯度裁剪
                user_bias_grad = e - self.reg * self.user_biases[u]
                item_bias_grad = e - self.reg * self.item_biases[i]

                # 梯度裁剪
                user_bias_grad = np.clip(user_bias_grad, -self.grad_clip, self.grad_clip)
                item_bias_grad = np.clip(item_bias_grad, -self.grad_clip, self.grad_clip)

                self.user_biases[u] += self.lr * user_bias_grad
                self.item_biases[i] += self.lr * item_bias_grad

                # 更新隐向量 - 添加梯度裁剪
                u_factor = self.user_factors[u]
                i_factor = self.item_factors[i]

                # 计算梯度
                user_grad = e * i_factor - self.reg * u_factor
                item_grad = e * u_factor - self.reg * i_factor

                # 梯度裁剪
                user_grad = np.clip(user_grad, -self.grad_clip, self.grad_clip)
                item_grad = np.clip(item_grad, -self.grad_clip, self.grad_clip)

                self.user_factors[u] += self.lr * user_grad
                self.item_factors[i] += self.lr * item_grad

            # 关闭内层进度条
            inner_bar.close()

            # 计算并更新RMSE显示在外层进度条
            if len(shuffled_data) - nan_count > 0:
                rmse = np.sqrt(total_loss / (len(shuffled_data) - nan_count))
                epoch_bar.set_postfix({"RMSE": f"{rmse:.4f}", "无效样本": nan_count})
            else:
                epoch_bar.set_postfix({"RMSE": "NaN", "无效样本": nan_count})
                print(f"\n警告: 第 {epoch+1}/{self.n_epochs} 轮训练所有样本无效! 请调整学习率或正则化参数")

    def _predict_pair(self, u_idx, i_idx):
        """使用内部ID进行预测"""
        prediction = self.global_mean + self.user_biases[u_idx] + self.item_biases[i_idx]

        # 添加点积前检查数值范围
        user_factor = self.user_factors[u_idx]
        item_factor = self.item_factors[i_idx]

        # 检查NaN值
        if not (np.isnan(user_factor).any() or np.isnan(item_factor).any()):
            dot_product = np.dot(user_factor, item_factor)
            # 确保点积不会导致数值溢出
            if not (math.isnan(dot_product) or math.isinf(dot_product)):
                prediction += dot_product

        return prediction

    def predict(self, test_data):
        """预测测试集评分"""
        predictions = []

        # 创建预测进度条
        pred_bar = tqdm(test_data.iterrows(), total=len(test_data), desc="生成预测", position=0)  # 预测进度条在位置0

        for _, row in pred_bar:
            user = row["user"]
            item = row["item"]

            # 处理冷启动情况
            if user in self.user_map and item in self.item_map:
                u_idx = self.user_map[user]
                i_idx = self.item_map[item]
                pred = self._predict_pair(u_idx, i_idx)
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

            # 确保评分在合理范围内（假设1-5分）
            predictions.append(max(1.0, min(5.0, pred)))

        # 关闭预测进度条
        pred_bar.close()
        return predictions


# 主程序
if __name__ == "__main__":
    import argparse

    # 设置命令行参数
    parser = argparse.ArgumentParser(description="基于矩阵分解的评分预测系统")
    parser.add_argument("--train", type=str, required=True, help="训练数据文件路径 (Train.txt)")
    parser.add_argument("--test", type=str, required=True, help="测试数据文件路径 (Test.txt)")
    parser.add_argument("--output", type=str, default="Predictions.txt", help="预测结果输出路径 (默认: Predictions.txt)")
    parser.add_argument("--factors", type=int, default=20, help="隐因子维度 (默认: 20)")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数 (默认: 50)")
    parser.add_argument("--lr", type=float, default=0.005, help="学习率 (默认: 0.005)")  # 降低了默认学习率
    parser.add_argument("--reg", type=float, default=0.05, help="正则化系数 (默认: 0.05)")  # 增加了默认正则化
    parser.add_argument("--grad_clip", type=float, default=5.0, help="梯度裁剪阈值 (默认: 5.0)")

    args = parser.parse_args()

    # 使用 pathlib 处理路径
    train_path = Path(args.train)
    test_path = Path(args.test)
    output_path = Path(args.output)

    # 验证文件路径
    if not train_path.exists():
        print(f"错误: 训练文件不存在 - {train_path}")
        exit(1)

    if not test_path.exists():
        print(f"错误: 测试文件不存在 - {test_path}")
        exit(1)

    # 1. 读取并解析数据
    print(f"正在加载训练数据: {train_path}")
    train_df = parse_data(train_path, has_ratings=True)
    print(f"训练数据加载完成，共 {len(train_df)} 条评分记录")

    print(f"\n正在加载测试数据: {test_path}")
    test_df = parse_data(test_path, has_ratings=False)
    print(f"测试数据加载完成，共需预测 {len(test_df)} 个评分")

    # 2. 训练模型
    print("\n开始训练BiasSVD模型...")
    print(f"模型参数: 隐因子={args.factors}, 学习率={args.lr}, 正则化={args.reg}, 轮数={args.epochs}, 梯度裁剪={args.grad_clip}")

    model = BiasSVD(n_factors=args.factors, lr=args.lr, reg=args.reg, n_epochs=args.epochs, verbose=True, grad_clip=args.grad_clip)
    model.fit(train_df)
    print("\n模型训练完成！")

    # 3. 进行预测
    print("\n开始生成预测结果...")
    predictions = model.predict(test_df)
    test_df["prediction"] = predictions

    # 4. 保存结果（保持原始格式）
    print(f"\n正在保存预测结果到: {output_path}")
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 按用户分组输出
    with output_path.open("w", encoding="utf-8") as f:
        # 按用户分组
        grouped = test_df.groupby("user")
        for user, group in grouped:
            num_items = len(group)
            f.write(f"{user}|{num_items}\n")
            for _, row in group.iterrows():
                f.write(f"{row['item']}\t{row['prediction']:.4f}\n")

    print(f"预测结果已成功保存至 {output_path}")
    print("\n任务完成！")
