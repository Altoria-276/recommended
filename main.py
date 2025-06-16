# main.py
import numpy as np
import pandas as pd
import re
import math
import time
import os
import argparse
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from pathlib import Path

# 导入模型
from models import BiasSVD, BaseModel, NeuralMF, LightGCN, SVDpp, BPRMF, NGCF, NeuMF, VAE_CF # , SVDppp # 导入解耦后的模型

# 导入监控工具
from moniter import monitor_function  # 假设监控代码保存在 monitor_util.py


def parse_data(file_path: Union[str, Path], has_ratings: bool = True) -> pd.DataFrame:
    users: List[str] = []
    items: List[str] = []
    ratings: Optional[List[float]] = [] if has_ratings else None

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if "|" in lines[i]:
            user_id, cnt = lines[i].strip().split("|")
            cnt = int(cnt)
            for j in range(1, cnt + 1):
                if i + j >= len(lines):
                    break
                parts = re.split(r"\s+", lines[i + j].strip())
                item_id = parts[0]
                users.append(user_id)
                items.append(item_id)
                if has_ratings and ratings is not None:
                    ratings.append(float(parts[1]) if len(parts) > 1 else 0.0)
            i += cnt + 1
        else:
            i += 1
    df = pd.DataFrame({"user": users, "item": items})
    if has_ratings and ratings is not None:
        df["rating"] = ratings  # type: ignore
    return df


def save_training_stats(
    stats_path: Path,
    model: BaseModel,
    args: argparse.Namespace,
    max_memory: float,
    training_time: float,
) -> None:
    """保存训练统计信息到文件

    参数:
        stats_path: 统计文件路径
        model: 训练好的模型
        args: 命令行参数
        max_memory: 峰值内存使用量(MB)
        training_time: 总训练时间(秒)
    """
    with stats_path.open("w", encoding="utf-8") as f:
        f.write("===== 训练统计信息 =====\n\n")
        f.write(f"模型: {model.model_name}\n")  # 使用模型的名称属性
        f.write(f"训练文件: {args.train}\n")
        f.write(f"测试文件: {args.test}\n")
        f.write(f"输出文件: {args.output}\n")
        f.write(f"统计文件: {stats_path}\n\n")

        f.write("===== 模型参数 =====\n")
        f.write(f"隐因子维度: {args.factors}\n")
        f.write(f"训练轮数: {args.epochs}\n")
        f.write(f"学习率: {args.lr}\n")
        f.write(f"正则化系数: {args.reg}\n")
        f.write(f"梯度剪裁阈值: {args.grad_clip}\n")
        f.write(f"最小评分: {args.min_rating}\n")
        f.write(f"最大评分: {args.max_rating}\n\n")

        f.write("===== 性能指标 =====\n")
        f.write(f"总训练时间: {training_time:.2f} 秒\n")
        f.write(f"平均每轮时间: {training_time/args.epochs:.2f} 秒\n")
        f.write(f"峰值内存使用: {max_memory:.2f} MB\n")

        if model.rmse_history:
            f.write(f"最终 Train RMSE: {model.rmse_history[-1]:.4f}\n")
            f.write(f"最小 Train RMSE: {min(model.rmse_history):.4f}\n")
        if model.val_rmse_history:
            f.write(f"最终 Val RMSE: {model.val_rmse_history[-1]:.4f}\n")
            f.write(f"最小 Val RMSE: {min(model.val_rmse_history):.4f}\n")
        if model.test_rmse_history:
            f.write(f"最终 Test RMSE: {model.test_rmse_history[-1]:.4f}\n")
            f.write(f"最小 Test RMSE: {min(model.test_rmse_history):.4f}\n")

        f.write("\n===== RMSE 历史记录 =====\n")
        f.write("Epoch,TrainRMSE,ValRMSE,TestRMSE\n")
        for i in range(len(model.rmse_history)):
            trm = model.rmse_history[i]
            vrm = model.val_rmse_history[i] if i < len(model.val_rmse_history) else float("nan")
            trasm = model.test_rmse_history[i] if i < len(model.test_rmse_history) else float("nan")
            f.write(f"{i+1},{trm:.4f},{vrm:.4f},{trasm:.4f}\n")

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


def save_predictions(df: pd.DataFrame, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        grouped = df.groupby("user")
        for user, group in grouped:
            f.write(f"{user}|{len(group)}\n")
            for _, row in group.iterrows():
                f.write(f"{row['item']}\t{row['prediction']:.4f}\n")


def train_and_predict(args: argparse.Namespace) -> BaseModel:
    """执行训练和预测的核心函数"""
    # 加载数据
    print(f"正在加载训练数据: {args.train}")
    train_df = parse_data(args.train, has_ratings=True)
    print(f"训练数据加载完成，共 {len(train_df)} 条记录，评分范围 {train_df['rating'].min():.1f}-{train_df['rating'].max():.1f}")

    print(f"正在加载测试数据: {args.test}")
    test_df = parse_data(args.test, has_ratings=False)
    print(f"测试数据加载完成，共 {len(test_df)} 条记录需要预测")

    # 根据参数选择模型
    if args.model == "BiasSVD":
        ModelClass = BiasSVD 
    elif args.model == 'NeuralMF': 
        ModelClass = NeuralMF 
    elif args.model == 'LightGCN': 
        ModelClass = LightGCN 
    elif args.model == 'SVDpp': 
        ModelClass = SVDpp 
    elif args.model == 'SVDppp': 
        ModelClass = SVDppp 
    elif args.model == 'BPRMF': 
        ModelClass = BPRMF 
    elif args.model == 'NGCF': 
        ModelClass = NGCF 
    elif args.model == 'NeuMF': 
        ModelClass = NeuMF 
    elif args.model == 'VAE_CF': 
        ModelClass = VAE_CF 
    else:
        raise ValueError(f"未知模型类型: {args.model}")

    # 划分训练/验证集
    print("\n===== 划分训练集和验证集 =====")
    train_part, val_part = train_test_split(train_df, test_size=0.2, random_state=42)
    print(f"划分训练集 {len(train_part)} 条，验证集 {len(val_part)} 条")

    # 初始化模型
    model = ModelClass(
        n_factors=args.factors,
        lr=args.lr,
        reg=args.reg,
        n_epochs=args.epochs,
        grad_clip=args.grad_clip,
        rating_scale=(args.min_rating, args.max_rating),
        verbose=args.verbose,
    )

    # 训练模型
    print("开始训练模型...")
    model.fit(train_part, val_data=val_part)
    print("模型训练完成")

    # 完整训练集 RMSE
    full_train_rmse = model._evaluate(train_df)
    print(f"完整训练集 RMSE: {full_train_rmse:.4f}")

    # 预测并保存结果
    print("开始生成预测结果...")
    preds = model.predict(test_df)
    test_df["prediction"] = preds
    save_predictions(test_df, Path(args.output))
    print(f"预测结果已保存至 {args.output}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="基于矩阵分解的评分预测系统")
    # 数据参数
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--output", type=str, default="./results/predictions.txt")
    parser.add_argument("--stats", type=str, default="./results/training_stats.txt")

    # 模型选择
    parser.add_argument("--model", type=str, default="bias_svd", help="选择使用的模型 (默认: bias_svd)")

    # 模型参数
    parser.add_argument("--factors", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--reg", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=100.0)
    parser.add_argument("--min_rating", type=float, default=0.0)
    parser.add_argument("--max_rating", type=float, default=100.0)

    # 训练参数
    parser.add_argument("--monitor", action="store_true", help="启用内存和时间监控")
    parser.add_argument("--verbose", type=bool, default=True, help="显示训练进度")

    args = parser.parse_args()

    print("基于矩阵分解的评分预测系统 - 内存监控版")
    print(f"使用模型: {args.model}")

    # 初始化监控数据
    max_memory = 0.0
    training_time = 0.0

    if args.monitor:
        print("启用内存和时间监控...")
        # 使用监控函数包装核心功能
        model, max_memory, training_time = monitor_function(train_and_predict, args)
    else:
        print("未启用内存监控")
        start_time = time.time()
        model = train_and_predict(args)
        training_time = time.time() - start_time

    # 保存统计信息
    print(f"开始保存训练统计信息至 {args.stats}")
    save_training_stats(Path(args.stats), model, args, max_memory, training_time)
    print(f"统计信息已保存至 {args.stats}")

    # 控制台摘要
    print("\n===== 训练摘要 =====")
    print(f"模型: {model.model_name}")
    print(f"总训练时间: {training_time:.2f} 秒")
    print(f"峰值内存使用: {max_memory:.2f} MB")
    if model.rmse_history:
        print(f"最终 Train RMSE: {model.rmse_history[-1]:.4f}, 最小 Train RMSE: {min(model.rmse_history):.4f}")
    if model.val_rmse_history:
        print(f"最终 Val RMSE: {model.val_rmse_history[-1]:.4f}, 最小 Val RMSE: {min(model.val_rmse_history):.4f}")

    print("任务完成！")


if __name__ == "__main__":
    main()
