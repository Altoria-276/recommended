import numpy as np
from collections import defaultdict, Counter
import pandas as pd

def count_dataset_stats(file_path: str):
    original_user_ids = set()
    original_item_ids = set()
    all_ratings = []

    user_rating_dict = defaultdict(list)
    item_rating_dict = defaultdict(list)

    with open(file_path, "r") as f:
        while True:
            user_line = f.readline()
            if not user_line:
                break

            user_line = user_line.strip()
            if "|" not in user_line:
                continue

            user_id_str, num_ratings_str = user_line.split("|")
            user_id = int(user_id_str)
            num_ratings = int(num_ratings_str)

            original_user_ids.add(user_id)

            for i in range(num_ratings): 
                rating_line = f.readline().strip() 
                parts = rating_line.split()
                item_id = int(parts[0])
                rating = float(parts[1])

                original_item_ids.add(item_id)

                user_rating_dict[user_id].append(rating)
                item_rating_dict[item_id].append(rating)
                all_ratings.append(rating)

    # 建立 ID 重映射（连续编号）
    user_id_map = {uid: idx for idx, uid in enumerate(sorted(original_user_ids))}
    item_id_map = {iid: idx for idx, iid in enumerate(sorted(original_item_ids))}

    num_users = len(user_id_map)
    num_items = len(item_id_map)
    total_ratings = len(all_ratings)
    rating_array = np.array(all_ratings)
    sparsity = 1 - total_ratings / (num_users * num_items)

    # 每个评分值的统计信息
    rating_counter = Counter(all_ratings)
    score_stats = {f"评分 {score:.1f}": {
        "数量": count,
        "占比": count / total_ratings
    } for score, count in sorted(rating_counter.items())}

    # 汇总信息
    result = {
        "用户数量（重新编号后）": num_users,
        "项目数量（重新编号后）": num_items,
        "总评分记录数": total_ratings,
        "评分矩阵稀疏度": sparsity,
        "评分最小值": rating_array.min(),
        "评分最大值": rating_array.max(),
        "评分均值": rating_array.mean(),
        "评分标准差": np.sqrt(np.mean((rating_array - rating_array.mean()) ** 2)), 
    }

    result.update(score_stats)
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("使用方法: python stats.py <数据文件路径>")
        sys.exit(1)

    file_path = sys.argv[1] 
    stats = count_dataset_stats(file_path)

    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for sub_k, sub_v in v.items():
                print(f"  {sub_k}: {sub_v:.4f}" if isinstance(sub_v, float) else f"  {sub_k}: {sub_v}")
        else:
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
