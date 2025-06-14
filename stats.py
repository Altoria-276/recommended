def count_dataset_stats(file_path: str) -> tuple[int, int, int, int, int]:
    """
    统计数据集的基本信息

    参数:
        file_path: 数据文件路径

    返回:
        包含以下统计信息的元组:
            - 用户数量
            - 物品数量
            - 总评分记录数
            - 最大用户ID
            - 最大物品ID
            - 最低评分
            - 最高评分
    """
    user_ids: set[int] = set()
    item_ids: set[int] = set()
    total_ratings: int = 0
    max_user_id: int = 0
    max_item_id: int = 0
    min_rating: float = float("inf")
    max_rating: float = float("-inf")

    with open(file_path, "r") as f:
        while True:
            # 读取用户行
            user_line: str = f.readline().strip()
            if not user_line:  # 文件结束
                break

            # 处理用户行
            if "|" not in user_line:
                continue

            user_id_str: str
            num_ratings_str: str
            user_id_str, num_ratings_str = user_line.split("|")

            user_id: int = int(user_id_str)
            num_ratings: int = int(num_ratings_str)

            user_ids.add(user_id)
            if user_id > max_user_id:
                max_user_id = user_id

            total_ratings += num_ratings

            # 处理该用户的评分行
            for _ in range(num_ratings):
                rating_line: str = f.readline().strip()
                if not rating_line:
                    break

                # 提取物品ID
                parts: list[str] = rating_line.split()
                item_id: int = int(parts[0])

                item_ids.add(item_id)
                if item_id > max_item_id:
                    max_item_id = item_id

    return len(user_ids), len(item_ids), total_ratings, max_user_id, max_item_id


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("使用方法: python stats.py <数据文件路径>")
        sys.exit(1)

    file_path: str = sys.argv[1]
    num_users: int
    num_items: int
    num_ratings: int
    max_user: int
    max_item: int

    num_users, num_items, num_ratings, max_user, max_item = count_dataset_stats(file_path)

    print(f"用户数量: {num_users}")
    print(f"物品数量: {num_items}")
    print(f"评分记录数: {num_ratings}")
    print(f"最大用户ID: {max_user}")
    print(f"最大物品ID: {max_item}")
