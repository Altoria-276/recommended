import psutil
import time
import os
import threading
import sys
from typing import Any, Callable, Tuple


class Monitor:
    """内存监控类，用于实时监控进程内存使用情况"""

    def __init__(self, interval: float = 0.01):
        """
        初始化内存监控器

        参数:
            interval: 监控间隔（秒）
        """
        self.interval = interval
        self.monitoring = False
        self.max_memory = 0.0
        self.process = psutil.Process(os.getpid())
        self.thread = None

    def _monitor(self):
        """监控线程的内部实现"""
        while self.monitoring:
            try:
                memory_used = self.process.memory_info().rss / (1024 * 1024)  # MB
                if memory_used > self.max_memory:
                    self.max_memory = memory_used
                time.sleep(self.interval)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

    def start(self):
        """开始监控内存使用"""
        if not self.monitoring:
            self.monitoring = True
            self.max_memory = 0.0
            self.thread = threading.Thread(target=self._monitor)
            self.thread.daemon = True
            self.thread.start()
            print("内存监控已启动")

    def stop(self) -> float:
        """停止监控并返回最大内存使用量"""
        if self.monitoring:
            self.monitoring = False
            if self.thread is not None:
                self.thread.join(timeout=1.0)
            print(f"内存监控已停止，峰值内存: {self.max_memory:.2f} MB")
            return self.max_memory
        return 0.0


def monitor_function(func: Callable, *args, **kwargs) -> Tuple[Any, float, float]:
    """
    监控函数执行的时间和内存使用

    参数:
        func: 要监控的函数
        *args, **kwargs: 函数的参数

    返回:
        (最大内存使用量(MB), 运行时间(秒))
    """
    monitor = Monitor()
    monitor.start()

    start_time = time.time()
    try:
        result = func(*args, **kwargs)
    finally:
        max_memory = monitor.stop()
        end_time = time.time()

    run_time = end_time - start_time
    print(f"函数执行时间: {run_time:.2f} 秒")

    return result, max_memory, run_time
