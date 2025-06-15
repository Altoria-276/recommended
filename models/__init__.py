# models/__init__.py

from .BaseModel import BaseModel
from .BiasSVD import BiasSVD
from .NeuralMF import NeuralMF 
# 可选：添加其他模型导入
# from .AnotherModel import AnotherModel

# 定义 __all__ 变量控制导入行为
__all__ = [
    "BaseModel",
    "BiasSVD", 
    "NeuralMF"
    # 'AnotherModel',
]
