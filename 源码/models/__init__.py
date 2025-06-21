# models/__init__.py

from .BaseModel import BaseModel
from .BiasSVD import BiasSVD 
from .SVD import SVD 
from .NeuralMF import NeuralMF 
# from .remove.LightGCN import LightGCN 
# from .remove.SVDpp import SVDpp 
# from .SVDppp import SVDppp 
# from .remove.NGCF import NGCF 
# from .remove.NeuMF import NeuMF 
# from .remove.VAE_CF import VAE_CF 
# from .remove.BPRMF import BPRMF 
# 可选：添加其他模型导入
# from .AnotherModel import AnotherModel

# 定义 __all__ 变量控制导入行为
__all__ = [
    "BaseModel",
    "BiasSVD", 
    "SVD", 
    "NeuralMF", 
    # "LightGCN", 
    # "SVDpp", 
    # "SVDppp", 
    # "BPRMF", 
    # "NGCF", 
    # "NeuMF", 
    # "VAE_CF"
    # 'AnotherModel',
]
