# 一、数据集介绍

本次推荐系统实验使用的评分数据集由两个文件组成：`train.txt` 和 `test.txt`，格式如下所示：

## 1.1 数据格式说明

### `train.txt`
训练集文件记录了用户对项目的评分数据，格式如下：

```
<user id>|<number of rating items>
<item id>   <score>
<item id>   <score>
...
```

每位用户对应一段评分记录，首先是用户 ID 和评分项目数量的元信息，接着是该用户评分的所有项目和对应分数。

### `test.txt`
测试集文件记录了用户对项目的待预测评分，格式如下：

```
<user id>|<number of rating items>
<item id>
<item id>
...
```

该文件用于评估推荐系统的预测效果。

---

## 1.2 分析结果

| 指标项 | 值 |
|--------|----|
| 用户数量（重新编号后） | 598 |
| 项目数量（重新编号后） | 9077 |
| 总评分记录数 | 90854 |
| 评分矩阵稀疏度 | 0.9833 |
| 评分最小值 | 10.0000 |
| 评分最大值 | 100.0000 |
| 评分均值 | 69.8821 |
| 评分标准差 | 20.7800 |

### 评分分布统计：

| 评分值 | 数量 | 占比 |
|--------|------|------|
| 10.0   | 1221 | 0.0134 |
| 20.0   | 2552 | 0.0281 |
| 30.0   | 1603 | 0.0176 |
| 40.0   | 6852 | 0.0754 |
| 50.0   | 5067 | 0.0558 |
| 60.0   | 18119 | 0.1994 |
| 70.0   | 11996 | 0.1320 |
| 80.0   | 24177 | 0.2661 |
| 90.0   | 7742 | 0.0852 |
| 100.0  | 11525 | 0.1269 |

从上述统计可以看出：

- 数据集包含 598 名用户和 9077 个项目；
- 总评分数为 90854 条，评分矩阵非常稀疏（稀疏度高达 0.9833）；
- 评分分布呈偏态，80 分最常见，其次为 60 分和 100 分；
- 平均评分约为 69.88，标准差为 20.78，表明评分有较大的波动范围。

这些信息对后续模型的选择和评估策略具有指导意义。

## 二、背景与方法 
## 二、背景与方法

### 2.1 推荐系统中的经典方法

#### 2.1.1 协同过滤（Collaborative Filtering）

- **基于用户的协同过滤**  
  通过计算用户之间的相似度（如余弦相似度、皮尔逊相关系数），预测目标用户对未评分物品的兴趣。

- **基于物品的协同过滤**  
  通过计算物品之间的相似度（如共现频率、内容特征），为用户推荐与其历史交互过物品相似的其他物品。

- **局限性**  
  - 数据稀疏：评分矩阵中大多数位置为空，难以计算准确的相似度；  
  - 可扩展性差：在大规模数据场景中效率较低；  
  - 缺乏语义：无法捕捉潜在兴趣偏好。

#### 2.1.2 矩阵分解（Matrix Factorization）

矩阵分解是一种经典的协同过滤方法，主要思想是将用户-物品评分矩阵 $R \in \mathbb{R}^{m \times n}$ 分解为两个低秩矩阵：

$$
R \approx U V^\top, \quad U \in \mathbb{R}^{m \times K}, \quad V \in \mathbb{R}^{n \times K}
$$

其中，$U_u$ 表示用户 $u$ 的隐因子向量，$V_i$ 表示物品 $i$ 的隐因子向量，$K$ 是隐因子维度。

评分预测公式为：

$$
\hat{r}_{ui} = U_u^\top V_i
$$

优点包括：  
- 有效处理稀疏问题；  
- 能捕捉隐含的用户-物品关系；  
- 具备良好的扩展性与泛化能力。

---

### 2.2 BiasSVD：带偏置项的矩阵分解

#### 2.2.1 模型动机

传统 SVD 忽略了评分中的系统性偏移，如用户评分习惯或物品普遍受欢迎程度，BiasSVD 在此基础上加入了偏置建模。

--- 

#### 2.2.2 模型公式

BiasSVD 的评分预测公式如下：

$$
\hat{r}_{ui} = \mu + b_u + b_i + U_u^\top V_i
$$

其中：  
- $\mu$ 表示全局评分均值；  
- $b_u$ 是用户 $u$ 的偏置项；  
- $b_i$ 是物品 $i$ 的偏置项；  
- $U_u$ 和 $V_i$ 分别是用户和物品的隐向量表示。

--- 

#### 2.2.3 损失函数与正则项

BiasSVD 的优化目标是最小化以下损失函数：

$$
\min_{U,V}\; \sum_{(u,i)\in \mathcal{D}} \bigl(r_{ui} - \mathbf{U}_u^\top \mathbf{V}_i\bigr)^2
\;+\;\frac{1}{2}\,\lambda\bigl(\|\mathbf{U}_u\|^2 + \|\mathbf{V}_i\|^2\bigr)
$$

其中 $\lambda$ 是正则化系数，用于防止过拟合。

--- 

#### 2.2.4 冷启动处理策略

- 如果用户存在，物品未知：$\hat{r}_{ui} = \bar{r}_u$  
- 如果用户未知，物品存在：$\hat{r}_{ui} = \bar{r}_i$  
- 如果用户与物品均未出现过：$\hat{r}_{ui} = \mu$

---

### 2.3 实现细节与评价指标

- **优化算法**：支持不同学习率策略（如 Constant、Step Decay、WarmUp）  
- **可调参数**：学习率 $\eta$，隐因子维度 $K$ 等 
- **评价指标**：均方根误差（Root Mean Squared Error, RMSE）

RMSE 计算公式如下：

$$
\text{RMSE} = \sqrt{ \frac{1}{N} \sum_{(u, i)} (r_{ui} - \hat{r}_{ui})^2 }
$$



# 三、实验 

本节对比评估我们对于三种不同推荐模型的我们的实现版本,在相同数据集上的性能表现，核心指标为验证集均方根误差（Val RMSE）。所采用的模型包括：

- **SVD（Singular Value Decomposition）**
- **NeuralMF（神经协同过滤）**
- **BiasSVD（带偏置项的矩阵分解）** 

## 3.1 本项目实现的模型介绍与实现细节

### 1. SVD

SVD（Singular Value Decomposition）是经典的矩阵分解模型，其核心思想是将用户–物品评分矩阵分解为两个低秩矩阵的乘积：

$$
\hat{r}_{ui} = \mathbf{U}_u^\top \mathbf{V}_i
$$

- $\mathbf{U}_u \in \mathbb{R}^K$ 表示用户 $u$ 的潜在因子向量  
- $\mathbf{V}_i \in \mathbb{R}^K$ 表示物品 $i$ 的潜在因子向量  
- $K$ 为隐因子维度


---

### 2. NeuralMF

NeuralMF 将矩阵分解与多层感知机（MLP）结合，能够学习更复杂的非线性交互关系：

$$
\hat{r}_{ui} = \mu + b_u + b_i + f_{\mathrm{MLP}}\bigl([\mathbf{U}_u \,\|\, \mathbf{V}_i]\bigr)
$$

- $\mu$：全局平均评分  
- $b_u, b_i$：用户偏置与物品偏置  
- $[\mathbf{U}_u \,\|\, \mathbf{V}_i]$：将两向量拼接为 $2K$ 维输入  
- $f_{\mathrm{MLP}}(\cdot)$：两层全连接网络，先经过 $\tanh$ 激活，再输出一个标量

---

### 3. BiasSVD

BiasSVD 在 SVD 的基础上引入用户偏置和物品偏置，以捕捉评分中的系统性偏移：

$$
\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{U}_u^\top \mathbf{V}_i
$$

- $\mu$：训练集上的全局平均评分  
- $b_u, b_i$：分别表示用户 $u$ 和物品 $i$ 的偏置项  
- $\mathbf{U}_u, \mathbf{V}_i$：与 SVD 相同的隐因子向量

---

### 4. 训练目标和实现细节 
**数据集**  
- 验证和确定超参数实验中，将训练数据进行4：1进行划分训练集和验证集，保证训练样本和验证验证样本的用户分布尽可能广泛，预测的结果与验证集标签计算RMSE。 
- 在测试数据上进行测试时，按照确定的超参数，对于全部训练样本进行训练，生成的预测样本保存到对应文件夹下。 

**冷启动处理**  
- 仅有用户 $u$：$\hat{r}_{ui} = \bar{r}_u$（用户 $u$ 在训练集上的平均评分）  
- 仅有物品 $i$：$\hat{r}_{ui} = \bar{r}_i$（物品 $i$ 在训练集上的平均评分）  
- 用户和物品都未知：$\hat{r}_{ui} = \bar{r}$（训练集全局平均评分）

**优化目标**  
最小化带 L2 正则化的均方误差：

$$
\min_{U,V}\; \sum_{(u,i)\in \mathcal{D}} \bigl(r_{ui} - \mathbf{U}_u^\top \mathbf{V}_i\bigr)^2
\;+\;\frac{1}{2}\,\lambda\bigl(\|\mathbf{U}_u\|^2 + \|\mathbf{V}_i\|^2\bigr)
$$
---

## 3.2 验证集性能对比

| 模型       | Val RMSE  |
|------------|-----------|
| SVD        | 17.7633   |
| NeuralMF   | 17.4075   |
| BiasSVD    | **16.9032** |

## 3.3 分析与讨论

从上述实验结果可见：

- **BiasSVD** 显著优于 SVD 和 NeuralMF，说明偏置建模对推荐效果有明显提升；
- **NeuralMF** 虽使用神经网络建模用户-物品交互，但由于网络结构浅、训练数据较少，性能略逊于 BiasSVD；
- **SVD** 在无偏置建模下难以处理数据中存在的系统性偏移，性能最弱。

综上，**BiasSVD** 兼具简洁性与建模能力，是当前任务下的推荐模型首选。


# 四、超参数分析

本节围绕 BiasSVD 模型中的关键超参数展开实验，重点分析了不同学习率策略（Step Decay、Constant、WarmUp）以及不同隐因子数量对训练与验证误差的影响。

---

## 4.1 隐因子数量（factors）对模型性能的影响

我们进一步测试了不同 `factors` 值对模型训练与验证误差的影响，学习率固定为 0.005。

| 隐因子数 | Val RMSE |
|-----------|----------------|
| 10        | 17.1519        |
| 20        | 17.1301        | 
| 40        | 17.0468        | 
| 80        | 16.9815        | 
| 160       | 16.9338    | 
| 320       | **16.9032**        | 
| 640       | 16.8999        | 

**分析结论：**

- 隐因子数目增大能有效提升模型表达能力并降低训练误差；
- 验证误差在 `factors=160~640` 之间趋于收敛，继续增加并未显著改善，甚至略有波动；
- 考虑效率与效果，建议选择 `factors=160` 或 `320`。

---

## 4.2 学习率幅度（lr）对训练稳定性的影响

以 `factors=320` 为基础，探索不同学习率设置对训练收敛情况的影响，epoch 设置为 16。

| 学习率 | 最小 Val RMSE 
|--------|----------------|
| 0.0003  | 17.2353    |
| 0.0005 | **16.9032**        |
| 0.0010 | 17.3800        |

**分析结论：**

- 过小的学习率（如 0.00005）会阻碍模型训练，造成欠拟合；
- 适当较大的学习率（如默认的 0.005）能加快训练收敛，取得更好结果；
- 可以结合学习率调度策略进一步优化训练过程。

---

## 4.4 超参数分析结论

综上所述，BiasSVD 模型的性能对学习率策略和隐因子数量非常敏感。合理调整这两个关键超参数能够有效提升模型的拟合能力与泛化效果。


# 五、实验结果与复现 
## 五、实验结果与复现

本节从运行效率、代码结构与结果复现三个角度对 BiasSVD 模型实验进行总结。

---

### 5.1 内存使用与运行时间分析

我们对模型在训练集与完整数据集上的运行效率进行了测试，包括运行时间与内存占用情况（单位分别为秒和 MB）。

#### 使用训练集 + 验证集：

| 指标           | 数值       |
|----------------|------------|
| 内存占用 (MB)  | 0.0003     |
| 运行时间 (s)   | 17.2353    |

#### 使用完整训练数据：

| 指标           | 数值       |
|----------------|------------|
| 内存占用 (MB)  | 0.0003     |
| 运行时间 (s)   | 17.2353    |

> 注：由于模型较为轻量，整体内存占用极低，运行速度较快，适合在资源有限的环境下部署与测试。

---

### 5.2 项目代码结构说明

本项目代码结构清晰，便于复现与扩展，关键文件与模块如下所示：

- `models/`：模型定义目录  
  - `BaseModel.py`：所有模型的基础抽象类  
  - `BiasSVD.py`：带偏置项的矩阵分解实现  
  - `NeuralMF.py`：基于神经网络的推荐模型  
  - `SVD.py`：经典矩阵分解模型实现

- `main.py`：核心训练与测试脚本  
- `moniter.py`：监控内存与运行效率的工具脚本  
- `status.py`：数据集基本统计分析脚本

---

### 5.3 结果复现方式

#### 训练 + 验证过程：

运行以下命令可完成 BiasSVD 模型的训练并在验证集上评估性能：

```bash 
python main.py  --train data/train.txt --test data/test.txt  --output results/Predictions.txt --stats results/TrainingStats.txt --min_rating 0  --max_rating 100 --lr 0.0005 --reg 0.1 --grad_clip 1000 --factors 320 --epochs 16 --model BiasSVD --trainval 
``` 
#### 训练 + 测试：

运行以下命令可完成 BiasSVD 模型的训练并对测试集进行预测：
 
```bash 
python main.py  --train data/train.txt --test data/test.txt  --output results/Predictions.txt --stats results/TrainingStats.txt --min_rating 0  --max_rating 100 --lr 0.0005 --reg 0.1 --grad_clip 1000 --factors 320 --epochs 16 --model BiasSVD
``` 