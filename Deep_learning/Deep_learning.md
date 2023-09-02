# 数据清洗/特征选择

![](./屏幕截图%202022-10-18%20172947.jpg)

## 数据描述

1. **<font color = cornflowerblue>鸢尾花数据 / iris.data</font>**
   
   150个样本，每行为1个样本；每个样本有5个字段，分别是
   
   - 花萼长度【cm】
   
   - 花萼宽度【cm】
   
   - 花瓣长度【cm】
   
   - 花瓣宽度【cm】
   
   - 类别
     
     - Iris Setosa
     
     - Iris Versicolour
     
     - Iris Virginica

2. **<font color = cornflowerblue>车辆数据 / car.data</font>**
   
   共1728个样本，每个样本有7个特征
   
   - 购买价格  low / med / high / vhigh
   
   - 维护价格  low / med / high / vhigh
   
   - 车门数量  2 / 3 / 4 / 5more
   
   - 载人数目  2 / 4 / more
   
   - 后备箱大小  small / med / big
   
   - 安全程度  low / med / high
   
   - 接受程度  unacc / acc / good / vgood

## Pandas 数据读取和处理

`pd.set_option('display.width', 200) # 表示输出结果一行最多200个字符`

## Fuzzywuzzy 字符串模糊查找

## 数据清洗和校正

## 特征提取主成分分析 PCA

## One-hot 编码

# 多元回归 / Logistic 回归

若预测身高，温度，价格等连续值称为**回归**

若目标值为离散值，称为**分类**

## 线性回归

$y = kx + b$

<mark>多变量</mark>

$h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2$

$h_\theta(x) = \sum_{i=1}^{m} \theta_i x_i = \theta^T x$

使用极大似然估计解释最小二乘

$$
y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}
$$

<font color = cornflowerblue>误差 LOSS</font> $\epsilon^{(i)} \ (1 \le i \le m)$ 是独立同分布的，服从均值为0，方差为某定值 $\sigma^2$ 的高斯分布

$$
p(\epsilon^{(i)}) = \frac{1}{\sqrt{2\pi} \sigma} exp(-\frac{(\epsilon^{(i)})^2}{2 \sigma^2})
$$

$$
p(y^{(i)} | x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi} \sigma} exp(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2 \sigma^2})
$$

$$
\quad \\
最大似然函数 \\
L(\theta) = \prod_{i=1}^{m} p(y^{(i)} | x^{(i)}; \theta) \\
= \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi} \sigma} exp(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2 \sigma^2})
$$

$x^{(i)}, y^{(i)}, \sigma$ 是已知的，因此是关于参数 $\theta$ 的函数

**<font color = Red>两边取对数</font>**

$$
l(\theta) = \log L(\theta) \\
\quad \\
= \log \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi} \sigma} exp(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2 \sigma^2}) \\
\quad \\
= \sum_{i=1}^{m} \log \frac{1}{\sqrt{2\pi} \sigma} exp(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2 \sigma^2}) \\
\quad \\
= m \log \frac{1}{\sqrt{2\pi} \sigma} - \frac{1}{\sigma^2} \cdot \frac{1}{2} \sum_{i=1}^{m} (y^{(i)} - \theta^T x^{(i)})^2 \\
$$

$m \log \frac{1}{\sqrt{2\pi} \sigma} - \frac{1}{\sigma^2}$ 均为已知数，求 $l(\theta)$ 最大值，则 $L(\theta) = \frac{1}{2} \sum_{i=1}^{m} (y^{(i)} - \theta^T x^{(i)})^2$ 最小值

---

**<font color = Red>$\theta$ 的解析式求解过程</font>**

- 将 $M$ 个 $N$ 维样本组成矩阵 $X$
  
  - $X$ 的每一行对应一个样本，共 $M$ 个样本
  
  - $X$ 的每一列对应样本的一个维度，共 $N$ 维
  
  - <mark>第一列还有额外的一维常数项，全为1</mark>

- 目标函数
  
  $$
  L(\theta) = \frac{1}{2} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 = \frac{1}{2} (X\theta - y)^T (X\theta - y)
  $$

- 梯度【对 $\theta$ 求偏导】
  
  $$
  \nabla_\theta L(\theta) = \nabla_\theta(\frac{1}{2} (X\theta - y)^T (X\theta - y)) \\
\quad \\
= \nabla_\theta(\frac{1}{2} (X^T \theta^T - y^T) (X\theta - y)) \\
\quad \\
= \nabla_\theta (\frac{1}{2} (X \theta X^T \theta^T - X^T \theta^T y - X \theta y^T + y y^T)) \\
\quad \\
= \frac{1}{2} (2X^T X \theta - X^T y - (X y^T)^T) \\
\quad \\
= X^T X \theta - X^T y
  $$
  
  令 $X^T X \theta - X^T y = 0$
  
  $X^T X$ 可逆时，求解出
  
  $$
  \theta = (X^T X)^{-1} X^T y
  $$
  
  $X^T X$ 不可逆或防止过拟合时，增加 $\lambda$ 扰动，即 L2 - norm
  
  $$
  \theta = (X^T X + \lambda I)^{-1} X^T y
  $$
  
  【正则项 / 防止过拟合】
  
  $\lambda > 0; \rho \in [0, 1]$
  
  - L2 - norm 【Ridge】
    
    $$
    L(\theta) = \frac{1}{2} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{i=1}^{n} \theta_j^2
    $$
  
  - L1 - norm 【LASSO】
    
    $$
    L(\theta) = \frac{1}{2} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{i=1}^{n} |\theta_j|
    $$
    
    LASSO 的系数绝对值都很小，小到可以忽略的程度，有降维能力
  
  - 弹性网  Elastic Net
    
    $$
    L(\theta) = \frac{1}{2} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 \\
    $$
+ $$
  \lambda (\rho \cdot \sum_{i=1}^{n} |\theta_j| + (1 - \rho \cdot \sum_{i=1}^{n} \theta_j^2))
  $$
  
  ![](./屏幕截图%202022-10-18%20224305.jpg)

---

<mark>梯度下降算法</mark>

求解以下 Loss

$$
L(\theta) = \frac{1}{2} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
$$

1. 初始化 $\theta$ 【随机初始化】

2. 沿着负梯度方向迭代，更新后的 $\theta$ 使 $L(\theta)$ 更小
   
   $$
   \theta = \theta - \alpha \cdot \frac{\partial L(\theta)}{\partial \theta}
   $$
   
   > $\alpha$ : 学习率
   
   ![](./屏幕截图%202022-10-18%20225036.jpg)
   
   <font color = cornflowerblue>具体两种</font>
   
   - 批量梯度下降算法
     
     $$
     \theta_j: \theta_j + \alpha \sum_{i=1}^{m} (y^{(i)} -  h_\theta(x^{(i)}))x^{(i)}_j
     $$
     
     沿着所有样本的梯度去下降
   
   - 随机梯度下降算法【用的更多】
     
     $$
     \theta_j: \theta_j + \alpha (y^{(i)} -  h_\theta(x^{(i)}))x^{(i)}_j
     $$
     
     沿着某个样本下降
     
     【第1个样本下降一次，第2个样本再下降一次，...，第 m 个样本再下降一次】
   
   > 若干个样本的平均梯度作为更新方向，则是 mini-batch 梯度下降算法

## Logistic 回归

**<font color = Red>sigmoid 函数</font>**

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

![](./屏幕截图%202022-10-19%20170950.jpg)

<mark>Logistic</mark>

$$
h_\theta (x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
$$

<font color = cornflowerblue>参数迭代</font>

$$
\theta_j: = \theta_j + \alpha (y^{(i)} -  h_\theta(x^{(i)}))x^{(i)}_j
$$

<font color = cornflowerblue>损失函数</font>

$$
loss(y_i, \hat{y}_i) = \sum_{i = 1}^{m} (\ln(1 + e^{-y_i \cdot f_i}))
$$

## 多分类: Softmax 回归

K 分类

# 决策树 / 随机森林

## 决策树

### ID3

使用信息增益 / 互信息 g(D, A) 进行特征选择

- 取值多的属性，更容易使数据更纯，其信息增益更大

- 训练得到的是一颗庞大且深度浅的树：不合理

### C4.5

信息增益率

$$
g_r(D, A) = g(D, A) / H(A)
$$

### CART

Classification and regression tree

基尼指数

---

<font color = cornflowerblue>一个属性的信息增益（率）/ gini 指数越大，表明属性对样本的熵减少的能力更强，这个属性使得数据由不确定性变成确定性的能力越强</font>

> <font color = Red>决策树非常容易过拟合</font>
> 
> ![](./屏幕截图%202022-10-22%20204752.jpg)
> 
> <mark>防止过拟合的方式</mark>
> 
> 1. 剪枝
>    
>    三种决策树的剪枝过程算法相同，区别仅是对于当前树的评价标准不同
>    
>    【剪枝总体思路】
>    
>    - 由完全树 $T_0$ 开始，剪枝部分结点得到 $T_1$，再次剪枝部分节点得到 $T_2 \dots$ 直到仅剩树根的树 $T_k$
>    
>    - 在验证数据集上对这 $k$ 个树分别评价，选择损失函数最小的树 $T_\alpha$
>    
>    有预剪枝和后剪枝，前者更常用
> 
> 2. 随机森林
>    
>    见下述

## 随机森林

对样本和特征都做随机

【Bootstraping】

**<font color = cornflowerblue>Bagging 的策略【样本随机】</font>**

- 从样本集中重采样【有重复的】选出 n 个样本

- 在所有属性上，对这 n 个样本建立分类器【ID3, C4.5, CART, SVM, Logistic 等】

- 重复以上两步 m 次，即获得了 m 个分类器

- 将数据放在这 m 个分类器上，最后根据这 m 个分类器的投票结果，决定数据属于哪一类

有放回采样

![](./屏幕截图%202022-10-22%20234001.jpg)

Bootstrap 每次约有36.79%的样本不会出现在 Bootstrap 所采集的样本集合中，将未参与模型训练的数据称为袋外数据 OOB(Out Of Bag)；它可以用于取代测试集用于误差估计

**<font color = cornflowerblue>随机森林</font>**

在 Bagging  基础上做了修改

- 从样本集中用 Bootstrap 采样选出 n 个样本

- 从所有属性中随机选择 k 个属性，选择最佳分割属性作为节点建立 CART 决策树

- 重复以上两步 m 次，即建立了 m 棵 CART 决策树

- 这 m 个 CART 形成随机森林，通过投票表决结果，决定数据属于哪一类

---

【随机森林 / Bagging 和决策树的关系】

可以使用决策树作为基本分类器，也可以使用 SVM, Logistic 回归等其他分类器，习惯上，这些分类器组成的 “总分类器”，任然叫作随机森林

---

【投票机制】

- 简单投票机制
  
  - 一票否决（一致表决）
  
  - 少数服从多数
    
    - 有效多数（加权）
  
  - 阈值表决
    
    e.g.  大于多少不要

- 贝叶斯投票机制
  
  > e.g.
  > 
  > 假定有 N 个用户可以为 X 个电影投票（假定投票者不能给同一电影重复投票），投票有1，2，3，4，5星共5档
  > 
  > 如何根据用户投票，对电影排序？
  > 
  > > 本质仍然是分类问题
  > > 
  > > 对于某个电影，有 N 个决策树，每个决策树对该电影有1个分类（1，2，3，4，5）类，求这个电影应该属于哪一类

---

【样本不均衡的常用处理方法】

假定样本数目 A 类比 B 类多，且严重不平衡

对 A 类进行**降采样**

![](./屏幕截图%202022-10-23%20001728.jpg)

对 B 类进行**过采样**

![](./屏幕截图%202022-10-23%20001927.jpg)

同时进行

![](./屏幕截图%202022-10-23%20002011.jpg)

---

【使用 RF 建立计算样本间相似度】

<mark>原理</mark>: 若两样本同时出现在相同叶结点的次数越多，则二者越相似

<mark>算法过程</mark>

- 记样本个数为 N，初始化 N * N 的零矩阵 S，S[i, j] 表示样本 i 和样本 j 的相似度

- 对于 m 颗决策树形成的随机森林，遍历所有决策树的所有叶子结点
  
  - 记该叶结点包含的样本为 sample[1, 2, ..., k]，则 S[i][j] 累加1
    
    - 样本 i, j 属于 sample[1, 2, ..., k]
    
    - 样本 i, j 出现在相同叶结点的次数增加1次

- 遍历结束，则 S 为样本间相似度矩阵

# SVM

<mark>三种不同的 SVM</mark>

1. 线性可分支持向量机
   
   ![](./屏幕截图%202022-10-23%20171510.jpg)
   
   - 硬间隔最大化  hard margin maximization
   
   - 硬间隔支持向量机

2. 线性支持向量机【分类允许犯错】
   
   - 软间隔最大化  soft margin maximization
   
   - 软间隔支持向量机

3. 非线性支持向量机
   
   - 核函数  kernel function

$C / \gamma$ 值变大，正确率提高，但是有可能会产生过拟合，泛化能力在降低

![](./屏幕截图%202022-10-23%20173120.jpg)

<mark>样本到直线距离取最小值的最大值就是 SVM</mark>

<font color = cornflowerblue>输入数据</font>

- 假设给定一个特征空间上的训练数据集 $T = \{ ( x_1, y_1), (x_2, y_2), \dots, (x_N, y_N)\}$，其中 $x_i \in R^n, y_i \in \{+1, -1 \}, i = 1, 2, \dots, N$；

- $x_i$ 为第 $i$ 个实例（若 $n > 1$，$x_i$ 为向量；

- $y_i$ 为 $x_i$ 的类标记，当 $y_i = +1$ 时，称 $x_i$ 为正例；当 $y_i = -1$ 时，称 $x_i$ 为负例

- $(x_i, y_i)$ 称为样本点

## 线性可分支持向量机

- 给定线性可分训练数据集，通过间隔最大化得到的分离超平面
  
  $$
  w^T \Phi (x) + b = 0
  $$
  
  相应的分类决策函数 $f(x) = sign (w^T \Phi (x) + b)$
  
  该决策函数称为线性可分支持向量机

- $\Phi (x)$ 是某个确定的特征空间转换函数，它的作用是将 $x$ 映射到更高的维度
  
  > 最简单直接的 $\Phi (x) = x$

- 求解分离超平面问题可以等价为求解相应的凸二次规划问题

<font color = cornflowerblue>目标函数</font>

点到直线的距离公式

$$
d = \frac{y_i \cdot (w^T \cdot \Phi (x_i) + b)}{||w||}
$$

根据样本到直线距离取最小值的最大值就是 SVM

$$
\argmax_{w, b} \quad \{\frac {1}{||w||} \min_i [y_i \cdot (w^T \cdot \Phi (x_i) + b)] \}
$$

![](./屏幕截图%202022-10-23%20195515.jpg)

总可以通过等比例缩放 $w$ 的方法，使得两类点的函数值都满足 $|y| \ge 1$

则 $y_i \cdot (w^T \cdot \Phi (x_i) + b) \ge 1$

得新目标函数

$$
\argmax_{w, b} \frac{1}{||w||}
$$

分母最小值，即 $\min ||w|| = \min \frac{1}{2} ||w||^2$

最终

$$
\min_{w, b} \frac{1}{2} ||w||^2 \\
s.t. \quad y_i \cdot (w^T \cdot \Phi (x_i) + b) \ge 1 \quad i = 1, 2, \dots, n
$$

## 线性支持向量机

若数据线性不可分，则增加松弛因子 $\xi_i \ge 0$，使函数间隔加上松弛变量大于等于1，则约束条件变成

$$
y_i \cdot (w \cdot x_i + b) \ge 1 - \xi_i
$$

目标函数变为

$$
\min_{w, b} \frac{1}{2} ||w||^2 + C \sum_{i = 1}^N \xi_i
$$

## 核函数

使用核函数，将原始输入空间映射到新的特征空间，从而，使得原本线性不可分的样本可能在核空间可分

- 多项式核函数
  
  $$
  \kappa(x_1, x_2) = (x_1 \cdot x_2 + c)^d
  $$

- 高斯核 RBF 函数
  
  $$
  \kappa(x_1, x_2) = exp(-\gamma \cdot ||x_1 - x_2||^2)
  $$

- Sigmoid 核函数
  
  $$
  \kappa(x_1, x_2) = tanh(x_1 \cdot x_2 + c)
  $$

# 聚类

聚类就是对大量未知标注的数据集，按数据的内在相似性将数据集划分为多个类别，使类别内的数据相似度较大而类别间的数据相似度较小

## K-Means

![](./屏幕截图%202022-10-24%20010157.jpg)

K-Means 是初值敏感的

## 层次聚类

对给定数据集进行层次分解，直到某种条件满足为止；具体又可分为

1. AGNES 算法
   
   凝聚的层次聚类
   
   一种自底向上的策略，首先将每个对象作为一个簇，然后合并这些原子簇为越来越大的簇，直到某个终结条件被满足
   
   <mark>簇间距离的不同定义</mark>
   
   - 最小距离
     
     - 两个集合中最近的两个样本的距离
     
     - 容易形成链状结构
   
   - 最大距离  complete
     
     - 两个集合中最远的两个样本的距离
     
     - 若存在异常值则不稳定
   
   - 平均距离  average
     
     - 两个集合中样本间两两距离的平均值
   
   - 方差  Ward
     
     - 使得簇内距离平方和最小，簇间平方和最大

2. DIANA 算法
   
   分裂的层次聚类
   
   采用自顶向下的策略，它首先将所有对象置于一个簇中，然后逐渐细分为越来越小的簇，直到达到了某个终结条件

## 密度聚类

指导思想    只要样本点的密度大于某阈值，则将该样本添加到最近的簇中

可发现任意形状的聚类，且对噪声数据不敏感

1. DBSCAN    Density-Based Spatial Clustering of Applications with Noise
   
   <font color = cornflowerblue>本质将簇定义为密度相连的点的最大集合</font>，能够把具有足够高密度的区域划分为簇
   
   ![](./屏幕截图%202022-10-24%20013750.jpg)
   
   <mark>算法流程</mark>
   
   - 如果一个点 $p$ 的 $\epsilon$ - 领域包含多于 $m$ 个对象，则创建一个 $p$ 作为核心对象的新簇
   
   - 寻找合并核心对象直接密度可达的对象
   
   - 没有新点可以更新簇时，结束
   
   根据以上算法可知
   
   1）每个簇至少包含一个核心对象
   
   2）非核心对象可以的簇的一部分，构成簇的边缘
   
   3）包含过少对象的簇被认为是噪声

2. 密度最大值算法
   
   一种简洁优美的聚类算法，可以识别各种形状的类簇，并且参数很容易确定
   
   定义：局部密度 $\rho_i$
   
   $$
   \rho_i = \sum_j \chi (d_{ij} - d_c) \\
    \chi(x) =
    \begin{cases}
    1 \quad x < 0 \\
    0 \quad otherwise
    \end{cases}
   $$
   
   $d_c$ 是一个截断距离，$\rho_i$ 即到对象 $i$ 的距离小于 $d_c$ 的对象的个数；由于该算法只对 $\rho_i$ 的相对值敏感，所以对 $d_c$ 的选择是稳健的，一种推荐做法是选择 $d_c$，使得平均每个点的邻居数为所有点的1%-2%
   
   定义：高局部密度点距离 $\delta_i$
   
   ![](./%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202022-10-24%20084247.jpg)
   
   
   密度比当前黄色星星大的才成为为候选值，候选值中取最近的值作为到黄色数据的高局部密度点距离
   
   $$
   \delta_i = \min_{j: \rho_j > \rho_i} (d_{ij})
   $$
   
   ![](./屏幕截图%202022-10-24%20084823.jpg)

# EM

# 前向 / 反向传播

## Siftmax 分类器

- 归一化
  
  $$
  P(Y = k | X = x_i) = \frac{e^s k}{\sum_j s^s j}
  $$

- 计算损失值
  
  $$
  L_i = - \log P(Y = y_i | X =x_i)
  $$

![](./屏幕截图%202022-10-27%20093101.jpg)

## 前向传播

![](./屏幕截图%202022-10-27%20093609.jpg)

以上即是一个前向传播

## 反向传播

![](./屏幕截图%202022-10-27%20102951.jpg)

![](./屏幕截图%202022-10-27%20103349.jpg)

- 加法门单元：均等分配

- MAX 门单元：给最大的

- 乘法门单元：互换的感觉

# 神经网络整体架构

![](./屏幕截图%202022-10-27%20103723.jpg)

1. 层次结构

2. 神经元
   
   矩阵的大小，特征

3. 全连接
   
   input layer 中第一个神经元连接着 hidden layer 1 中的四个神经元，以此类推

4. 非线性
   
   ![](./屏幕截图%202022-10-27%20105005.jpg)
   
   - 线性方程
     
     $$
     f = W x
     $$
   
   - 非线性方程
     
     $$
     f = W_2 \max (0, W_{1x})
     $$
     
     继续堆叠一层
     
     $$
     f = W_3 \max(0, W_2 \max(0, W_{1x})
     $$

<mark>神经网络的强大之处在于，用更多的参数来拟合复杂的数据</mark>

<font color = cornflowerblue>神经元个数越多，训练出来效果越好，但是要注意神经元过多容易过拟合</font>

## 正则化 / 防止过拟合 / 梯度下降算法

<mark>正则化</mark>

$\lambda > 0; \rho \in [0, 1]$

- L2 - norm 【Ridge】
  
  $$
  L(\theta) = \frac{1}{2} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{i=1}^{n} \theta_j^2
  $$

- L1 - norm 【LASSO】
  
  $$
  L(\theta) = \frac{1}{2} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{i=1}^{n} |\theta_j|
  $$
  
  LASSO 的系数绝对值都很小，小到可以忽略的程度，有降维能力

- 弹性网 Elastic Net
  
  $$
  L(\theta) = \frac{1}{2} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 \\
  $$

- $$
  \lambda (\rho \cdot \sum_{i=1}^{n} |\theta_j| + (1 - \rho \cdot \sum_{i=1}^{n} \theta_j^2))
  $$
  
  ![](./屏幕截图%202022-10-18%20224305.jpg)

<mark>防止过拟合</mark>

![](./屏幕截图%202022-10-27%20132043.jpg)

---

<mark>梯度下降算法</mark>

求解以下 Loss

$$
L(\theta) = \frac{1}{2} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
$$

1. 初始化 $\theta$ 【随机初始化】

2. 沿着负梯度方向迭代，更新后的 $\theta 使 L(\theta)$ 更小
   
   $$
   \theta = \theta - \alpha \cdot \frac{\partial L(\theta)}{\partial \theta}
   $$
   
   > $\alpha$ : 学习率
   
   ![](./屏幕截图%202022-10-18%20225036.jpg)
   
   <font color = cornflowerblue>具体两种</font>
   
   - 批量梯度下降算法
     
     $$
     \theta_j: \theta_j + \alpha \sum_{i=1}^{m} (y^{(i)} -  h_\theta(x^{(i)}))x^{(i)}_j
     $$
     
     沿着所有样本的梯度去下降
   
   - 随机梯度下降算法【用的更多】
     
     $$
     \theta_j: \theta_j + \alpha (y^{(i)} -  h_\theta(x^{(i)}))x^{(i)}_j
     $$
     
     沿着某个样本下降
     
     【第1个样本下降一次，第2个样本再下降一次，...，第 m 个样本再下降一次】

> 若干个样本的平均梯度作为更新方向，则是 mini-batch 梯度下降算法

---

<mark>惩罚力度对结果的影响</mark>

![](./屏幕截图%202022-10-27%20121113.jpg)

$\lambda = 0.001$ 有可能过拟合

<mark>参数个数对结果的影响</mark>

![](./屏幕截图%202022-10-27%20121321.jpg)

## 激活函数

常见的激活函数有 Sigmoid, Relu, Tanh 等

<mark>激活函数对比</mark>

- Sigmoid
  
  ![](./屏幕截图%202022-10-27%20121656.jpg)
  
  图像上限以及下限的导数约等于0，也就是说参数不进行更新了

- Relu
  
  ![](./屏幕截图%202022-10-27%20121758.jpg)

---

## 预处理 / 参数初始化

1. 数据预处理
   
   不同的预处理结果会使得模型的效果发生很大的差异
   
   ![](./屏幕截图%202022-10-27%20131205.jpg)

2. 参数初始化
   
   参数初始化同样重要
   
   通常我们都使用随机策略来进行参数初始化
   
   `w = 0.01 * np.random.randn(D, H)`

# 卷积神经网络

## 应用领域

- 检测任务
  
  ![](./屏幕截图%202022-10-27%20132803.jpg)

- 分类与检索
  
  【检索：拍照搜同款】
  
  ![](./屏幕截图%202022-10-27%20132835.jpg)

- 超分辨率重构
  
  ![](./屏幕截图%202022-10-27%20133024.jpg)

- 医学任务

- 无人驾驶

- 人脸识别
