# 机器学习概述

## 机器学习的应用

- 异常检测是数据挖掘中一个重要方面，用来发现“小的模式”（相当于聚类），即数据集中间显著不同于其他数据的对象。

- 异常探测应用
  
  - 电信和信用卡欺骗
  
  - 贷款审批
  
  - 药物研究
  
  - 气象预报
  
  - 金融领域
  
  - 客户分类
  
  - 网络入侵
  
  - 检测故障检测与诊断等

## 什么是机器学习

数据

模型

预测

- 数据集的构成
  
  特征值 + 目标值

## 机器学习算法分类

- 监督学习
  
  - 目标值 : 类别 ----- 分类问题
  
  - 目标值 : 连续性问题 ----- 回归问题

- 无监督学习、
  
  目标值 : 无

# 数据集

## 可用数据集

- scikit-learn 数据集
  
  > 特点：1. 数据量较小
  > 
  >             2. 方便学习
  
  [scikit-learn](http://scikit-learn.org/stable/datasets/index.html#datasets)

- UCI 数据集
  
  > 特点：1. 收录了 360 个数据集
  > 
  >             2. 覆盖科学、生活、经济等领域
  > 
  >             3. 数据量几十万
  
  [UCI](http://archive.ics.uci.edu/ml/)

- Kaggle
  
  > 特点：1. 大数据竞赛平台
  > 
  >             2. 80 万科学家
  > 
  >             3. 真实数据
  > 
  >             4. 数据量巨大
  
  [Kaggle](https://www.kaggle.com/datasets)

## Scikit-learn 工具介绍

1. Python 语言的机器学习工具

2. Scikit-learn 包括许多知名的机器学习算法的实现

3. Scikit-learn 文档完善、容易上手，丰富的 API

4. 目前稳定版本 0.19.1

# sklearn 数据集

## scikit-learn 数据集 API 介绍

- sklearn.datasets
  
  - 加载获取流行数据集
  
  - datasets.load_*()
    
    - 获取小规模数据集，数据包含在 datasets 里
  
  - datasets.fetch_*(data_home = None)
    
    - 获取大规模数据集，需要从网络上下载，函数的第一个参数是 data_home, 表示数据集下载的目录，默认是 ~/scikit-learn_data/

## sklearn 小数据集

- sklearn.datasets.load_iris()
  
  加载并返回鸢尾花数据集

- sklearn.datasets.load_boston()
  
  加载并返回波士顿房价数据集

## sklearn 大数据集

- `sklearn.datasets.fetch_20newsgroups(data_home = None, subset = 'train')`
  
  - subset: 'train' 或者 'test', 'all' 可选，选择要加载的数据集
  
  - 训练集的 ’训练‘，测试集的 ’测试‘ ，两者的 ’全部‘

## sklearn 数据集的使用

1. **sklearn 数据集返回值的介绍**
   
   load 和 fetch 返回的数据类型 datasets.base.Bunch (继承自字典)
   
   - data: 特征数据数组，是 [n_samples * n_features] 的二维 numpy.ndarray 数组
   
   - target: 目标值 / 标签数组，是 n_samples 的一维 numpy.ndarray 数组
   
   - DESCR: 数据描述
   
   - feature_names: 特征名
   
   - target_names: 目标值的名字 / 标签名
     
     > <u>datasets.base.Bunch</u>
     > 
     > dict['key'] = values
     > 
     > bunch.key = values
   
   ```python
   from sklearn.datasets import load_iris
   # 获取鸢尾花数据集
   iris = load_iris()
   print('鸢尾花数据集的返回值:\n', iris)
   # 返回值是一个继承自字典的 Bench
   print('鸢尾花的特征值:\n', iris['data'])
   print('鸢尾花的目标值:\n', iris.target)
   print('鸢尾花特征的名字:\n', iris.feature_names)
   print('鸢尾花目标值的名字:\n', iris.target_names)
   print('鸢尾花的描述:\n', iris.DESCR)
   ```

2. **数据集的划分**
   
   - 训练数据：用于训练，构建模型
   
   - 测试数据：在模型检验时使用，用于评估模型是否有效，比例 20%-30%
   
   > **数据集划分 API**
   > 
   > - `sklearn.model_selection.train_test_split(arrays, *options)`
   >   
   >   - x 数据集的特征值
   >   
   >   - y 数据集的标签值
   >   
   >   - test_size 测试集的大小，一般为 float
   >   
   >   - random_state 随机数种子，不同的种子会造成不同的随机采样结果；相同的种子采样结果相同
   >   
   >   - return 训练集特征值，测试集特征值，训练集目标值，测试集目标值 (x_train, x_test, y_train, y_test)
   >   
   >   ```python
   >   from sklearn.model_selection import train_test_split
   >   x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.2, random_state = 22)
   >   # test_size 默认为 0.25
   >   print('训练集的特征值:\n', x_train, x_train.shape)
   >   ```

# 特征工程简介

## 为什么需要特征工程 (Feature Engineering)

业界广泛流传，数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已

## 什么是特征工程

特征工程是使用专业背景知识和技巧处理数据，使得特征能在机器学习算法上发挥更好的作用的过程

## 特征工程包含内容

### 1. 特征抽取

将任意数据 (如文本或图像) 转换为可用于机器学习的数字特征叫**特征抽取**

> 特征值化是为了计算机更好的去理解数据

**特征提取 API**

`sklearn.feature_extraction`

- 字典特征提取  [属于类别的数据 -> one-hot 编码]
  
  `sklearn.feature_extraction.DictVectorizer(sparse = True, ...)`
  
  - DictVectorizer.fit_transform(X)
    
    X : 字典或者包含字典的迭代器
    
    返回值 : 返回 sparse 矩阵
  
  - DictVectorizer.inverse_transform(X)
    
    X : array 数组或者 sparse 矩阵 (稀疏矩阵 : 将非零值按位置表示出来)
    
    返回值 : 转换之前数据格式
  
  - DictVectorizer.get_feature_names()
    
    返回类别名称
  
  ```python
  [{'city':'北京', 'temperature':100},
  {'city':'上海', 'temperature':60},
  {'city':'深圳', 'temperature':30}]
  ```
  
  Out: ![](./屏幕截图%202022-04-10%20221214.png)
  
  sklearn 实现
  
  ```python
  from sklearn.feature_extraction import DictVectorizer
  def dict_demo():
      data = [{'city':'北京', 'temperature':100}, {'city':'上海', 'temperature':60}, {'city':'深圳', 'temperature':30}]
      # 1.实例化一个转换器类
      transfer = DictVectorizer()
      # transfer = DictVectorizer(sparse = False)
      # 则与上一段代码结果相同
      # 2.调用 fit_transform()
      data_new = transfer.fit_transform(data)
      print('data_new:\n', data_new)
      print('特征名字:\n', transfer.get_feature_names())
  
  dict_demo()
  ```
  
  Out: data_new:
  (0, 1)    1.0
  (0, 3)    100.0
  (1, 0)    1.0
  (1, 3)    60.0
  (2, 2)    1.0
  (2, 3)    30.0
  
  特征名字:
  ['city=上海', ‘city=北京', 'city=深圳', 'temperature']
  
  > **应用场景**
  > 
  > 1) pclass, sex 数据集当中类别特征比较多的时候
  > 
  > 2) 本身拿到的数据就是字典类型

- 文本特征提取 [特征词]
  
  `sklearn.feature_extraction.text.CountVectorizer(stop_word = [])`  [统计每个样本特征词出现的个数]
  
  - 返回词频矩阵
  
  - CountVectorizer.fit_transform(X)
    
    X : 文本或者包含文本字符串的可迭代对象
    
    返回值 : 返回 sparse 矩阵
  
  - CountVectorizer.inverse_transform(X)
    
    X : array 数组或者 sparse 矩阵
    
    返回值 : 转换之前数据格
  
  - CountVectorizer.get_feature_names()
    
    返回值 : 单词列表
  
  ```python
  ['life is short, i like python',
  'life is too long, i dislike python']
  ```
  
  Out: ![](./屏幕截图%202022-04-11%20153912.png)
  
  sklearn 实现
  
  ```python
  from sklearn.feature_extraction.text import CountVectorizer
  def count_demo():
      data = ['life is short, i like  like python', 'life is too long, i dislike python']
      # 1.实例化一个转换器类
      transfer = CountVectorizer()
      # 2.调用 fit_transform
      data_new = transfer.fit_transform(data)
      print('data_new:\n', data_new.toarray())
      print('特征名字\n', transfer.get_feature_names())
  
  count_demo()
  ```
  
  Out: data_new:
  [[0 1 1 2 0 1 1 0]
  [1 1 1 0 1 1 0 1]]
  特征名字
  ['dislike', 'is', 'life', 'like', 'long', 'python', 'short', 'too']
  
  stop_words --  停用词
  
  **中文文本**
  
  ```python
  import jieba
  def cut_word(text):
      return " ".join(list(jieba.cut(text)))
  def count_chinese_demo():
      data = ['我们看到的从很远是系来是在几百万年之前发出的，这样当我们看到']
      data_new = []
      for sent in data:
          data_new.append(cut_word(sent))
      print(data_new)
      # 1.实例化一个转换器类
      transfer = CountVectorizer()
      # 2.调用 fit_transform
      data_final = transfer.fit_transform(data_new)
      print('data_final:\n', data_final.toarray())
      print('特征名字\n', transfer.get_feature_names())
  ```
  
  **另一种方法**
  
  **Tf-idf 文本特征提取**
  
  衡量一个词的重要程度
  
  ① 公式
  
  - 词频 (term frequency, tf) 指的是某一个给定的词语在该文件中出现的频率
  
  - 逆向文档频率 (inverse document frequency, idf) 是一个词语普遍重要性的度量；某一特定词的 idf, 可以由总文件数目除以包含该词语之文件的数目，再将得到的商取以 10 为底的对数得到

### 2.  特征预处理

概念

通过**一些转换函数**将特征数据**转换成成更加适合算法模型**的特征数据过程

- 包含内容
  
  - 数值型数据的无量钢化
    
    - 归一化
    
    - 标准化

- 特征预处理 API
  
  `sklearn.preprocessing`
  
  - 为什么要进行归一化\标准化
    
    特征的**单位或者大小相差较大，或者某特征的方差相比其它的特征要大出几个数量级，容易影响（支配）目标结果**，使得一些算法无法学习到其它的特征
    
    我们需要用到一些方法进行**无量钢化，使不同规格的数据转换到统一规格**

- 归一化
  
  - 定义
    
    通过对原始数据进行交换把数据映射到 (默认为[0, 1]) 之间
  
  - 公式
    
    $$
    X' = \frac{x - \min}{\max - \min} \\
    X'' = X' \times (mx - mi) + mi
    $$
    
    > 作用于每一列，$\max$ 为一列的最大值，$\min$ 为一列的最小值，那么 $x$ 为最终结果，$mx, mi$ 分别为指定区域值默认 $mx$ 为 1，$mi$ 为 0
  
  - API
    
    - `sklearn.preprocessing.MinMaxScaler(fearure_range = (0, 1)...)`
      
      i). MinMaxScaler.fit_transform(X)
      
          X: numpy array 格式的数据 [n_samples, n_features]
      
      ii). 返回值: 转换后的形状相同的 array
    
    - sklearn 实现
      
      ```python
      from sklearn.preprocessing import MinMaxScaler
      import pandas as pd
      def minmax_demo():
          # 1.获取数据
          pd.read_csv('dating.txt')
          data.iloc[:, :3]
          # 2.实例化一个转换器类
          transfer = MinMaxScaler()
          # 3.调用 fit_transform
          data_new = transfer.fit_transform(data)
          print('data_new:\n', data_new)
      ```

- 标准化
  
  - 定义
    
    通过对原始数据进行变换把数据变换到均值为 0，标准差为 1 范围内
  
  - 公式
    
    $$
    X' = \frac{x -mean}{\sigma}
    $$
    
    > 作用于每一列，$mean$ 为平均值，$\sigma$ 为标准差
  
  - API
    
    - `sklearn.preprocessing.StandardScaler()`
      
      i). 处理之后，对每列来说，所有数据都聚集在均值为 0 附近，标准差为 1
      
      ii). StandardScaler.fit_transform(X)
      
           X: numpy array 格式的数据 [n_sample, n_features]
      
      iii). 返回值: 转换后的形状相同的 array
    
    - sklearn 实现
      
      ```python
      from sklearn.preprocessing import StandardScaler
      def stand_demo():
          # 1.获取数据
          pd.read_csv('dating.txt')
          data.iloc[:, :3]
          # 2.实例化一个转换器类
          transfer = StandardScaler()
          # 3.调用 fit_transform 
          data_new = transfer.fit_transform(data)
          print('data_new:\n', data_new)
      ```
    
    > 标准化总结
    > 
    > 在已有样本足够多的情况下比较稳定，适合现代嘈杂大数据场景

### 3. 特征降维

降维是指在某些限定条件下，**降低随机变量 (特征) 个数**，得到**一组“不相关”主变量**的过程

- 相关特征 (correlated feature)
  
  - 相对湿度与降雨量之间的相关
  
  - 等等

> 正是因为在进行训练的时候，我们都是使用特征进行学习，如果特征本身存在问题或者特征之间相关性较强，对于算法学习预测会影响较大

**降维的两种方法**

- **特征选择**
  
  数据中包含**冗余或相关变量 (或称特征、属性、指标等)**，旨在从**原有特征中找出主要特征**
  
  - 方法
    
    - Filter (过滤式) : 主要探究特征本身特点、特征与特征和目标值之间关联
      
      **a)** 方差选择法：低方差特征过滤
      
      **b)** 相关系数
    
    - Embedded (嵌入式) : 算法自动选择特征 (特征与目标值之间的关联)
      
      **a)** 决策树 : 信息熵、信息增益
      
      **b)** 正则化 : L1、L2
      
      **c)** 深度学习 : 卷积等
  
  - 模块
    
    `sklearn.feature_selection`
  
  - 过滤式
    
    - 低方差特征过滤
      
      i. 特征方差小 : 某个特征大多样本的值比较相近
      
      ii. 特征方差大 : 某个特征很多样本的值都有差别，适合保留
    
    - API
      
      `sklearn.feature_selection.VarianceThreshold(threshold = 0.0`
      
      **a)** 删除所有低方差特征
      
      **b)** Variance.fit_transform(X)
      
      X: numpy array 格式的数据 [n_samples, n_features]
      
      返回值：训练集差异低于 threshold 的特征将被删除；默认值是保留所有非零方差特征，即删除所有样本中具有相同值的特征
      
      - sklearn 实现
        
        ```python
        from sklearn.feature_selection import VarianceTheshold
        def variance_demo():
            # 1.获取数据
            data = pd.read_csv('factor_returns.csv')
            data = data.iloc[:, 1:-2]
            print('data:\n', data)
            # 2.实例化一个转换器类
            transfer = VarianceThreshold()
            # 3.调用 fit_transform
            data_new = transfer.fit_transform(data)
            print('data_new:\n', data_new)
        
        variance_demo()
        ```
    
    - 相关系数

- **主成分分析**
  
  > - 定义：**高维数据转化为低维数据的过程**，在此过程中**可能会舍弃原有数据、创造新的变量**
  > 
  > - 作用：**是数据维数压缩，尽可能降低原数据的维数（复杂度），损失少量信息**
  > 
  > - 应用：回归分析或者聚类分析当中
  
  - API
    
    `sklearn.decomposition.PCA(n_components=None, 
    copy=True, whiten=False, svd_solver=’auto’,
     tol=0.0, iterated_power=’auto’, random_state=None)`
    
    - 将数据分解成较低维数空间
    
    - n_components
      
      i. 小数：表示保留百分之多少的信息
      
      ii. 整数：减少到多少特征
      
      要保留的成分数量，其值类型可以设为整型，浮点型，字符串。如果不指定该值，n_components == min(n_samples, n_features)；如果n_components == ‘mle’，并且svd_solver == ‘full’，则使用Minka’s MLE方法估计维度。当0 < n_components < 1时，并且svd_solver == 'full’时，方差值必须大于n_components，如果 n_components == ‘arpack’，则n_components必须严格的等于特征与样本数之间的最小值。
    
    - PCA.fit_transform(X)  X: numpy array 格式的数据 [n_samples, n_features]
    
    - 返回值 : 转换后指定维度的 array
    
    sklearn 实现
    
    ```python
    from sklearn.decomposition import PCA
    def pca_demo():
        data = [[2, 8, 4, 5], 
                [6, 3, 0, 8],
                [5, 4, 9, 1]]
        # 1.实例化一个转换器
        transfer = PCA(n_components = 2)
        # 2.调用 fit_transform
        data_new = transfer.fit_transform(data)
        print('data_new:\n', data_new)
    
    pca_demo()
    ```

# 特征编码

## 便签编码

```python
from sklearn.preprocessing import LabelEncoder

# define example
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = array(data)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
```

## one-hot 编码

```python
'''
接标签编码代码
'''
from sklearn.preprocessing import OneHotEncoder

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
```

```python
from sklearn.preprocessing import LabelBinarizer

# 单个特征
nominal = np.array([["A"],
                   ["B"],
                   ["C"],
                   ["D"]])
# 导入LabelBinarizer

one_hot = LabelBinarizer()  # 创建one-hot编码器
one_hot.fit_transform(nominal)  # 对特征进行one-hot编码


'''
# 转换前nominal
array([['A'],
       ['B'],
       ['C'],
       ['D']], dtype='<U1')
# 转换后结果
array([[1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 1, 0],
       [0, 0, 0, 1]])
'''
```

```python
from sklearn.preprocessing import MultiLabelBinarizer

# 多个特征
multi_nominal = np.array([["A","Black"],
                         ["B","White"],
                         ["C","Green"],
                         ["D","Red"]])

multi_one_hot = MultiLabelBinarizer()
multi_one_hot.fit_transform(multi_nominal)


'''
# 转换前结果
array([['A', 'Black'],
       ['B', 'White'],
       ['C', 'Green'],
       ['D', 'Red']], dtype='<U5')
# 转换后结果
array([[1, 0, 1, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 1, 0, 1, 0, 0],
       [0, 0, 0, 0, 1, 0, 1, 0]])
'''
```

## 倒置编码

```python
'''
接标签编码代码
'''
# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
print(inverted)
```

# 监督学习

## sklearn 转换器和估计器

- 转换器
  
  - 实例化（实例化的是一个转换器类 (Transformer)）
  
  - 调用 fit_transform (对于文档建立分类词频矩阵，不能同时调用)

- 估计器 (sklearn 机器学习算法的实现)
  
  在 sklearn 中，估计器 (estimator) 是一个重要的角色，是一类实现了算法的 API
  
  - 用于分类的估计器
    
    - sklearn.neighbors  k-近邻算法
    
    - sklearn.naive_bayes  贝叶斯
    
    - sklearn.linear_model.LogisticRegression  逻辑回归
    
    - sklearn.tree  决策树与随机森林
  
  - 用于回归的估计器
    
    - sklearn.linear_model_LinearRegression  线性回归
    
    - sklearn.linear_model.Ridge  岭回归
  
  - 用于无监督学习的估计器
    
    - sklearn.cluster.KMeans  聚类
  
  估计器工作流程
  
  1. 实例化一个 estimator
  
  2. 调用 estimator.fit(x_train, y_train) 方法 [调用完毕，模型生成]
  
  3. 模型评估
     
     a) 直接比对真实值和预测值
     
     ```python
     y_predict = estimator.predict(x_test)
     y_test == y_predict
     ```
     
     b) 计算准确率
     
     `accuracy = estimator.score(x_test, y_test)`

## K-近邻算法 (KNN 算法)

### 什么是 KNN 算法 (K Nearest Neighbor 算法)

根据你的 “邻居” 来推断出你的类别

### KNN 算法原理

- 定义
  
  如果一个样本在特征空间中的 **k 个最相似 (即特征空间中最邻近) 的样本中的大多数属于某一个类别**，则该样本也属于这个类别
  
  > k = 1 , 容易受到异常点的影响

- 距离公式
  
  比如 a(a1,a2,a3); b(b1,b2,b3)
  
  $\sqrt{(a1-b1)^2 + (a2-b2)^2 + (a3-b3)^2}$
  
  > k 值取得过小，容易收到异常点的影响
  > 
  > k 值取得过大，样本不均衡的影响

### API

`sklearn.neighbors.KNeighborsClassifier(n_neighbors=5,algorithm='auto')`

- n_neighbors (k值) : int, 可选（默认 = 5）, k_neighbors 查询默认使用的邻居数 

- algorithm: {'auto', 'ball_tree', 'kd_tree', 'brute'}, 可选用于计算最近邻居的算法：'ball_tree', 'kd_tree' 将使用 KDTree; 'auto' 将尝试根据传递给 fit 方法的值来决定最合适的算法 (不同实现方式影响效率)

### 案例: 鸢尾花种类预测

iris 数据集是常用的分类实验数据集，由 Fisher, 1936收集整理；Iris 也称鸢尾花卉数据集，是一类多重变量分析的数据集；关于数据集的具体介绍：

> 实例数量：150 (三个类各有50个)
> 
> 属性数量：4 (数值型，数值型，帮助预测的属性和类)
> 
> Attribute Information: 
> 
> - sepal length 萼片长度（厘米）
> 
> - sepal width 萼片宽度（厘米）
> 
> - petal length 花瓣长度（厘米）
> 
> - petal width 花瓣宽度（厘米）
> 
> - class:
>   
>   - Iris-Setosa 山鸢尾
>   
>   - Iris-Versicolor 变色鸢尾
>   
>   - Iris-Virginica 维吉尼亚鸢尾

> **案例分析**
> 
> 1. 获取数据
> 
> 2. 数据集划分
> 
> 3. 特征工程
>    
>    - 标准化
> 
> 4. KNN 预估器流程
> 
> 5. 模型评估

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
def knn_iris():
    # 1.获取数据
    iris = load_iris()
    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 6)
    # 3.特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 4.KNN算法预估器
    estimator = KNeighborsClassifier(n_neighbors = 3)
    estimator.fit(x_train, y_train)
    # 5.模型预估
    # 方法1：直接比对真实值和预测
    y_predict = estimator.predict(x_test)
    print('y_predict:\n', y_predict)
    print('直接比对真实值和预测值:\n', y_test == y_predict)
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print('准确率为:\n', score)

knn_iris()
```

### KNN 算法优缺点

**优点**

- 简单；易于理解

- 易于实现

- 无需训练

**缺点**

- 懒惰算法，对测试样本分类时的计算量大，内存开销大

- 必须指定 K 值，K 值选择不当则分类精度不能保证

> 使用场景：小数据场景，几千~几万样本，具体场景具体业务去测试

## 模型选择与调优

### 什么是交叉验证 (cross validation)

将拿到的训练数据，分为训练和验证集；以下图为例：将数据分成4份，其中一份作为验证集；然后经过4次的测试，每次都更换不同的验证集；即得到4组模型的结果，取平均值作为最终结果；又称4折交叉验证

![](./屏幕截图%202022-04-24%20222740.png)

> 为什么需要交叉验证
> 
> 目的：为了让被评估的模型更加准确可信

### 超参数搜索-网格搜索 (Grid Search)

通常情况下，**有很多参数是需要手动指定的 (如 K-近邻算法中的 K 值), 这种叫超参数**；但是手动过程繁杂，所以需要对模型预设几种超参数组合；**每组超参数都采用交叉验证来进行评估；最后选出最优参数组合建立模型**。

### API

`sklearn.model_selection.GridSearchCV(estimator,param_grid=None,cv=None)`

- 对估计器的指定参数值进行详尽搜索

- estimator: 估计器对象

- param_grid: 估计器参数 (dict){'n_neighbors':[1, 3, 5]}

- cv: 指定几折交叉验证

- fit(): 输入训练数据

- score(): 准确率

- 结果分析
  
  - 最佳参数: best_params_
  
  - 最佳结果: best_score_
  
  - 最佳估计器: best_estimator_
  
  - 交叉验证结果: cv_result_

### 案例: 鸢尾花增加 K 值调优

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
def knn_iris_gscv():
    # 1.获取数据
    iris = load_iris()
    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 6)
    # 3.特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 4.KNN算法预估器
    estimator = KNeighborsClassifier()
    # 加入网格搜索与交叉验证
    # 参数准备
    param_dict = {'neighbors':[1, 3, 5, 7, 9, 11]}
    estimator = GridSearchCV(estimator, param_grid = param_dict, cv = 10)
    estimator.fit(x_train, y_train)
    # 5.模型预估
    # 方法1：直接比对真实值和预测
    y_predict = estimator.predict(x_test)
    print('y_predict:\n', y_predict)
    print('直接比对真实值和预测值:\n', y_test == y_predict)
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print('准确率为:\n', score)
```

### 案例: 预测 facebook 签到位置

本次大赛的目的是预测一个人想签入到哪个地方；对于本次比赛的目的，Facebook 的创建一个人造的世界，包括位于 10 平方公里超过 10 万米的地方；对于一个给定的坐标，你的任务是返回最有可能的地方的排名列表；数据制作出类似于移动设备的位置的信号，给你需要什么与不准确的，嘈杂的价值观复杂的真实数据工作一番风味；不一致和错误的位置数据可能破坏，如 Facebook 入住服务经验。

**File descriptions**

train.csv, test.csv

- row_id: if of the check-in event

- x y: coordinates

- accuracy: location accuracy

- time: timestamp (时间戳)

- place_id: id of the business, this is the target you are predicting

#### Analysis

1. 获取数据

2. 数据处理
   
   - 特征值 x
   
   - 目标值 y
   
   - 缩小数据范围 [2 < x < 2.5; 1.0 < y < 1.5]
   
   - time -> 年月日时分秒
   
   - 过滤签到次数少的地点

3. 特征工程

4. KNN 算法预估流程

5. 模型选择与调优

6. 模型评估

#### Code

> jupyter notebookestimator = GridSearchCV(estimator, param_grid = param_dict, cv = 10)

```python
import pandas as pd
# 获取数据
data = pd.read_csv('./FBlocation/train.csv')
data.head()
# 数据处理
# 缩小数据范围 [2 < x < 2.5; 1.0 < y < 1.5] 
data.query('x < 2.5 & x > 2 & y < 1.5 & y > 1.0')
# time -> 年月日时分秒
time_value = pd.to_datetime(data['time'], unit = 's')
data = pd.DatatumeIndex(time_value)
data['day'] = data.day
data['weekday'] = data.weekday
data['hour'] = data.hour
# 过滤签到次数少的地点
place_count = data.groupby('place_id').count()['row_id]
place_count[place_count > 3]
data_final = data[data['place_id'].isin(place_count[place_count > 3].index.values)]
# 筛选特征值和目标值
x = data_final[['x', 'y', 'accuracy', 'day', 'weekday', 'hour']]
y = data_final['place_id']
# 划分数据集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)
# 特征工程
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
# KNN 算法预估流程
estimator = KNeighborsClassifier()
# 模型选择与调优
# 参数准备
param_dict = {'neighbors':[3, 5, 7, 9]}
estimator = GridSearchCV(estimator, param_grid = param_dict, cv = 3)
estimator.fit(x_train, y_train)
# 模型评估
# 方法1：直接比对真实值和预测
y_predict = estimator.predict(x_test)
print('y_predict:\n', y_predict)
print('直接比对真实值和预测值:\n', y_test == y_predict)
# 方法2：计算准确率
score = estimator.score(x_test, y_test)
print('准确率为:\n', score)
```

## 朴素贝叶斯算法

### 朴素贝叶斯原理

#### 什么是朴素贝叶斯算法

分类算法之后会出现概率值，取概率比较大的为最终结果

#### 概率基础

##### 概率 (Probability) 定义

- 一件事情发生的可能性
  
  - 认出**质地均匀**的硬币，结果头像朝上

- P(X) : 取值在 [0, 1]

##### 联合概率、条件概率与相互独立

1. **联合概率**：包含多个条件，且所有条件同时成立的概率
   
   记作：P(A, B)
   
   例如：P(程序员, 匀称)；P(程序员, 超重|喜欢)

2. **条件概率**：事件 A 在另外一个事件 B 已经发生的条件下发生的概率
   
   记作：P(A|B)
   
   例如：P(程序员|喜欢)；P(程序员, 超重|喜欢)

3. **相互独立**：如果 P(A, B) = P(A)P(B)，则称事件 A 与事件 B  相互独立

##### 贝叶斯公式

$$
P(C|W) = \frac{P(W|C)P(C)}{P(W)}
$$

注：W 为给定文档的特征值 (频数统计, 预测文档提供)，C 为文档类别

#### 原理

朴素贝叶斯，简单理解，就是假定了特征与特征之间相互独立的贝叶斯公式

也就是说，朴素贝叶斯、之所以朴素，就在于假定了特征与特征相互独立

> 朴素？
> 
> 即为假设：特征与特征之间是相互独立的

朴素贝叶斯算法：朴素 + 贝叶斯

#### 应用场景

文本分类

### 朴素贝叶斯算法对文本分类

$$
P(C|W) = \frac{P(W|C)P(C)}{P(W)}
$$

> 公式分为三个部分
> 
> - P(C): 每个文档类别的概率 (某文档类别数 / 总文档数量)
> 
> - P(W|C): 给定类别下特征 (被预测文档中出现的词) 的概率
>   
>   - 计算方法: P(F1|C) = Ni / N (训练文档中去计算)
>     
>     - Ni 为该 F1 词在 C 类别所有文档中出现的次数
>     
>     - N 为所属类别 C 下的文档所有词出现的次数和
> 
> - P(F1, F2, ...) 预测文档中每个词的概率

##### 拉普拉斯平滑系数

目的：防止计算出的分类概率为0

$$
P(F1|C) = \frac{Ni+\alpha }{N+\alpha m}
$$

> $\alpha$ 为指定的系数一般为 1，$m$ 为训练文档中统计出的**特征词**个数

#### API

`sklearn.naive_bayes.MultinomialNB(alpha=1.0)`

- 朴素贝叶斯分类

- alpha: 拉普拉斯平滑系数

#### 案例: 20类新闻分类

**分析**

1. 获取数据

2. 划分数据集

3. 特征工程
   
   - 文本特征抽取

4. 朴素贝叶斯预估器流程

5. 模型评估

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomiaNB
def nb_news():
    # 获取数据
    news = fetch_20newsgroups(subset = 'all')
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)
    # 文本特征抽取
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 朴素贝叶斯预估器流程
    estimator = MultinomialNB()
    estimator.fit(x_train, y_train)
    # 模型评估
    # 方法1：直接比对真实值和预测
    y_predict = estimator.predict(x_test)
    print('y_predict:\n', y_predict)
    print('直接比对真实值和预测值:\n', y_test == y_predict)
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print('准确率为:\n', score)
```

### 朴素贝叶斯算法优缺点

**优点**

- 朴素贝叶斯模型发源于古典数学理论，有稳定的分类效率

- 对缺失数据不太敏感，算法也比较简单，常用于文本分类

- 分类准确度高，速度快

**缺点**

- 由于使用了样本属性独立性的假设，所以如果特征属性有关联时其效果不好

## 决策树

### 认识决策树

决策树思想的来源非常朴素，程序设计中的条件分支结构就是 if-else 结构，最早的决策树就是利用这类结构分割数据的一种分类学习方法

![](./屏幕截图%202022-04-29%20224624.png)

> 如何高效的进行决策？
> 
> 决定一下特征的先后顺序

### 决策树分类原理

![](./屏幕截图%202022-04-29%20225151.png)

已知 四个特征，需预测 是否贷款给某个人

先看房子，再看工作 即可决定是否贷款

#### 信息熵

信息的衡量 (信息量)

> 信息：香农提出，为消除随机不定性的东西
> 
> 小明  年龄    “我今年18岁”  --  信息
> 
> 小华  “小明明年19岁”  --  不是信息

**H 的专业术语称之为信息熵，单位为比特**

公式

$$
H(X) = -(\sum_{i = 1}^n P(x_i) \log_bP(x_i))
$$

#### 信息增益 ---- 决策树划分依据之一

**特征 A 对训练数据集 D 的信息增益 g(D, A), 定义为集合 D 的信息熵 H(D) 与特征 A 给定条件下 D 的信息条件熵 H(D|A) 之差**

公式

$$
g(D,A) = H(D)-H(D|A)
$$

条件熵的计算

$$
H(D|A) = \sum_{i=1}^n \frac {\vert D_{ik}\vert}{\vert D\vert}H(D_i) \\
= -\sum_{i=1}^n \frac {\vert D_i\vert}{\vert D\vert}\sum_{k=1}^K \frac{\vert D_{ik}\vert}{\vert D_i\vert} \log \frac{\vert D_{ik}\vert}{\vert D_i \vert}
$$

*注：信息增益表示得知特征 X 的信息而息的不确定性减少的程度使得类 Y 的信息熵减少的程度*；**信息增益越大则该项作为划分的第一个特征**

> 例: 某人 已知年龄 工作 房子 信贷情况 问是否贷款
> 
> $$
> H(D) = -(\frac{6}{15} \times \log_2 \frac{6}{15} + \frac{9}{15} \times \log_2 \frac{9}{15}) \approx 0.971 \\
-\\
H(D|年龄) = \frac{1}{3} H(青年) + \frac{1}{3} H(中年)+\frac{1}{3} H(老年) \\
-\\
H(青年) = -(\frac{2}{5} \times \log \frac{2}{5} + \frac{3}{5} \times \log \frac{3}{5}) \\
-\\
H(中年) = -(\frac{2}{5} \times \log \frac{2}{5} + \frac{3}{5} \times \log \frac{3}{5}) \\
-\\
H(老年) = -(\frac{1}{5} \times \log \frac{1}{5} + \frac{4}{5} \times \log \frac{4}{5})
> $$

### API

`class sklearn.tree.DecisionTreeClassifier(criterion = 'gini', max_depth = None, random_state = None)`

- 决策树分类器

- criterion: 默认是 'gini' 系数，也可以选择信息增益的熵 'entropy'

- max_depth: 树的深度大小

- random_state: 随机数种子

#### 案例: 鸢尾花

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
def decision_iris():
    # 1.获取数据
    iris = load_iris()
    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 22)
    # 3.决策树预估器
    estimator = DecisionTreeClassifier(criterion = 'entropy')
    estimator.fit(x_train, y_train)
    # 4.模型评估
    # 方法1：直接比对真实值和预测
    y_predict = estimator.predict(x_test)
    print('y_predict:\n', y_predict)
    print('直接比对真实值和预测值:\n', y_test == y_predict)
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print('准确率为:\n', score)
    # 5.可视化决策树
    export_graphviz(estimator, out_file = 'iris_tree.dot', feature_names = iris.feature_names)
```

#### 决策树可视化

1. **保存树的结构到 dot 文件**
   
    `sklearn.tree.export_graphviz()`该函数能够导出 DOT 格式
   
       `tree.export_graphviz(estimator, out_file = 'tree.dot', feature_names = ['',''])`

2. **网站显示结构**
   
   <http://webgraphviz.com/>

### 决策树优缺点

**优点**

- 简单的理解和解释，数目可视化，可解释能力强

**缺点**

- 决策树学习者可以创建不能很好地推广数据的过于复杂的树，这被称为过拟合

**改进**

- 减枝 cart 算法 (决策树 API 当中已经实现，随机森林参数调优有相关介绍)

- 随机森林

### 案例: 泰坦尼克号乘客生存预测

泰坦尼克号数据

在泰坦尼克号和 titanic2 数据帧描述泰坦尼克号上的个别乘客的生存状态；这里使用的数据集是由各种研究人员开始的；其中包括许多研究人员创建的旅客名单，由 Michael A. Findlay 编辑；我们提取的数据集中的特征是票的类别，存货，乘坐班，年龄，登陆，home.dest，房间，票，船和性别

- **乘坐班是指乘客班 (1, 2, 3)，是社会经济阶层的代表**

- **其中 age 数据存在缺失**
  
  数据：<http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt>

流程分析  <特征值  目标值>

1. 获取数据

2. 数据处理
   
   - 缺失值处理
   
   - 特征值  —>  字典类型

3. 准备特征值  目标值

4. 划分数据集

5. 特征工程：字典特征抽取

6. 决策树预估器流程

7. 模型评估

> jupyter notebook

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
# 1.获取数据
path = 'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txthttp://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt'
titanic = pd.read_csv(path)
# 3.准备特征值 目标值/筛选特征值目标值
x = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
# 2.数据处理
# 1）缺失值处理
x['age'].fillna(x['age'].mean(), inplace = True)
# 2）特征值转化字典类型
x = x.to_dict(orient = 'records')
# 4.划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 22)
# 5.特征工程：字典特征抽取
transfer = DictVectorizer()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
# 6.决策树预估器
estimator = DecisionTreeClassifier(criterion = 'entropy')
estimator.fit(x_train, y_train)
# 7.模型评估
# 方法1：直接比对真实值和预测
y_predict = estimator.predict(x_test)
print('y_predict:\n', y_predict)
print('直接比对真实值和预测值:\n', y_test == y_predict)
# 方法2：计算准确率
score = estimator.score(x_test, y_test)
print('准确率为:\n', score)
# 5.可视化决策树
export_graphviz(estimator, out_file = 'titanic_tree.dot', feature_names = transfer.get_feature_names())
```

## 随机森林

### 什么是集成学习方法

集成学习通过建立几个模型组合来解决单一预测问题；它的工作原理是**生成多个分类器/模型**，各自独立地学习和作出预测；**这些预测最后结合成组合预测，因此优于任何一个单分类的做出预测**

### 什么是随机森林

在机器学习中，**随机森林是一个包含多个决策树的分类器**，并且其输出的类别是由个别树输出的类别的众数而定

### 随机森林原理

> **随机**
> 
> 两个随机：
> 
> - 训练集随机
>   
>   bootstrap  随机有放回抽样
> 
> - 特征随机
>   
>   从 M 个特征中随机抽取 m 个特征

### API

`class sklearn.ensemble.RandomForestClassifier(n_estimator = 10, criterion = 'gini', max_depth = None, bootstrap = True, random_state = None, min_samples_split = 2)`

- n_estimators: integer, optional (default = 10) 森林里的树木数量 120, 200, 300, 500, 800, 1200

- criteria: string, 可选 (default = 'gini') 分割特征的测量方法

- max_depth: integer 或 None，可选 (默认 = 无) 树的最大深度 5, 8, 15, 25, 30

- max_features = 'auto', 每个决策树的最大特征数量
  
  - If 'auto', then `max_features = sqrt(n_features)`
  
  - If 'sqrt', then `max_features = sqrt(n_features)` (same as 'auto')
  
  - If 'log2', then `max_features = log2(n_features)`
  
  - If None, then `max_features = n_features`

- bootstrap: boolean, optional (default = True) 是否在构建树时使用放回抽样

- min_samples_split: 节点划分最少样本数

- min_samples_leaf: 叶子结点的最小样本数

超参数: n_estimator, max_depth, min_samples_split, min_sample_leaf

### 案例: 泰坦尼克号

```python
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
# 1.获取数据
path = 'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txthttp://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt'
titanic = pd.read_csv(path)
# 3.准备特征值 目标值/筛选特征值目标值
x = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
# 2.数据处理
# 1）缺失值处理
x['age'].fillna(x['age'].mean(), inplace = True)
# 2）特征值转化字典类型
x = x.to_dict(orient = 'records')
# 4.划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 22)
# 5.特征工程：字典特征抽取
transfer = DictVectorizer()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
# 6.随机森林
estimator = RandomForestClassifier()
# 7.网格搜索与交叉验证
# 参数准备
param_dict = {'n_estimators':[120, 200, 300, 500, 800, 1200]
              'max_depth':[5, 8, 15, 25, 30]}
estimator = GridSearchCV(estimator, param_grid = param_dict, cv = 3)
estimator.fit(x_train, y_train)
# 8.模型评估
# 方法1：直接比对真实值和预测
y_predict = estimator.predict(x_test)
print('y_predict:\n', y_predict)
print('直接比对真实值和预测值:\n', y_test == y_predict)
# 方法2：计算准确率
score = estimator.score(x_test, y_test)
print('准确率为:\n', score)
```

### 优点

- 在当前所有算法中，具有极好的准确率

- 能够有效地运行在大数据集上，处理具有高维特征的输入样本，而且不需要降维

- 能够评估各个特征在分类问题上的重要性

## 分类评估方法

#### 精确率与召回率

> **混淆矩阵**
> 
> 在分类任务下，预测结果 (Predicted Condition) 与正确标记 (True Condition) 之间存在四种不同的组合，构成混淆矩阵（适用于多分类）
> 
> ![](./屏幕截图%202022-05-31%20203319.png)
> 
> TP = True Possitive
> 
> FN = False Negative

1. ***精确率  Precision***
   
   预测结果为正例样本中真实为正例的比例
   
   ![](./屏幕截图%202022-05-31%20203625.png)

2. ***召回率 Recall***
   
   真实为正例的样本中预测结果为正例的比例（查的全，对正样本的区分能力）
   
   ![](./屏幕截图%202022-05-31%20203942.png)
   
   应用场景：
   
   1）肿瘤预测
   
   2）工厂质量检测

3. ***F1-score, 反映了模型的稳健性***
   
   $$
   F1 = \frac{2TP}{2TP+FN+FP} = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
   $$

##### API

- <u>**精确率**</u>
  
  `sklearn.metrics.precision_score`

- <u>**召回率**</u>
  
  `sklearn.metrics.recall_score`

- <u>**精确率与召回率**</u> 【分类模型评价报告】
  
  `sklearn.metrics.classification_report(y_true, y_pred, labels = [], target_names = None)`
  
  - y_true: 真实目标值
  
  - y_pred: 估计器预测目标值
  
  - labels: 指定类别对应的数字
  
  - target_names: 目标类别名称
  
  - return: 每个类别**精确率与召回率**

- <u>**F1 值  F1-score**</u>
  
  `sklearn.metrics.cohen_kappa_score`

##### 以上案例进行评估

```python
from sklearn.metrics import classification_report
report = classification_report(y_test, y_predict, labels = [2, 4], target_names = ['良性', '恶性'])
```

![](./屏幕截图%202022-05-31%20210229.png)

#### ROC 曲线与 AUC 指标

样本不均衡的评估

$$
TPR = \frac{TP}{TP + FN}
$$

> 所有真实类别为1的样本中，预测类别为1的比例；即为召回率

$$
FPR = \frac{FP}{FP + TN}
$$

> 所有真实类别为0的样本中，预测类别为1的比例

##### ROC 曲线

ROC 曲线的横轴就是 FPRate，纵轴就是 TPRate，当二者相等时，表示的意义则是：对于不论真实类别是1还是0的样本，分类器预测为1的概率是相等的，此时 AUC 为0.5

![](./屏幕截图%202022-05-31%20210859.png)

`sklearn.metrics_roc_curve`

##### AUC 指标

- AUC 的概率意义是随机抽取一对正负样本，正样本得分大于负样本的概率

- AUC 的最小值为0.5，最大值为1，取值越高越好

- **AUC = 1，完美分类器，采用这个预测模型时，不管设定什么阈值都能得出完美预测；绝大多数预测的场合，不存在完美分类器**

- **0.5 < AUC < 1，优于随机猜测；这个分类器（模型）妥善设定阈值的话，能有预测价值**

> 最终 AUC 的范围在 [0.5, 1] 之间，并且越接近1越好

##### AUC 计算 API

`sklearn.metrics.roc_auc_score(y_true, y_score)`

- 计算 ROC 曲线面积，即 AUC 值

- y_true: 每个样本的真实类别，必须为0（反例），1（正例）标记

- y_score: 预测得分，可以是正类的估计概率、置信值或者分类方法的返回值

![](./屏幕截图%202022-05-31%20213111.png)

> - **AUC 只能用来评价二分类**
> 
> - **AUC 非常适合评价样本不平衡中的分类器性能**

## 以上算法属于是分类算法

## 线性回归

回归问题：目标值 -- 连续性的数据

### 线性回归的原理

#### 线性回归应用场景

- 房价预测 目标值：房价

- 销售额度预测 目标值：销售额

- 金融：贷款额度预测、利用线性回归以及系数分析因子

#### 什么是线性回归

1. 定义与公式
   
   线性回归 (Linear regression) 是利用回归方程（函数）对一个或**多个自变量（特征值）和因变量（目标值）之间**关系进行建模的一种分析方式
   
   > 只有一个自变量的情况称为单变量回归，多于一个自变量情况的叫作多元
   
   通用公式（线性模型）：
   
   $$
   h(w) = w_1x_1 + w_2x_2 + \cdots +b = w^Tx + b
   $$
   
   其中 $w,x$ 可以理解为矩阵
   
   $$
   w = \begin{bmatrix}
    b \\
    w_1 \\
    w_2
    \end{bmatrix},
    x = \begin{bmatrix}
    1 \\
    x_1 \\
    x_2
    \end{bmatrix}
   $$

2. 线性回归的特征与目标的关系分析
   
   广义线性模型 -- 非线性关系
   
   线性回归当中线性模型有两种，一种是线性关系，另一种是非线性关系。**在这里我们智能画一个平面更好去理解，所以都用单个特征或两个特征举例子**
   
   - 线性关系
     
     <img title="" src="./屏幕截图 2022-05-20 113424.png" alt="" data-align="inline">
     
     <img title="" src="./屏幕截图 2022-05-20 113007.png" alt="" data-align="inline">
     
     > 单特征与目标值的关系呈直线关系
     > 
     > 两个特征与目标值呈现平面关系
     > 
     > 更高维度的我们不用自己去想，记住这种关系即可
   
   - 非线性关系
     
     ![](./屏幕截图%202022-05-20%20114357.png)
     
     > 满足线性模型的两种情况（任意一种）
     > 
     > - 自变量（特征值）是一次的
     > 
     > - 参数是一次的
     >   
     >   比如
     >   
     >   $$
     >   y = w_1x_1 + w_2x_1^2 + w_3x_1^3 + \cdots + b
     >   $$
     >   
     >   **线性关系一定是线性模型，线性模型不一定是线性关系**

### 损失函数

 如何找到一组准确合适的线性模型？怎么算把模型求出来？

未知：权重和偏置 /  回归系数和偏置

目标：求出回归系数，回归系数能都使得预测准确

> 房价例子
> 
> 真实关系：
> 
> $$
> 真实房价= 0.02 \times 中心区域的距离+0.04\times 城市一氧化氮浓度+(-0.12\times 自住房平均房价)+0.254\times 城镇犯罪率
> $$
> 
> 随意假定：
> 
> $$
> 预测房价=0.25\times 中心区域的距离+0.14\times 城市一氧化氮浓度+0.42\times 自住房平均房价+0.34\times 城镇犯罪率
> $$
> 
> 将特征值带入假定公式，与真实值有误差使用一种方法迭代更新将误差减小至接近0，则回归系数会比较准确

真实值与预测值之间的差距我们如何来衡量呢？

将衡量他们的关系叫作**损失函数/cost/成本函数/目标函数**

**损失函数定义**

总损失定义为

$$
J(\theta)=(h_w(x_1)-y_1)^2+(h_w(x_2)-y_2)^2+\cdots +(h_w(x_m)-y_m)^2 \\
 = \sum_{i=1}^m (h_wx_i)-y_i)^2
$$

- $y_1$ 为第 $i$ 个训练样本的真实值

- $h(x_i)$ 为第 $i$ 个训练样本特征值组合预测函数

- 又称最小二乘法

> 如何去减少这个损失，使我们预测的更加准确些？既然存在了这个损失，我们一直说机器学习有自动学习的功能，在线性回归这里更是能够体现。这里可以通过一些优化方法去优化（其实是数学当中的求导功能）回归的总损失！

### 优化算法

如何去求解模型当中的W，使得损失最小？（目的是找到最小损失对应的W值）

#### 正规方程

直接求解W

$$
w = (X^TX)^{-1}X^Ty
$$

理解：$X$ 为特征值矩阵，$y$ 为目标值矩阵；直接求到最好的结果

缺点：当特征过多过复杂时，求解速度太慢并且得不到结果

##### API

`sklearn.linear_model.Li nearRegression(fit_intercept = True)`

- fit_intercept: 是否计算偏置

- LinearRegression.coef_: 回归系数

- LinearRegression.intercept_: 偏置

#### 梯度下降 (Gradient Descent)

$$
w_1 = w_1 - \alpha \frac{\partial cost(w_0+w_1x_1)}{\partial x_1} \\
w_0 = w_0 - \alpha \frac{\partial cost(w_0+w_1x_1)}{\partial x_1} 
$$

![](./屏幕截图%202022-05-21%20211808.png)

理解：$\alpha$ 为学习率，需要手动指定（超参数），$\alpha$ 旁边的整体表示方向，沿着这个函数下降的方向找，最后就能找到山谷的最低点，然后更新w值

使用：面对训练数据规模十分庞大的任务，能够找到较好的结果

##### API

`sklearn.linear_model.SGDRegressor(loss = 'squared_loss', fit_intercept = True, learning_rate = 'invscaling', eta0 = 0.01)`

- SGDRegressor 类实现了随机梯度下降学习，它支持不同的 **loss 函数和正则化惩罚项**来拟合线性回归模型

- loss: 损失类型
  
  - **loss = 'squared_loss**: 普通最小二乘法

- fit_intercept: 是否计算偏置

- learning_rate: string, optional
  
  - 学习率填充
  
  - **'constant': eta = eta0**
  
  - **'optimal': eta = 1.0/(alpha * (t + t_0)) [default]**
  
  - **'invscaling': eta = eta0/pow(t, power_t)**
    
    - **power_t = 0.25: 存在父类当中**
  
  - **对于一个常数值的学习率来说，可以使用 learning_rate = 'constant', 并使用 eta0 来指定学习率**

- SGDRegressor.coef_: 回归系数

- SGDRegressor.intercept_: 偏置

##### 一些梯度下降的方法 GD SGD SAG

1. GD
   
   梯度下降 (Gradient Descent)，原始的梯度下降法需要计算所有样本的值才能够得出梯度，计算量大，所以后面才会有一系列的改进

2. SGD
   
   随机梯度下降 (Stochastic gradient descent) 是一个优化方法；它在一次迭代时只考虑一个训练样本
   
   - 优点：高效，容易实现
   
   - 缺点
     
     需要许多超参数，比如正则项参数、迭代数
     
     对于特征标准化是敏感的

3. SAG
   
   随机平均梯度法 (Stochasitic Average Gradient)，由于收敛的速度太慢，有人提出 SAG 等基于梯度下降的算法
   
   Scikit-learn: 岭回归、逻辑回归等当中都会有 SAG 优化

#### 两种方法对比

| 梯度下降       | 正规方程                |
| ---------- | ------------------- |
| 需要选择学习率    | 不需要                 |
| 需要迭代求解     | 一次运算得出              |
| 特征数量较大可以使用 | 需要计算方程，时间复杂度高 O(n3) |

**模型选择**

- 小规模数据
  
  - LinearRegression (不能解决拟合问题)
  
  - 岭回归

- 大规模数据：SGDRegressor

### 回归性能评估

均方误差 (Mean Squared Error) MSE 评价机制

$$
MSE = \frac{1}{m}\sum _{i=1}^m (y_i-\bar{y})^2
$$

$y^i$ 为预测值，$\bar{y}$ 为真实值

`sklearn.metrics.mean_squared_error(y_true, y_pred)`

- 均方误差回归损失

- y_true: 真实值

- y_pred: 预测值

- return: 浮点数结果

### 案例: 波士顿房价预测

数据集介绍

![](./屏幕截图%202022-05-21%20215432.png)

![](./屏幕截图%202022-05-21%20220240.png)

流程：

1. 获取数据集

2. 划分数据集

3. 特征工程
   
   - 无量纲化 -- 标准化

4. 预估器流程
   
   - fit() ---- 模型
     
     coef_; intercept_

5. 模型评估

#### 使用正规方程

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# 1.获取数据
boston = load_boston()
# 2.划分数据集
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state = 22)
# 3.标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
# 4.预估器
estimator = LinearRegression() # 实例化
estimator.fit(x_train, y_train)
# 5.得出模型
print('正规方程权重系数为\n', estimator.coef_)
print('正规方程偏置为\n', estimator.intercept_)
# 6.模型评估
y_predict = estimator.predict(x_test)
print('预测房价:\n', y_predict)
error = mean_squared_error(y_test, y_predict)
print('正规方程-均方误差为:\n', error)
```

#### 使用梯度下降

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegression
# 1.获取数据
boston = load_boston()
# 2.划分数据集
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state = 22)
# 3.标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
# 4.预估器
estimator = SGDRegression() # 实例化
estimator.fit(x_train, y_train)
# 5.得出模型
print('梯度下降权重系数为\n', estimator.coef_)
print('梯度下降偏置为\n', estimator.intercept_)
# 6.模型评估
y_predict = estimator.predict(x_test)
print('预测房价:\n', y_predict)
error = mean_squared_error(y_test, y_predict)
print('梯度下降-均方误差为:\n', error)
```

## 欠拟合与过拟合

### 什么是过拟合与欠拟合

1. 欠拟合
   
   定义：一个假设在训练数据上不能获得更好的拟合，并且在测试数据集上也不能很好地拟合数据，此时认为这个假设出现了欠拟合的现象（模型过于简单）
   
   ![](./屏幕截图%202022-05-23%20224931.png)

2. 过拟合
   
   定义：一个假设在训练数据上能够获得比其他假设更好的拟合，但是在测试数据集上却不能很好地拟合数据，此时认为这个假设出现了过拟合的现象（模型过于复杂）
   
   训练集上表现的好，测试集上不好
   
   ![](./屏幕截图%202022-05-23%20225104.png)

![](./屏幕截图%202022-05-23%20225609.png)

### 原因及其解决方法

1. 欠拟合原因及其解决方法
   
   - 原因：学习到数据的特征过少
   
   - 解决方法：增加数据的特征数量

2. 过拟合原因及其解决方法
   
   - 原因：原始特征过多，存在一些嘈杂特征，模型过于复杂是因为模型尝试去兼顾各个测试数据点
   
   - 解决方法：正则化 
     
     尽量减小高次项特征的影响

#### 正则化类别

L1, <u>L2 正则化</u> (常用)

- L1 正则化
  
  - 作用：可以使得其中一些 w 的值直接为0，删除这个特征的影响
  
  - LASSO 回归
  
  - 损失函数
    
    损失函数 + 惩罚项 (w 的绝对值)

- L2 正则化
  
  - 作用：可以使得其中一些 w 很小，都接近于0，削弱某个特征的影响
  
  - 优点：越小的参数说明模型越简单，越简单的模型则越不容易产生过拟合现象
  
  - Ridge 回归 (岭回归)
  
  - 加入 L2 正则化后的损失函数
    
    $$
    J(w) = \frac{1}{2m}\sum_{i=1}^m (h_w(x_i)-y_i)^2+\lambda \sum_{j=1}^n w_j^2
    $$
    
    损失函数 + 惩罚系数 * 惩罚项
    
    > m 为样本数，n 为特征数

## 岭回归 -- 线性回归的改进

岭回归，其实也是一种线型回归，只不过在算法建立回归方程的时候，加上 L2 正则化的限制，从而达到解决过拟合的效果

### API

`sklearn.linear_model.Ridge(alpha = 1.0, fit_intercept = True, solver = 'auto', normalize = False)`

- **alpha: 正则化力度，也叫 $\lambda$**
  
  - $\lambda$ **取值**：0-1，1-10

- **solver: 会根据数据自动选择优化方法**
  
  - **sag: 如果数据集、特征都比较大，选择该随机梯度下降优化**

- normalize: 数据是否进行标准化
  
  - normalize = False: 可以在 fit 之前调用 preprocessing.StandardScaler 标准化数据

- Ridge.coef_: 回归权重

- Ridge.intercept_: 回归偏置

> Ridge 方法相当于 SGDRegressor(penalty = '12', loss = 'squared_loss'), 只不过 SGDRegressor 实现了一个普通的随机梯度下降学习，推荐使用 Ridge (实现了 SAG)

`sklearn.linear_model.RidgeCV(_BaseRidgeCV, RegressorMixin)`

- L2 正则化的线性回归，可以进行交叉验证

- coef_: 回归系数

### 正则化程度的变化

![](./屏幕截图%202022-05-24%20205000.png)

- 正则化力度越大，权重系数越小

- 正则化力度越小，权重系数越大

### 案例: 波士顿房价预测

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
# 1.获取数据
boston = load_boston()
# 2.划分数据集
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state = 22)
# 3.标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
# 4.预估器
estimator = Ridge() # 实例化
estimator.fit(x_train, y_train)
# 5.得出模型
print('岭回归权重系数为\n', estimator.coef_)
print('岭回归偏置为\n', estimator.intercept_)
# 6.模型评估
y_predict = estimator.predict(x_test)
print('预测房价:\n', y_predict)
error = mean_squared_error(y_test, y_predict)
print('岭回归-均方误差为:\n', error)
```

## 逻辑回归

逻辑回归 (Logistic Regression) 是机器学习中的一种分类模型，逻辑回归是一种分类算法，虽然名字中带有回归，但是它与回归之间有一定的联系；由于算法的简单和高效，在实际中应用非常广泛

### 应用场景

- 广告点击率

- 是否为垃圾邮件

- 是否患病

- 金融诈骗

- 虚假账号

根据以上例子，发现其中特点是都属于两个类别之间的判断；逻辑回归就是解决二分类问题的利器

### 原理

1. **输入**
   
   线性回归的输出 就是 逻辑回归的输入
   
   $$
   h(w) = w_1x_1 + w_2x_2 + w_3x_3 \cdots + b
   $$

2. **激活函数**
   
   sigmoid 函数
   
   $$
   g(\theta^T x) = \frac{1}{1+e^{-\theta^T x}}
   $$
   
   - 回归的结果输入到 sigmoid 函数当中
   
   - 输出结果：[0, 1] 区间中的一个概率值，默认为 0.5 为阈值
     
     > 上一步输出的结果映射到 sigmoid 函数中，出现的结果如果大于 0.5 就认为属于该类别，小于 0.5 则不属于该类别

### 损失以及优化

#### 损失函数

逻辑回归的损失，称之为**对数似然损失**

$$
cost(h_\theta (x), y) = \begin{cases}
- \log (h_\theta (x)) \qquad if\quad y = 1 \\
- \log (1 - h_\theta (x)) \qquad if \quad y = 0
\end{cases}
$$

![](./屏幕截图%202022-05-25%20223803.png)

![](./屏幕截图%202022-05-25%20224024.png)

**综合完整损失函数**

$$
cost(h_\theta (x), y) = \sum_{i=1}^m -y_i \log (h_\theta (x))
- (1-y_i) \log (1-h_\theta (x))
$$

![](./屏幕截图%202022-05-25%20224817.png)

#### 优化

同样使用梯度下降优化算法，去减少损失函数的值；这样去更新逻辑回归前面对应算法的权重参数，提升原本属于1类别的概率，降低原本是0类别的概率

### API

`sklearn.linear_model.LogisticRegression(solver = 'liblinear', penalty = '12', C = 1.0)`

- solver: 优化求解方式（默认开源的 liblinear 库实现，内部使用了坐标轴下降法来迭代优化损失函数）
  
  - sag: 根据数据集自动选择，随机平均梯度下降

- penalty: 正则化的种类

- C: 正则化力度

> 默认将类别数量少的当作正则

### 案例: 癌症分类预测-良 / 恶性乳腺癌肿瘤预测

数据描述

- 699条样本，共11列数据，第一列用于检索的 id，后9列分别是与肿瘤相关的医学特征，最后一列表示肿瘤类型的数值

- 包含16个缺失值，用 “?” 标出

流程分析

1. 获取数据
   
   读取的时候加上 names

2. 数据处理
   
   缺失值处理

3. 数据集划分

4. 特征工程
   
   无量纲化 - 标准化

5. 逻辑回归预估器

6. 模型评估

> jupyter notebook

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# 1.读取数据
path = 'http:'
column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Ce...']
data = pd.read_csv(path, names = column_name)
# 2.缺失值处理
# 2.1替换 np.nan
data = data.replace(to_replace = '?', value = np.nan)
# 2.2删除缺失样本
data.dropna(inplace = True)
# 3.数据集划分
# 筛选特征值和目标值
x = data.iloc[:, 1:-1]
y = data['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y)
# 4.标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
# 5.逻辑回归预估器
estimator = LogisticRegression()
estimator.fit(x_train, y_train)
# 6.模型评估
# 逻辑回归的模型参数 回归系数和偏置
print(estimator.coef_)
# 方法1：直接比对真实值和预测
y_predict = estimator.predict(x_test)
print('y_predict:\n', y_predict)
print('直接比对真实值和预测值:\n', y_test == y_predict)
# 方法2：计算准确率
score = estimator.score(x_test, y_test)
print('准确率为:\n', score)
```

## 回归评估方法

回归模型的性能评估不同于分类模型，虽然都是对照真实值进行评估，但由于回归模型的预测结果和真实值都是连续的，所以不能够求取 Precision, Recall 和 F1 值等评价指标；

平均绝对误差，均方误差和中值绝对误差的值越靠近0，模型性能越好，可解释方差值和 R 方值则越靠近1模型性能越好

1. ***平均绝对误差***
   
   - <u><em>最佳值</em></u> ：0.0
   
   - <u><em>sklearn 函数</em></u> ：`metrics.mean_absolute_error`

2. ***均方误差***
   
   - <u><em>最佳值</em></u> ：0.0
   
   - <u><em>sklearn 函数</em></u> ：`metrics.mean_squared_error`

3. ***中值绝对误差***
   
   - <u><em>最佳值</em></u> ：0.0
   
   - <u><em>sklearn 函数</em></u> ：`metrics.median_absolute_error`

4. ***可解释方差值***
   
   - <u><em>最佳值</em></u> ：1.0
   
   - <u><em>sklearn 函数</em></u> ：`metrics.explained_variance_score`

5. ***R 方值***
   
   - <u><em>最佳值</em></u> ：1.0
   
   - <u><em>sklearn 函数</em></u> ：`metrics.r2_score`

# 无监督学习

**什么是无监督学习**

无目标值 即为 无监督学习

> - 一家广告平台需要根据相似的人口学特征和购买习惯将美国人口分成不同的小组，以便广告客户可以通过有关联的广告接触到他们的目标客户
> 
> - Airbnb 需要将自己的房屋清单分组成不同的社区，以便用户能更轻松地查阅这些清单
> 
> - 一个数据科学团队需要降低一个大型数据集的维度的数量，以便简化建模和降低文件的大小

**无监督学习包含算法**

- 聚类
  
  - K-means (K 均值聚类)

- 降维
  
  - PCA

## K-means 算法

### 概述

K -- 超参数

- 看需求

- 调节超参数

效果图

![](./屏幕截图%202022-06-02%20003738.png)

步骤

1. 随机设置 K 个特征空间内的点作为初始的聚类中心

2. 对于其他每个点计算到 K 个中心的距离，未知的点选择最近的一个聚类中心作为标记类别

3. 接着对着标记的聚类中心之后，重新计算出每个聚类的新中心点（平均值）

> 如果计算得出的新中心点与原中心点一样，那么结束，否则重新进行第二步过程

步骤图解

![](./屏幕截图%202022-06-02%20005233.png)

### API

`sklearn.cluster.KMeans(n_clusters = 8, init = 'k-means++')`

```python
 KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001,   
         precompute_distances='auto', verbose=0, random_state=None,  
         copy_x=True, n_jobs=None, algorithm='auto')
```

- n_clusters: 开始的聚类中心数量

- init: 初始化方法，默认为 'k-means++'

- labels_: 默认标记的类型，可以和真实值比较 (不是值比较)

![](./屏幕截图%202022-08-09%20221954.png)

```python
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 3)
kmeans.fit(data_new_1)
labels = kmeans.predict(data_new_1)
plt.scatter(data_new_1[:, 0], data_new_1[: , 1], c = labels, s = 10, cmap = 'plasma')
plt.show()
```

### 性能评估指标

1. 轮廓系数
   
   $$
   SC_i = \frac{b_i - a_i}{\max (b_i,a_i)}
   $$
   
   对于每个点 $i$ 为已聚类数据中的样本，$b_i$ 为 $i$ 到其它族群的所有样本的距离（外部距离）最小值，$a_i$ 为 $i$ 到本身簇的距离（内部距离）平均值；最终计算出所有的样本点的轮廓系数平均值

2. 轮廓系数分析
   
   ![](./屏幕截图%202022-06-02%20151812.png)

3. 结论
   
   若 $b_i >> a_i$；趋近于1效果越好，$b_i << a_i$；趋近于-1，效果不好；轮廓系数的值是介于 [-1,1]，越趋近于1代表内聚度和分离度都相对较优

#### API

`sklearn.metrics.sihouette_score(X, labels)`

- X: 特征值

- labels: 被聚类标记的目标值

#### 评价指标

1. ***ARI 评价法 (兰德系数)***
   
   - <u>*需要真实值*</u>
   
   - <u>*最佳值*</u> ：1.0
   
   - <u>*sklearn 函数*</u> ：`adjusted_rand_score`

2. ***AMI 评价法 (互信息)***
   
   - <u>*真实值*</u> ：需要
   
   - <u>*最佳值*</u> ：1.0
   
   - <u>*sklearn 函数*</u> ：`adjusted_mutual_info_score`

3. ***V-measure 评分***
   
   - <u>*真实值*</u> ：需要
   
   - <u>*最佳值*</u> ：1.0
   
   - <u>*sklearn 函数*</u> ：`completeness_score`

4. ***FMI 评价法***
   
   - <u><em>真实值</em></u> ：需要
   
   - <u><em>最佳值</em></u> ：1.0
   
   - <u><em>sklearn 函数</em></u> ：`fowlkes_mallows_score`

5. ***轮廓系数评价法***
   
   - <u><em>真实值</em></u> ：不需要
   
   - <u><em>最佳值</em></u> ：畸变程度最大
   
   - <u><em>sklearn 函数</em></u> ：`silhouette_score`

6. ***Calinski-Harabasz 指数评价法***
   
   - <u><em>真实值</em></u> ：不需要
   
   - <u><em>最佳值</em></u> ：相较较大
   
   - <u><em>sklearn 函数</em></u> ：`calinski_harabaz_score`

---

```python
'''
寻找最优的聚类簇
'''
from sklearn.metrics import silhouette_score
for k in range(2, 9):
    model = KMeans(n_clusters = k).fit(data['data'])
    print(k, silhoutte_score(data['data'], model.labels_))
```

### 优缺点

- 特点分析：采用迭代式算法，直观易懂并且非常使用

- 缺点：容易收敛到局部最优解（多次聚类）

> 聚类一般做在分类之前

```python
import numpy as np
import pandas as pd
from sktime.datasets import load_arrow_head
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == '__main__':

    data = pd.read_excel('./插值后的高频指标.xlsx')
    X, y =load_arrow_head(return_X_y=True)
    X_train, X_test, y_train,y_test =train_test_split(X, y)
    classifier=TimeSeriesForestClassifier()
    classifier.fit(X_train, y_train)
    y_pred= classifier.predict(X_test)
    accuracy_score(y_test, y_pred)
    print(X_test)
```
