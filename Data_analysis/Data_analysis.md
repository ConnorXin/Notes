# NumPy

## Numpy 简介

Numpy 是使用 Python 进行科学计算的基础包。它包含如下内容：

- 一个强大的 N 维数组对象

- 复杂的(广播)功能

- 用于集成 C/C++ 和 Fortran 代码的工具

- 有用的线性代数，傅里叶变换和随机数功能

除了明显的科学用途外，NumPy 还可以用作通用数据的高效多维容器。可以定义任意数据类型。这使 NumPy 能够无缝快速地与各种数据库集成

### numpy 的特点

- numpy 能提供类似于 C 的数组的结构 ndarray，一维的常称为向量 vector，多维的称之为矩阵 matrix。

- numpy 的数组(向量、矩阵)的各类运算要比 Python 里类似结构类型 list 列表运算处理速度要快很多！

## Numpy 和原生 Python 的对比

语法实现对比

- **使用 Python 原生语法**
  
  ```python
  import numpy as np
  def python_sum(n):
      a = [i**2 for i in range(n)]
      b = [i**3 for i in range(n)]
      c = []
      for i in range(n):
          c.append(a[i] + b[i])
      return c
  ```

- **使用 Numpy 实现**
  
  ```python
  def numpy_sum(n):
      a = np.arange(n) ** 2
      b = np.arange(n) ** 3
      return a+b
  ```

## Numpy 的核心 array 对象

**array 对象的背景**

* Numpy 的核心数据结构，就叫做 array 就是数组，array 对象可以是一维数组，也可以是多维数组；
* Python 的 List 也可以实现相同的功能，但是 array 比 List 的优点在于性能好、包含数组元数据信息、大量的便捷函数；
* Numpy 成为事实上的 Scipy、Pandas、Scikit-Learn、Tensorflow、PaddlePaddle 等框架的“通用底层语言”
* Numpy 的 array 和 Python 的 List 的一个区别，是它元素必须都是同一种数据类型，比如都是数字int类型，这也是 Numpy 高性能的一个原因；

**array 本身支持的大量操作和函数**

- 直接逐元素的加减乘除等算数操作
- 更好用的面向多维的数组索引
- 求 sum/mean 等聚合函数
- 线性代数函数，比如求解逆矩阵、求解方程组

## 数组属性

| 属性       | 说明                                       |
|:-------- |:---------------------------------------- |
| ndim     | 返回 int ;表示数组的维数                          |
| shape    | 返回 tuple ;表示数组的尺寸，对于 n 行 m 列的矩阵，形状为(n,m) |
| size     | 返回 int ;表示数组的元素总数，等于数组形状的乘积              |
| dtype    | 返回 data-type ;描述数组中元素的类型                 |
| itemsize | 返回 int ;表示数组的每个元素的大小(以字节为单位)             |

- **ndim**
  
  ```python
  X = np.array(
      [
      [1, 2, 3, 4],
      [5, 6, 7, 8]
  ])
  X.ndim
  ```
  
  Out: 2

- **shape**
  
  ```python
  x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
  X = np.array(
      [
      [1, 2, 3, 4],
      [5, 6, 7, 8]
  ])
  x.shape
  X.shape
  ```
  
  Out: (8,);(2,4)

- **size**
  
  ```python
  x.size
  X.size
  ```
  
  Out: 8;8

- **dtype**
  
  ```python
  x.dtype
  X.dtype
  ```
  
  Out: dtype('int32');dtype('int32')

## 创建 array 的方法

### 1. 从 Python 的列表 List 和嵌套列表创建 array

```python
import numpy as np
# 创建一个一维数组，也就是 Python 的单元素 List
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
x
```

Out: array([1, 2, 3, 4, 5, 6, 7, 8])

```python
# 创建一个二维数组，也就是 Python 的嵌套 List
X = np.array(
    [
    [1, 2, 3, 4],
    [5, 6, 7, 8]
]
)
X
```

Out: array([[1, 2, 3, 4],

                     [5, 6, 7, 8]])

### 2. 使用预定函数 arange、ones/ones_like、zeros/zeros_like、 empty/empty_like、full/full_like、eye 等函数创建

- **arange**
  
  `arange([start,] stop[, step,], dtype=None)`
  
  ```python
  import numpy as np
  a = np.arange(15).reshape(3,5)
  a
  ```
  
  Out:
  
  array([[ 0,  1,  2,  3,  4],
  [ 5, 6, 7, 8, 9],
  [10, 11, 12, 13, 14]])
  
  ```python
  np.arange(2, 10, 2)
  ```
  
  Out: array([2, 4, 6, 8])

- **ones**
  
  `np.ones(shape, dtype=None, order='C')`
  
  <u>
  shape : int or tuple of ints Shape of the new array, e.g., (2, 3) or 2.
  </u>
  
  ```python
  np.ones(10)
  ```
  
  Out: array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
  
  ```python
  np.ones((2,3))
  ```
  
  Out:
  
  array([[1., 1., 1.],
  [1., 1., 1.]])

- **ones_like**
  
  创建形状相同的数组
  
  `ones_like(a, dtype=float, order='C')`
  
  ```python
  np.ones_like(x)
  np.ones_like(X)
  ```
  
  Out:
  
  array([1, 1, 1, 1, 1, 1, 1, 1])
  
  array([[1, 1, 1, 1],
  
  [1, 1, 1, 1]])

- **zeros**
  
  `np.zeros(shape, dtype=None, order='C')`
  
  ```python
  np.zeros(10)
  ```
  
  Out: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
  
  ```python
  np.zeros((2,4))
  ```
  
  Out: 
  
  array([[0., 0., 0., 0.],
  [0., 0., 0., 0.]])

- **zeros_like**
  
  与ones_like同理

- **empty**
  
  `empty(shape, dtype=float, order='C')`
  
  > 注意：数据是未初始化的，里面的值可能是随机值不要用
  
  ```python
  np.empty(10)
  ```
  
  Out: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

- **empty_like**
  
  `empty_like(prototype, dtype=None)`
  
  ```python
  np.empty_like(x)
  ```
  
  Out: array([120,  46, 110, 100, 105, 109, 101,   0])
  
  ```python
  np.empty_like(X)
  ```
  
  Out:
  
  array([[0, 0, 0, 0],
  [0, 0, 0, 0]])

- **full**
  
  `np.full(shape, fill_value, dtype=None, order='C')`
  
  ```python
  np.full(10, 666)
  ```
  
  Out: array([666, 666, 666, 666, 666, 666, 666, 666, 666, 666])
  
  ```python
  np.full((2,4), 333)
  ```
  
  Out: array([[333, 333, 333, 333],

- **full_like**
  
  <u>np.full_like(a, fill_value, dtype=None)</u>
  
  ```python
  np.full_like(x, 666)
  ```
  
  Out: array([666, 666, 666, 666, 666, 666, 666, 666])

### 3. 生成随机数的 np.random 模块构建

`randn(d0, d1, ..., dn)`

```python
np.random.randn()
```

Out: -0.24298859935537268

```python
np.random.randn(3)
```

Out: array([-0.630065  , -1.15991332, -0.69356883])

```python
np.random.randn(3, 2)
```

Out: array([[-0.62787255, -0.01054911],
       [ 0.8695556 , -1.39004067],
       [ 0.16209569,  2.08941735]])

```python
np.random.randn(3, 2, 4)
```

Out:

array([[[ 0.97915519, -0.04827654, -0.24805357, -1.18815335],
        [ 0.81855172,  0.531464  , -0.0455545 ,  0.7775499 ]],

   [[-1.6144118 ,  0.11402734,  0.106872  ,  0.80540136],
    [-0.49410333,  0.13846393,  0.4897993 , -1.34085886]],

   [[ 0.25693092,  0.4697215 , -1.00145574,  0.33612804],
    [ 0.68994321,  0.17245859,  0.49834578,  1.37301829]]])

### 4. 更多API包括：linspace, logspace, diag, eye, randint, randn, random

## 通函数

> NumPy 提供熟悉的数学函数，例如 sin，cos 和 exp 等。
> 
> 在 NumPy 中，这些被称为“通函数”（ufunc）。
> 
> 在 NumPy 中，这些函数在数组上按元素进行运算，产生一个数组作为输出。

np.mean(data)     # 均值
np.std(data)          # 标准差
np.var(data)          # 方差
np.min(data)         # 最小值
np.max(data)        # 最大值
np.argmax(t, axis) # 最大值的位置
np.argmin(t, axis) # 最小值的位置
np.ptp(data) # 极值

**可对某行或某列的数据统计；方法是在函数中添加一个 axis 参数 0表示每列 1表示每行**

## Numpy 多维数组运算

ndarray 对象的运算效率极高，不用编写循环，运算直接应用在元素级上。

> ```python
> import numpy as np
> zarten_1 = np.array([[1, 2, 3],[4, 5, 6]])
> zarten_2 = zarten_1 * 2
> print(zarten_1)
> print('\n')
> ptint(zarten_2)
> ```
> 
> Out:
> 
> [[1 2 3]
>  [4 5 6]]
> [[ 2  4  6]
>  [ 8 10 12]]

> ```python
> zarten_1 = np.array([[1, 2, 3],[4, 5, 6]])
> zarten_2 = np,array([7, 8, 9])
> print(zarten_1 + zarten_2)
> ```
> 
> Out:
> 
> [[ 8 10 12]
>  [11 13 15]]

> ```python
> zarten_1 = np.array([[1, 2, 3],[4, 5, 6]])
> zarten_2 = np,array([7, 8, 9])
> print(zarten_1 > zarten_2)
> ```
> 
> Out:
> 
> [[False False False]
>  [ True  True  True]]
> 
> *多维数组之间比较，会产生一个布尔类型的数组*

**一元运算**

> ```python
> zarten = np.array([1, 2, 3, 4, 5])
> print(np.sqrt（zarten))
> print(np.sin(zarten))
> ```
> 
> Out:
> 
> [1.         1.41421356 1.73205081 2.         2.23606798]
> [ 0.84147098  0.90929743  0.14112001 -0.7568025  -0.95892427]
> 
> ![](D:\HANSHAN\Data%20Analysis\Numpy\1.jpg)

**二元运算**

> ```python
> zarten_1 = np.array([1, 2, 3, 4, 88])
> zarten_2 = np.array([6, 7, 8, 9, 10])
> print(np.maximum(zarten_1, zarten_2))
> print(np.add(1,2))
> ```
> 
> Out:
> 
> [ 6  7  8  9 88]
> 3
> 
> ![](D:\HANSHAN\Data%20Analysis\Numpy\2.jpg)

**集合运算**

numpy 中提供一些针对一维的集合运算，如 unique 函数、in1d 函数等

> unique 函数 跟 Python 中的 set 集合类似，但 unique 函数的输出结果是排好序的
> 
> ```python
> zarten_1 = np.array([5, 3, 6, 5, 4, 2, 1, 2])
> print(zarten_1)
> print('\n')
> print(np.unique(zarten_1))
> ```
> 
> Out: [5 3 6 5 4 2 1 2]
> 
> [1 2 3 4 5 6]

> in1d() 函数返回一个布尔型数组，用于判断一个数组的元素是否有在另一个数组内
> 
> ```python
> zarten_1 = np.array([5, 3, 6, 5, 4, 2, 1, 2])
> print(zarten_1)
> print('\n')
> print(np.in1d(zarten_1,[5, 3]))
> ```
> 
> Out: [5 3 6 5 4 2 1 2]
> [ True  True False  True False False False False]

## Numpy对数组按索引查询

三种索引方法

### 1. 基础索引

- 一维数组
  
  和 Python 的 list 一样

- 二维数组
  
  ```python
  X  = np.arange(20).reshape(4,5)
  X
  ```
  
  Out:
  
  array([[ 0,  1,  2,  3,  4],
  
  [ 5,  6,  7,  8,  9],
  [10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]])
  
  ```python
  # 分别用行坐标、列坐标实现行列筛选
  X[0, 0]
  ```
  
  Out: 0
  
  `X[-1, 2]`   Out: 17
  
  ```python
  # 可以省略后续索引值，返回的数据是降低一个维度的数组
  # 这里的2其实是要筛选第2行
  X[2]
  ```
  
  Out: array([10, 11, 12, 13, 14])
  
  ```python
  # 筛选多行
  X[:-1]
  ```
  
  Out:
  
  array([[ 0, 1, 2, 3, 4],
  [ 5, 6, 7, 8, 9],
  [10, 11, 12, 13, 14]])
  
  ```python
  # 筛选多行 多列
  X[:2, 2:4]
  ```
  
  Out:
  
  array([[2, 3],
  [7, 8]])
  
  ```python
  # 筛选所有行 多列
  X[:,2]
  ```
  
  Out: array([2, 7, 12, 17])

- 切片可以赋值
  
  ```python
  x[2:4] = 666
  X[:1, :2] = 666
  x
  X
  ```
  
  Out:
  
  array([  0,   1, 666, 666,   4,   5,   6,   7,   8,   9])
  
  array([[666, 666,   2,   3,   4],
  [  5,   6,   7,   8,   9],
  [ 10,  11,  12,  13,  14],
  [ 15,  16,  17,  18,  19]])

### 2. 神奇索引

用整数数组进行的索引叫神奇索引

- 一维数组
  
  ```python
  x = np.arange(10)
  x
  x[[3, 4, 7]]
  ```
  
  Out:
  
  array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  
  array([3, 4, 7])
  
  ```python
  index = np.array([[0, 2],[1, 3]])
  x[index]
  ```
  
  Out: array([[0, 2],
  
     [1, 3]])
  
  > *实例：获取数组中最大的前 N 个数字*
  > 
  > ```python
  > # 随机生成1到100之间的10个数字
  > arr = np.random.randint(1, 100, 10)
  > arr
  > ```
  > 
  > Out: array([92, 16, 12, 57,  1, 66, 26, 55, 68, 84])
  > 
  > ```python
  > # arr.argsort() 会返回排序后的索引index
  > # 取最大值对应的3个下标
  > arr.argsort()[-3:]
  > ```
  > 
  > Out: array([8, 9, 0], dtype=int64)
  > 
  > `arr[arr.argsort()[-3:]]`   Out: array([68, 84, 92])

- 二维数组
  
  ```python
  X = np.arange(20).reshape(4,5)
  X
  ```
  
  Out:
  
  array([[ 0,  1,  2,  3,  4],
  [ 5,  6,  7,  8,  9],
  [10, 11, 12, 13, 14],
  [15, 16, 17, 18, 19]])
  
  ```python
  # 筛选多行，列可以省略
  X[[0, 2]]
  ```
  
  Out:
  
  array([[ 0,  1,  2,  3,  4],
  [10, 11, 12, 13, 14]])
  
  `X[[0, 2], :]`  
  
  Out:
  
  array([[ 0,  1,  2,  3,  4],
  [10, 11, 12, 13, 14]])
  
  ```python
  # 筛选多列，行不能省略
  X[:, [0, 2, 3]]
  ```
  
  Out:
  
  array([[ 0,  2,  3],
  [ 5,  7,  8],
  [10, 12, 13],
  [15, 17, 18]])
  
  ```python
  # 同时指定行列-列表
  # 返回的是[(0, 1), (2, 3), (3, 4)]位置的数字
  X[[0, 2, 3], [1, 3, 4]]
  ```
  
  Out: array([ 1, 13, 19])
  
  ```python
  # 同时指定行列列表
  # 创建一个1*2的矩阵
  X[np.array([0, 2]), np.array([1, 3])]
  X[[0, 2], [1, 3]]
  ```
  
  Out: array([ 1, 13])
  
  ```python
  # 创建一个2*3矩阵
  X[np.array([[0, 2, 3], [1, 2, 3]]), np.array([[0, 1, 0], [0, 2, 3]])]
  a = (np.array([[0, 2, 3], [1, 2, 3]]), np.array([[0, 1, 2], [0, 2, 3]]))
  X[a]
  ```
  
  Out:
  
  array([[ 0, 11, 17],
  [ 5, 12, 18]])

### 3. 布尔索引

bool 数组可以通过直接指出保留的值（True）与舍弃的值（False），来构建输出的数组。

bool 数组的 shape 需要与被索引的数组 shape 严格对齐

- 一维数组
  
  ```python
  x = np,arange(10)
  x
  ```
  
  Out: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  
  `x > 5`  Out: array([False, False, False, False, False, False,  True,  True,  True,True])
  
  `x[x > 5]`  Out: array([6, 7, 8, 9])
  
  ```python
  # 实例 把一维数组进行01化处理
  # 比如把房价数字，变成“高房价”为1；“低房价”为0
  x[x <= 5] = 0
  x[x > 5] = 1
  x
  ```
  
  Out: array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
  
  ```python
  x = np.arange(10)
  x[x < 5] += 20
  x
  ```
  
  Out: array([20, 21, 22, 23, 24,  5,  6,  7,  8,  9])

- 二维数组
  
  ```python
  X = np.arange(20).reshape(4, 5)
  X
  ```
  
  Out:
  
  array([[ 0,  1,  2,  3,  4],
  [ 5,  6,  7,  8,  9],
  [10, 11, 12, 13, 14],
  [15, 16, 17, 18, 19]])
  
  `X > 5` 
  
  Out:
  
  array([[False, False, False, False, False],
  [False,  True,  True,  True,  True],
  [ True,  True,  True,  True,  True],
  [ True,  True,  True,  True,  True]])
  
  ```python
  # X > 5 的 boolean 数组，既有行又有列
  # 因此返回的是（行，列）一维结果
  X[X > 5]
  ```
  
  Out: array([ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
  
  ```python
  a = np.arange(15).reshape(3, 5)
  a
  ```
  
  Out:
  
  array([[ 0,  1,  2,  3,  4],
  [ 5,  6,  7,  8,  9],
  [10, 11, 12, 13, 14]])
  
  `a[np.array([True, False, True])]`
  
  Out:
  
  array([[ 0,  1,  2,  3,  4],
  [10, 11, 12, 13, 14]])
  
  ```python
  x = np.arange(30).reshape(2, 3, 5)
  x
  ```
  
  Out:
  
  array([[[ 0,  1,  2,  3,  4],
  [ 5,  6,  7,  8,  9],
  [10, 11, 12, 13, 14]],
  [[15, 16, 17, 18, 19],
  [20, 21, 22, 23, 24],
  
  [25, 26, 27, 28, 29]]])
  
  ```python
  b = np.array([[True, True, False], [False, True, True]])
  x[b]
  ```
  
  Out: 
  
  array([[ 0,  1,  2,  3,  4],
  [ 5,  6,  7,  8,  9],
  [20, 21, 22, 23, 24],
  [25, 26, 27, 28, 29]])
  
  ```python
  # 举例：怎样把第3列大于5的行筛选出来
  X[:, 3]
  ```
  
  Out: array([ 3,  8, 13, 18])
  
  `X[:, 3] > 6` 
  
  Out: array([False,  True,  True,  True])
  
  ```python
  # 这里是按照行进行的筛选
  X[X[:, 3] > 10]
  ```
  
  Out:
  
  array([[10, 11, 12, 13, 14],
  [15, 16, 17, 18, 19]])
  
  ```python
  X[X[:, 3] > 5] = 666
  X
  ```
  
  Out:
  
  array([[  0,   1,   2,   3,   4],
  [666, 666, 666, 666, 666],
  [666, 666, 666, 666, 666],
  [666, 666, 666, 666, 666]])

## 变换数组的形态

### 改变维度

在对数组进行操作时，经常要改变数组的维度。

在 Numpy 中，常用 reshape 函数改变数组的 “形状” ，也就是改变数组的维度。其参数为一个正整数元组，分别指定数组在每个维度上的大小。

reshape 函数在改变原始数据的形状的同时不改变原始数据的值。如果指定的维度和数组的元素数目不吻合，则函数将抛出异常。

```python
# 创建一维数组
arr = np.arange(12)
print('创建的一维数组为：', arr)
# 设置数组的形状
print('新的一维数组为：', arr.reshape(3, 4))
# 查看数组维度
print('数组维度为；', arr.reshape(3, 4).ndim)
```

Out: 创建的一维数组为： [ 0  1  2  3  4  5  6  7  8  9 10 11]
新的一维数组为： [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
数组维度为； 2

增加维度的方法

**np.newaxis** 关键字，使用索引的语法给数组添加维度

> ```python
> import numpy as np
> arr = np.arange(5)
> arr.shape
> ```
> 
> Out: (5,)
> 
> `arr[np.newaxis, :]`
> 
> Out: array([[0, 1, 2, 3, 4]])
> 
> `arr[np.newaxis, :].shape`
> 
> Out: (1, 5)
> 
> ```python
> arr[:, np.newaxis]
> arr[:, np.newaxis].shape
> ```
> 
> Out: (5, 1)

**np.expand_dims(arr, axis)** 和 np.newaxis 实现一样的功能, 给 arr 在 axis 位置添加维度

> ```python
> arr = np.arange(5)
> arr.shape
> ```
> 
> Out: (5,)
> 
> `np.expand_dims(arr, axis = 0)`
> 
> Out: array([[0, 1, 2, 3, 4]])
> 
> `np.expand_dims(arr, axis = 0).shape`
> 
> Out: (1, 5)
> 
> ```python
> np.expand_dims(arr, axis = 1)
> np.expand_dims(arr, axis = 1).shape
> ```
> 
> Out: (5, 1)

**np.reshape(a, newshape)** 给一个维度设置为 1 完成升维

> ```python
> arr = np.arange(5)
> arr.shape
> ```
> 
> Out: (5, )

### 展平数组

在 Numpy 中，可以使用 rabel 函数、flatten 函数完成数组展平工作。

目的是将任意形状的数组扁平化，变为一维数组。

**ravel()** 返回的是 原始数组的视图，原始数组本身并没有发生变化。

**flatten()** 会重新分配内存，完成一次从原始数组到新内存空间的深拷贝，但原始数组并没有发生任何变化。

```python
arr = np.arange(12).reshape(3, 4)
print('创建的二维数组为：', arr)
print('数组展平后为：', arr.ravel())
print('数组展平后为：', arr.ravel('F'))
```

Out: 创建的二维数组为： [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
数组展平后为： [ 0  1  2  3  4  5  6  7  8  9 10 11]
数组展平后为： [ 0  4  8  1  5  9  2  6 10  3  7 11]

```python
# 横向展平
print('数组展平为：', arr.flatten())
# 纵向展平
print('数组展平为：', arr.flatten('F'))
```

Out: 数组展平为： [ 0  1  2  3  4  5  6  7  8  9 10 11]
数组展平为： [ 0  4  8  1  5  9  2  6 10  3  7 11]

### 组合数组（堆叠数组）

除了可以改变数组 “形状” 外，Numpy 也可以对数组进行组合（堆叠）

组合主要有**横向组合**与**纵向组合**

使用 **hstack 函数**、**vstack 函数**以及 **concatenate 函数**来完成数组的组合

横向组合是将 ndarray 对象构成的元组作为参数，传给 hstack 函数

纵向组合同样是将 ndarray 对象构成的元组作为参数，传给 vstack 函数

concatenate 函数可以实现数组的横向组合和纵向组合

```python
arr1 = np.arange(12).reshape(3, 4)
print('创建的数组1为', arr1)
arr2 = arr1*3
print('创建的数组2为', arr2)
print('横向组合为', np.hstack((arr1,arr2)))
print('纵向组合', np.vstack((arr1,arr2)))
```

Out: 创建的数组1为 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
创建的数组2为 [[ 0  3  6  9]
 [12 15 18 21]
 [24 27 30 33]]
横向组合为 [[ 0  1  2  3  0  3  6  9]
 [ 4  5  6  7 12 15 18 21]
 [ 8  9 10 11 24 27 30 33]]
纵向组合 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [ 0  3  6  9]
 [12 15 18 21]
 [24 27 30 33]]

> **np.concatenate()** 需要注意的点：
> 
> 参数是列表：一定要加中括号或小括号
> 
> 维度必须相同
> 
> 形状相符
> 
> **方向默认是 shape 这个 tuple 的第一个值所代表的维度方向可通过 axis 参数改变级联的方向，默认是 0，(0 表示列相连，行发生改变，表示 Y 轴的事情，1 表示列相连，表示的 X 轴的事情)**
> 
> ```python
> print('横向组合为：', np.concatenate((arr1, arr2), axis = 1))
> print('纵向组合为：', np.concatenate((arr1, arr2), axis = 0))
> ```
> 
>  Out: 横向组合为： [[ 0  1  2  3  0  3  6  9]
>  [ 4  5  6  7 12 15 18 21]
>  [ 8  9 10 11 24 27 30 33]]
> 纵向组合为： [[ 0  1  2  3]
>  [ 4  5  6  7]
>  [ 8  9 10 11]
>  [ 0  3  6  9]
>  [12 15 18 21]
>  [24 27 30 33]]

### 分割数组

Numpy 提供了 **hsplit、vsplit、dsplit** 和 **split 函数**，可以将数组分割成相同大小的子数组，或指定原数组中需要分割的位置

使用 **hsplit 函数**可以对数组进行横向分割

使用 **vsplit 函数**可以对数组进行横向分割

**split 函数**同样可以实现数组分割

在参数 axis =  1 时，可以进行横向分割；0 时，可以进行纵向分割

```python
arr = np.arange(16).reshape(4, 4)
print('创建的二维数组为\n', arr)
print('横向分割为\n', np.hsplit(arr, 2))
print('纵向分割为\n', np.vsplit(arr, 2))
```

Out: 

创建的二维数组为
 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]
横向分割为
 [array([[ 0,  1],
       [ 4,  5],
       [ 8,  9],
       [12, 13]]), array([[ 2,  3],
       [ 6,  7],
       [10, 11],
       [14, 15]])]
纵向分割为
 [array([[0, 1, 2, 3],
       [4, 5, 6, 7]]), array([[ 8,  9, 10, 11],
       [12, 13, 14, 15]])]

```python
print('横向分割为\n', np.split(arr2, 2, axis = 1))
print('纵向分割为\n', np.split(arr, 2, axis = 0))
```

Out: 横向分割为
 [array([[ 0,  3],
       [12, 15],
       [24, 27]]), array([[ 6,  9],
       [18, 21],
       [30, 33]])]
纵向分割为
 [array([[0, 1, 2, 3],
       [4, 5, 6, 7]]), array([[ 8,  9, 10, 11],
       [12, 13, 14, 15]])]

## 行列交换

```python
t = np.arange(12, 24).reshape(3, 4)
# 行交换
t[[1, 2], :] = t[[2, 1], :]
# 列交换
t[:, [0, 2]] = t[:, [2, 0]]
```

## 小练习

现在希望把之前案例中两个国家的数据方法一起来研究分析，同时保留国家的信息 (每条数据的国家来源)  ，应该怎么办

```python
import numpy as np
us_data = './youtube_video_data/US_video_data_numbers.csv'
uk_data = './youtube_video_data/GB_video_data_numbers.csv'
# 添加国家数据
us_data = np.loadtxt(us_data, delimiter = ',', dtype = int)
uk_data = np.loadtxt(uk_data, delimiter = ',', dtype = int)
# 添加国家信息
# 构造全为0的数据
zeros_data = np.zeros(us_data.shape[0], 1)
ones_data = np.ones(uk_data.shape[0], 1)
# 分别添加一列全为0，1的数组
us_data = np.hstack((us_data, zeros_data))
uk_data = np.hstack((uk_data, ones_data))
#拼接两组数据
final_data = np.vstack((uk_data, uk_data))
print(final_date)
```

## 广播机制

Numpy 操作通常在逐个元素的基础上在数组对上完成；在最简单的情况下，两个数组必须具有完全相同的形状，如下所示

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([2.0, 2.0, 2.0])
a * b
```

Out: array([2., 4., 6.])

**广播 (Broadcasting)** 描述了 numpy 如何在算数运算期间处理具有不同形状的数组。受某些约束的影响，较小的数组在较大的数组上 “广播”，以便它们具有兼容的形状

广播的原则：如果两个数组的后缘维度 (trailing dimension, 即从末尾开始算起的维度) 的轴长度相符，或其中的一方的长度为 1，则认为它们是广播兼容的；广播会在缺失或长度为 1 的维度上进行。

广播主要发生在两种情况

- 一种是两个数组的维数不相等，但是它们的后缘维度的轴长度相符；

- 另一种是有一方后缘维度的轴长度为 1

## 随机数函数

| 函数          | 说明                          |
| ----------- | --------------------------- |
| seed        | 确定随机数生成器的种子                 |
| permutation | 返回一个序列的随机排列或返回一个随机排列的范围     |
| shuffle     | 对一个序列就地随机排列                 |
| rand        | 产生均匀分布的样本值                  |
| randint     | 从给定的上下限范围内随机选取整数            |
| randn       | 产生正态分布 (平均值为 0；标准差为 1) 的样本值 |
| binomial    | 产生二项分布的样本值                  |
| normal      | 产生正态 (高斯) 分布的样本值            |
| beta        | 产生 Beta 分布的样本值              |
| chisquare   | 产生卡方分布的样本值                  |
| gamma       | 产生 Gamma 分布的样本值             |
| uniform     | 产生在 [0, 1) 中均匀分布的样本值        |

```python
print('numpy.random随机数生成')
import numpy.random as npr
x = npr.randint(0, 2, size = 100000)  # 抛硬币
print((x > 0).sum())                  # 正面的结果
print(npr.normal(size = (2, 2)))
# 正态分布随机数数组 shape = (2, 2)
```

Out:

numpy.random随机数生成
49720
[[ 0.21193991 -1.33266635]
 [-0.71381904  0.40960239]]

```python
# 关于随机数种子
import numpy as np
np.random.seed(10)
# 如果不使用随机数种子，每次执行获得的随机数是不固定的
# 加上随机数种子，每次执行生成的随机数是一样的
t = np.random.randint(0, 20, (3, 4))
print(t)
```

Out:

[[ 9  4 15  0]
 [17 16 17  8]
 [ 9  0 10  8]]

关于分布：均匀分布：在相同的大小范围内的出现概率是等可能的，也即是随机分布，每个位置上出现的概率是随机的

## 数组的转置

- `.transpose()`

- `.T`

## 数组的读写

**save、savez** 和 **load 函数**单个数组的保存

```python
a = np.arange(1, 13).reshape(3, 4)
print(a)
np.save('arr.npy', a)   # np.save('arr', a)
c = np.load('arr.npy')
print(c)
```

Out:

[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]

多个数组的保存

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.arange(0, 1.0, 0.1)
c = np.sin(b)  # 长度为10
print(c)
np.savez('result.npz', a = a, b = b, sin_array = c)
r = np.load('result.npz')
r['a'][1:]     # 数组 a
```

Out:

[0.         0.09983342 0.19866933 0.29552021 0.38941834 0.47942554
 0.56464247 0.64421769 0.71735609 0.78332691]

Out[5]:

array([[4, 5, 6]])

## 文件的读写

**savetxt 函数**及 **loadtxt 函数**

> **loadtxt**
> 
> `np.loadtxt(frame, dtypr = np.folat, delimiter = None, skiprows = 0, usecols = None, unpack = False)`
> 
> | 参数        | 解释                                                        |
> | --------- | --------------------------------------------------------- |
> | frame     | 文件、字符串或产生器，可以是 .gz 或 bz2 压缩文件                             |
> | dtype     | 数据类型，可选，CSV 的字符串以什么数据类型读入数组中，默认 np.float                  |
> | delimiter | 分隔字符串，默认是任何空格 改为 逗号                                       |
> | skiprows  | 跳过前 x 行，一般跳过第一行表头                                         |
> | usecols   | 读取指定的列，索引，元组类型                                            |
> | unpack    | (转置) 如果 True，读入属性将分别写入不同数组变量，False 读入数据只写入一个数组变量，默认 False |

```python
a = np.arange(0, 12, 0.5).reshape(4, -1)
np.savetxt('data//a1-out.txt', a)
# 缺省按照'%.18e'格式保存数据
np.loadtxt('data//a1-out.txt')
np.savetxt('data//a2-out.txt', a, fmt = '%d', delimiter = ',')
# 改为保存为整数，以逗号分隔
np.loadtxt('data//a2-out.txt', delimiter = ',')
# 读入的时候也需要指定逗号分隔
```

## Nan、Inf

NAN: not a number 表示不是一个数字

当我们读取本地文件为 float 的时候，如果有缺失，就会出现 nan

当做了一个不合适的计算的时候 (比如无穷大减去无穷大)

INF: infinity; inf 表示正无穷; -inf 表示负无穷

比如一个数字除以0，python 会直接报错；numpy 是以一个 inf 或 -inf

> nan/inf 都是浮点类型

```python
np.nan == np.nan
```

Out: False

以这个特性可以判断数组中的 nan 的个数

方法：np.count_nonzero()

```python
t = [1, 2, 0, nan, 40, 5]
t != t
```

Out: [False, False, False, True, False, False]

```python
t = [1, 2, 0, nan, 40, 5]
t != t
np.count_nonzero(t != t)
```

Out: 1

判断数据里哪些值是 nan 的方法：**np.isnan()**

> nan 和任何值计算都为 nan

**在一组数据中单纯的把 nan 替换成0，不合适；全部替换为0的话，替换之前的平均值如果大于0，替换之后的均值肯定会变小，所以一般的方式是把缺失的数值替换成均值 (中值) 或者直接删除有缺失值的一行**

数据里有 nan 怎么处理

```python
def fill_ndarray(t1):
    for i in range(t1.shape[1]):  # 遍历每一列
    temp_col = t1[:, i]  # 当前的一列
    nan_num = np.count_nonzero(temp_col != temp_col)
        if nan_num != 0:  # 不为0，说明当前这一列中有 nan
            temp_not_nan_col = temp_col(temp_col == temp_col]  # 当前一列不为 nan 的 array

            # 选中当前为 nan 的位置，把值赋值给不为 nan 的均值
            temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
    return t1

t1 = np.arange(12).reshape(3, 4).astype('int32')
t1[1, 2:] = np.nan
print(t1)
t1 = fill_ndarray(t1)
print(t1)
```

## 数组排序

### 1. numpy.sort

返回排序后数组的拷贝

> ```python
> arr = np.array([3, 2, 4, 5, 1, 9, 7, 8, 6])
> np.sort(arr)
> ```
> 
> Out: array([1, 2, 3, 4, 5, 6, 7, 8, 9])
> 
> ```python
> arr2 = arr.copy()
> arr2.sort()
> ```
> 
> Out: array([1, 2, 3, 4, 5, 6, 7, 8, 9])

### 2. array.sort

原地排序数组而不是返回拷贝

> ```python
> np.random.seed(42)
> arr = np.random.randint(1, 10, size = 10)  # 生成随机数
> print('创建的数组为', arr)
> 
> arr.sort()  # 直接排序
> print('排序后数组为', arr)
> 
> arr = np.random.randint(1, 10, size = (3, 3))
> print('创建的数组为', arr)
> 
> arr.sort(axis = 1)  # 沿着横轴排序
> print('排序后数组为', arr)
> 
> arr.sort(axis = 0)  # 沿着纵轴排序
> print('排序后数组为', arr)
> ```
> 
> Out:
> 
> 创建的数组为 [7 4 8 5 7 3 7 8 5 4]
> 排序后数组为 [3 4 4 5 5 7 7 7 8 8]
> 创建的数组为 [[8 8 3]
>  [6 5 2]
>  [8 6 2]]
> 排序后数组为 [[3 8 8]
>  [2 5 6]
>  [2 6 8]]
> 排序后数组为 [[2 5 6]
>  [2 6 8]
>  [3 8 8]] 

### 3. numpy.argsort

间接排序，返回的是排序后的数字索引

> ```python
> arr = np.array([2, 3, 6, 8, 0, 7])
> print('创建的数组为', arr)
> print('排序后数组为', arr.argsort())
> # 返回值为重新排序值的下标
> ```
> 
> Out:
> 
> 创建的数组为 [2 3 6 8 0 7]
> 排序后数组为 [4 0 1 2 5 3]
> 
> ```python
> a = np.array([3, 2, 6, 4, 5])
> b = np.array([50, 30, 40, 20, 10])
> c = np.array([400, 300, 600, 100, 200])
> d = np.lexsort((a, b, c))
> # lexsort 函数只接受一个参数，即 (a, b, c)
> # 多个键值排序是按照最后一个传入数据计算的
> print('排序后数组为', list(zip(a[d], b[d], c[d])))
> ```
> 
> Out:
> 
> 排序后数组为 [(4, 20, 100), (5, 10, 200), (2, 30, 300), (3, 50, 400), (6, 40, 600)]

## 去重和重复数据

- 去重
  
  > ```python
  > names = np.array(['小明', '小黄', '小花', '小明', '小花', '小花', '小白'])
  > print('创建的数组为', names)
  > print('去重后的数组为', np.unique(names))
  > # 跟 np.unique 等价的 Python 代码实现过程
  > print('去重后的数组为', sorted(set(names)))
  > ```
  > 
  > Out:
  > 
  > 创建的数组为 ['小明' '小黄' '小花' '小明' '小花' '小花' '小白']
  > 去重后的数组为 ['小明' '小白' '小花' '小黄']
  > 去重后的数组为 ['小明', '小白', '小花', '小黄']
  > 
  > ```python
  > ints = np.array([1, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9, 10])
  > print('创建的数组为', ints)
  > print('去重后的数组为', np.unique(ints))
  > ```
  > 
  > Out:
  > 
  > 创建的数组为 [ 1  2  3  4  4  5  6  6  7  8  8  9 10]
  > 去重后的数组为 [ 1  2  3  4  5  6  7  8  9 10]

- 重复数据
  
  > **tile 函数**的格式为
  > 
  > `numpy.tile(array, repeats)`
  > 
  > array: 需要重复的数组
  > 
  > repeats: 指定重复的次数
  > 
  > ```python
  > arr = np.arange(5)
  > print('创建的数组为', arr)
  > print('重复后的数组为', np.tile(arr, 3))
  > ```
  > 
  > Out:
  > 
  > 创建的数组为 [0 1 2 3 4]
  > 重复后的数组为 [0 1 2 3 4 0 1 2 3 4 0 1 2 3 4]
  > 
  > **repeat 函数**的格式为
  > 
  > `numpy.repeat(array, repeats, axis = none)`
  > 
  > array: 需要重复的数组
  > 
  > repeats: 指定重复的次数
  > 
  > axis: 指定沿着哪个轴进行重复
  > 
  > ```python
  > np.random.seed(42)
  > arr = np.random.randint(0, 10, size = (3, 3))
  > print('创建的数组为', arr)
  > print('重复后的数组为', np.repeat(arr, 2, axis = 0))
  > print('重复后数组为', np.repeat(arr, 2, axis = 1))
  > ```
  > 
  > Out:
  > 
  > 创建的数组为 [[6 3 7]
  >  [4 6 9]
  >  [2 6 7]]
  > 重复后的数组为 [[6 3 7]
  >  [6 3 7]
  >  [4 6 9]
  >  [4 6 9]
  >  [2 6 7]
  >  [2 6 7]]
  > 重复后数组为 [[6 6 3 3 7 7]
  >  [4 4 6 6 9 9]
  >  [2 2 6 6 7 7]]

## numpy to list

```python
import numpy as np

# 2d array to list
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(f'NumPy Array:\n{arr}')

list1 = arr.tolist()

print(f'List: {list1}')
```

## array 不以科学计数法输出显示

`np.set_printoptions(suppress = True)`

## 案例

- 英国和美国各自 youtube 1000 的数据结合之前的 matplotlib 绘制出各自的评论数量的直方图

- 希望了解英国的 youtube 中视频的评论数和喜欢数的关系，应该如何改图

# Pandas

## Pandas 简介

在 Pandas 没有出现之前，Python 在数据分析任务中主要承担着数据采集和数据预处理的工作，但是这对数据分析的支持十分有限，并不能突出 Python 简单、易上手的特点；Pandas 的出现使得 Python 做数据分析的能力得到了大幅度提升，它主要实现了数据分析的五个重要环节：

- 加载数据

- 整理数据

- 操作数据

- 构建数据模型

- 分析数据

**主要特点**

- 它提供了一个简单、高效、带有默认标签 (也可以自定义标签) 的 DataFrame 对象

- 能够快速得从不同格式的文件中加载数据 (比如 Excel、CSV、SQL 文件)，然后将其转换为可处理的对象

- 能够按数据的行、列标签进行分组，并对分组后的对象执行聚合和转换操作

- 能够很方便地实现数据归一化操作和缺失值处理

- 能够很方便地对 DataFrame 的数据列进行增加、修改或者删除的操作

- 能够处理不同格式的数据集，比如矩阵数据、异构数据表、时间序列等

- 描述了多种处理数据集的方式，比如构建子集、切片、过滤、分组以及重新排序等

## Series

**Series 结构**，也称 Series 序列，是 Pandas 常用的数据结构之一，它是一种类似于一维数组的结构，由一组数据值 (value) 和一组标签 (即索引) 组成，其中标签与数据值之间是一一对应的关系

Series 可以保存任何数据类型，比如整数、字符串、浮点数、Python 对象等；它的标签默认为整数，从 0 开始依次递增；Series 结构图如图所示

<img title="" src="file:///D:/HANSHAN/Data Analysis/Pandas/Picture/屏幕截图 2022-04-05 222303.png" alt="" width="516">

- **创建 Series 对象**
  
  `pd.Series(data = None, index = None, dtype = None, name = None, copy = False, fastpath = False)`
  
  | 参数名称  | 描述                                    |
  | ----- | ------------------------------------- |
  | data  | 输入的数据，可以是列表、常量、ndarray 数组等            |
  | index | 索引值必须是唯一的，如果没有传递索引值，则默认为 np.arange(n) |
  | dtype | dtype 表示数据类型，如果没有提供，则会自动判断得出          |
  | copy  | 表示对 data 进行拷贝，默认为 False               |
  
  - 通过列表创建 Series
    
    ```python
    obj = pd.Series([1, -2, 3, -4])
    obj
    ```
    
    Out:
    
    0    1
    1   -2
    2    3
    3   -4
    dtype: int64
  
  - 通过字典创建 Series
    
    可以把 dict 作为输入数据；如果没有传入索引时会按照字典的键来构造索引；反之，当传递了索引时需要将索引标签与字典中的值一一对应
    
    ```python
    data = {'Ohio':35000, 'Texas':71000, 'Oregon':16000, 'Utah':5000}
    obj3 = pd.Series(data)
    obj3
    ```
    
    Out:
    
    Ohio      35000
    Texas     71000
    Oregon    16000
    Utah       5000
    dtype: int64
    
    *键值和指定的索引不匹配时*
    
    ```python
    data = {'a':100, 'b':200, 'e':300}
    obj = pd.Series(data, index = ['b', 'c', 'd', 'e', 'a'])
    obj
    ```
    
    Out:
    
    b    200.0
    c      NaN
    d      NaN
    e    300.0
    a    100.0
    dtype: float64
    
    > 当传递的索引值无法找到与其对应的值时，使用 NaN (非数字) 填充
    
    *Series 索引的修改*
    
    ```python
    obj = pd.Series([4,7,-3,2])
    print("修改前：\n",obj)
    obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
    print("修改后：\n",obj)
    ```
    
    Out:
    
    修改前：
     0    4
    1    7
    2   -3
    3    2
    dtype: int64
    修改后：
     Bob      4
    Steve    7
    Jeff    -3
    Ryan     2
    dtype: int64

- **Series 常用属性**
  
  | 名称     | 属性                             |
  | ------ | ------------------------------ |
  | axes   | 以列表的形式返回所有行索引标签                |
  | dtype  | 返回对象的数据类型                      |
  | empty  | 返回一个空的 Series 对象               |
  | ndim   | 返回输入数据的维数                      |
  | size   | 返回输入数据的元素数量                    |
  | values | 以 ndarray 的形式返回 Series 对象      |
  | index  | 返回一个 RangeIndex 对象，用来描述索引的取值范围 |
  
  - axes
    
    ```python
    s = pd.Series(np.random.randn(5))
    print('The axes are:', s.axes)
    s
    ```
    
    Out:
    
    The axes are: [RangeIndex(start=0, stop=5, step=1)]
    
    0   -1.088609
    1   -0.034602
    2   -0.845855
    3    0.311825
    4   -1.006665
    dtype: float64
  
  - empty
    
    ```python
    s = pd.Series(np.random.randn(5))
    print('是否为空对象', s.empty)
    ```
    
    Out: 是否为空对象 False

## DataFrame

**DataFrame** 是一个表格型的数据结构，一个 DataFrame 表示一个表格，包含一个经过排序的列表集，它的每一列都可以有不同的类型值 (数字，字符串，布尔等等)；DataFrame 有行和列的索引；

- 每列可以是不同的值类型 (数值、字符串、布尔值等)

- 既有行索引 index，也有列索引 columns

- 可以被看做由 Series 组成的字典

DataFrame 的每一行数据都可以看成一个 Series 结构，只不过，DataFrame 为这些行中每个数据值增加了一个列表前；因此 DataFrame 其实是从 Series 的基础上演变而来；在数据分析任务中 DataFrame 的应用非常广泛，因为它描述数据更为清晰、值观

### 创建 DataFrame 对象

`pd.DataFrame(data, index, columns, dtype, copy)`

| 参数名称    | 说明                                                       |
| ------- | -------------------------------------------------------- |
| data    | 输入的数据，可以是 ndarray, series, list, dict, 标量以及一个 DataFrame  |
| index   | 行标签，如果没有传递 index 值，则默认行标签是 np.arange(n), n 代表 data 的元素个数 |
| columns | 列标签，如果没有传递值，则默认列表签是 np,arange(n)                         |
| dtype   | 表示每一列的数据类型                                               |
| copy    | 默认为 False，表示复制数据 data                                    |

- 通过列表创建 DataFrame
  
  ```python
  df1 = pd.DataFrame(np.random.randn(3, 3), index = list('abc'), columns = list('ABC))
  print(df1)
  ```
  
  Out:        A         B         C
  
  a  0.429699  0.107459 -0.525072
  b  0.694352 -0.538983 -1.332525
  c  0.122877 -1.450702  0.943294

- 嵌套列表创建
  
  ```python
  data = [['Alex', 10], ['Bob', 12], ['Clarke', 13]]
  df = pd.DataFrame(data, columns = ['Name', 'Age'])
  print(df)
  ```
  
  Out: 
  
   Name  Age
  
  0    Alex   10
  1     Bob   12
  2  Clarke   13

- 字典嵌套列表创建
  
  ```python
  data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'], 'Age':[28, 34, 29, 42]}
  df = pd.DataFrame(data)
  print(df)
  ```
  
  Out:
  
  Name  Age
  
  0    Tom   28
  1   Jack   34
  2  Steve   29
  3  Ricky   42

- Series 创建 DataFrame 对象
  
  ```python
  d = {'one':pd.Series([1, 2, 3], index = ['a', 'b', 'c']),
  'two':pd.Series([1, 2, 3, 4], index = ['a', 'b', 'c', 'd'])}
  df = pd.DataFrame(d)
  print(df)
  ```
  
  Out:
  
   one  two
  a  1.0    1
  b  2.0    2
  c  3.0    3
  d  NaN    4

### DataFrame 属性

- 查看 DataFrame 的头尾
  
  使用 **head** 可以查看前几行的数据，默认的是前5行，不过也可以自己设置；
  
  使用 **tail** 可以查看后几行的数据，默认也是5行，参数可以自己设置
  
  ```python
  df6 = pd.DataFrame(np.arange(36).reshape(6, 6), index = list('abcdeg'), columns = list('ABCDEF'))
  print(df6.head())
  print(df6.tail())
  ```
  
  Out:
  
     A   B   C   D   E   F
  a   0   1   2   3   4   5
  b   6   7   8   9  10  11
  c  12  13  14  15  16  17
  d  18  19  20  21  22  23
  e  24  25  26  27  28  29
  
     A   B   C   D   E   F
  
  b   6   7   8   9  10  11
  c  12  13  14  15  16  17
  d  18  19  20  21  22  23
  e  24  25  26  27  28  29
  f  30  31  32  33  34  35

- 查看数据值
  
  使用 **values** 可以查看 DataFrame 里的数据值，返回的是一个数组
  
  ```python
  # 接上一段代码
  print(df6.values)
  ```
  
  Out:
  
  [[ 0  1  2  3  4  5]
   [ 6  7  8  9 10 11]
   [12 13 14 15 16 17]
   [18 19 20 21 22 23]
   [24 25 26 27 28 29]
   [30 31 32 33 34 35]]
  
  ```python
  # 查看某一列所有的数据值
  print(df6['B'.values)
  ```
  
  Out: [ 1  7 13 19 25 31]
  
  ```python
  # 查看某一行所有的数据值
  # 使用 iloc 查看，iloc 是根据数字索引 (也就是行号)
  print(df6.iloc[0].values)
  ```
  
  Out: [0 1 2 3 4 5]

- 展示相关信息概览
  
  **df.info()**
  
  行数，列数，列索引，列非空值个数，列类型，行类型，内存占用等

## 输出设置

`pd.set_option('parameters', None)`

1. `display.max_columns`: int or None
   
   设置了一个 int 类型的数字阈值，展示的最大列数为设置的值，None 默认为最大值展示

2. `display.width`: int or None
   
   可设置一个阈值来使输出的数据不换行，当超过这个阈值才会换行

3. `display.max_rows`: int or None
   
   设置一个 int 的数字阈值，展示的最大行数为设置的值，None 最大值展示行数据

4. `max_colwidth`: int or None
   
   设置一个阈值，每列的展示数据量会被限制

5. `display.chop_threshold`: float or None
   
   设置了一个浮点型的数值阈值，DataFrame 中所有绝对值小于该阈值的数值，都会以0展示

6. `display.colheader_justify`: 'left' / 'right'
   
   控制调整列名，调整展示的列名靠左对齐或者右对齐

7. `display.date_dayfirst`: boolean
   
   当 value 设置为 True，print 时，会将日期放在最前面

8. `display.date_yearfirst`: boolean
   
   当 value 设置为 True，print 时，会将年份放在最前面

9. `display.multi_sparse`: boolean
   
   当 value 设置为 True 时将多索引省略展示，当设置为 False 时，多索引全部展示
   
   ![](D:\HANSHAN\Python\Picture\屏幕截图%202022-11-09%20134817.png)

## 索引以及列名重命名

- <mark>**导入数据时**</mark>
  
  设置 `names` 参数
  
  ```python
  col_names = ['City', 'Colors_Reported', 'Shape_Reported', 'State', 'Time']
  data = pd.read_csv(r'ufo.csv', header = 0, names = col_names)
  ```

- <mark>**导入数据后**</mark>
  
  ```python
  '''
  对列名直接赋值
  '''
  data = pd.read_csv(r'ufo.csv')
  col_names = ['City', 'Colors_Reported', 'Shape_Reported', 'State', 'Time']
  data.coulumns = col_names
  ```
  
  ```python
  '''
  使用 rename 的键值对单个列名进行修改
  '''
  data = pd.read_csv(r'ufo.csv')
  data.rename(columns = {"Colors Reported" : "Colors_Reported",
  "Shape Reported" : "Shape_Reported"}, inplace = True)
  ```
  
  ```python
  '''
  使用 string 方法
  '''
  data = pd.read_csv(r"ufo.csv")
  data.columns = data.columns.str.replace(" ","_")
  ```

## 重建索引

**reindex**

作用是对 Series 或 DataFrame 对象创建一个适应新索引的新对象

```python
obj = pd.Series([7.2, -4.3, 4.5, 3.6], index = ['b', 'a', 'd', 'c'])
print(obj)
obj.reindex(['a', 'b', 'c', 'd', 'e'])
```

Out:

b    7.2
a   -4.3
d    4.5
c    3.6
dtype: float64

Out[4]:

a   -4.3
b    7.2
c    3.6
d    4.5
e    NaN
dtype: float64

若索引在原数据中没有匹配，则以 NaN 填充

若不想以 NaN 填充，则可以用 **fill_value** 方法来设置

`obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value = 0)`

Out:

a   -4.3
b    7.2
c    3.6
d    4.5
e    0.0

- 缺失值的向前填充
  
  ```python
  import numpy as np
  obj1 = pd.Series(['blue', 'red', 'black'], index = [0, 2, 4])
  print(ibj1)
  obj1.reindex(np.arange(6), method = 'ffill')
  ```
  
  Out:
  
  0     blue
  2      red
  4    black
  dtype: object
  
  Out[7]:
  
  0     blue
  1     blue
  2      red
  3      red
  4    black
  5    black
  dtype: object

- 缺失值的向后填充
  
  ```python
  obj2 = pd.Series(['blue', 'red', 'black'], index = [0, 2, 4])
  obj2.reindex(np.arange(6), method = 'backfill')
  ```
  
  Out:
  
  0     blue
  1      red
  2      red
  3    black
  4    black
  5      NaN
  dtype: object

- 二维数据框 reindex 操作
  
  ```python
  df4 = pd.DataFrame(np.arange(9).reshape(3, 3),
  index = ['a', 'c', 'd'], columns = ['one', 'two', 'four'])
  print(df4)
  ```
  
  Out:
  
  one  two  four
  a    0    1     2
  c    3    4     5
  d    6    7     8
  
  ```python
  df4_new = df4.reindex(index = ['a', 'b', 'c', 'd'], columns = ['one', 'two', 'three', four'])
  print(df4_new)
  ```
  
  Out:
  
  one  two  three  four
  a  0.0  1.0    NaN   2.0
  b  NaN  NaN    NaN   NaN
  c  3.0  4.0    NaN   5.0
  d  6.0  7.0    NaN   8.0
  
  ```python
  # 对 DataFrame 使用 ffill 方法，书里的写法会报错，用如下写法是没问题的
  print(df4)
  print(df4.reindex(index = ['a', 'b', 'c', 'd', 'e', 'f']).ffill())
  ```
  
  Out:
  
  one  two  four
  a    0    1     2
  c    3    4     5
  d    6    7     8
     one  two  four
  a  0.0  1.0   2.0
  b  0.0  1.0   2.0
  c  3.0  4.0   5.0
  d  6.0  7.0   8.0
  e  6.0  7.0   8.0
  f  6.0  7.0   8.0

## 重置索引

数据清洗时，会将带空值的行删除，此时 DataFrame 或 Series 类型的数据不再是连续的索引，可以使用 **reset_index()** 重置索引

```python
df = pd.DataFrame(np.arange(20).reshape(5, 4), index = [1, 3, 4, 6, 8])
print(df)
print(df.reset_index())
```

Out:

  0   1   2   3
1   0   1   2   3
3   4   5   6   7
4   8   9  10  11
6  12  13  14  15
8  16  17  18  19

   index   0   1   2   3
0      1   0   1   2   3
1      3   4   5   6   7
2      4   8   9  10  11
3      6  12  13  14  15
4      8  16  17  18  19

> 原来的 index 变成数据列，保留下来
> 
> 不想保留原来的 index ，使用参数 drop = True, 默认 False

`print(df.reset_index(drop = True))`

Out:

  0   1   2   3
0   0   1   2   3
1   4   5   6   7
2   8   9  10  11
3  12  13  14  15
4  16  17  18  19

**set_index** 方法是专门用来将某一列设置为 index 的方法；它具有简单，方便，快捷的特点；

主要参数:

- keys: 需要设置为 index 的列名

- drop: True or False; 在将原来的列设置为 index，是否需要删除原来的列；默认为True

- append: True or False；新的 index 设置之后，是否要删除原来的 index；默认为 True

- inplace: True or False；是否要用新的 DataFrame 取代原来的 DataFrame；默认 False

```python
data = {
    'name':['张三', '李四', '王五', '小明'],
    'sex':['female', 'female', 'male', 'male'],
    'year':[2001, 2001, 2003, 2002],
    'city':['北京', '上海', '广州', '北京']
}
df = pd.DataFrame(data)
df5 = df.set_index('city')
print(df5)
```

Out: name     sex  year
city                   
北京     张三  female  2001
上海     李四  female  2001
广州     王五    male  2003
北京     小明    male  2002

```python
data = {
    'name':['张三', '李四', '王五', '小明'],
    'sex':['female', 'female', 'male', 'male'],
    'year':[2001, 2001, 2003, 2002],
    'city':['北京', '北京', '深圳', '深圳']
}
df = pd.DataFrame(data)
df2 = df.set_index(['city', 'name'])
df2
```

Out:

![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-31%20201627.png)

`df2.loc['北京'].loc['张三`]`

Out:

sex     female
year      2001
Name: 张三, dtype: object

`df2.loc['北京']`

Out:

![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-31%20201801.png)

```python
df3 = df.set_index(['name', 'city'])
df3
```

Out:

![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-31%20202105.png)

`df3.swaplevel()`

Out:

![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-31%20201627.png)

## 切片与索引

- 选取列数据
  
  ```python
  w1 = df['name']
  print('选取1列数据:\n', w1)
  w2 = df[['name', 'year']]
  print('选取2列数据:\n', w2)
  ```
  
  Out:
  
  选取1列数据：
   0    张三
  1    李四
  2    王五
  3    小明
  Name: name, dtype: object
  选取2列数据：
     name  year
  0   张三  2001
  1   李四  2001
  2   王五  2003
  3   小明  2002

- 选取行数据
  
  ```python
  print(df)
  print('显示前2行:\n', df[: 2])
  print('显示2-3行:\n', df[1: 3])
  ```
  
  Out: city name     sex  year
  0   北京   张三  female  2001
  1   上海   李四  female  2001
  2   广州   王五    male  2003
  3   北京   小明    male  2002
  显示前2行:
     city name     sex  year
  0   北京   张三  female  2001
  1   上海   李四  female  2001
  显示2-3两行:
     city name     sex  year
  1   上海   李四  female  2001
  2   广州   王五    male  2003

- loc 通过标签选取行和列
  
  ```python
  print(df.loc[:, ['name', 'year']])
  # 显示 name 和 year 两列
  print(df5.loc[['北京', '上海'], ['name', 'year']])
  # 显示北京和上海行中的 name 和 year 两列
  ```
  
  Out: name  year
  0   张三  2001
  1   李四  2001
  2   王五  2003
  3   小明  2002
  
  name  year
  
  city           
  北京     张三  2001
  北京     小明  2002
  上海     李四  2001

- iloc 通过位置选取行和列
  
  ```python
  a = df5.iloc[:, 2]
  print(a.dtype)
  print(df5.iloc[:, 2])
  # 显示前两列
  print(df5.iloc[[1, 3], :])
  # 显示第1和第3行
  print(df5.iloc[[1, 3], [1, 2]])
  ```
  
  Out:
  
  int64
  a    2001
  b    2001
  c    2003
  d    2002
  Name: year, dtype: int64
    name     sex  year city
  b   李四  female  2001   上海
  d   小明    male  2002   北京
  
  sex  year
  
  b  female  2001
  d    male  2002

- ix 选取行和列

- 布尔选择
  
  `df5[df5['year'] = 2001]`
  
  Out: name  year
  city sex              
  北京   female   张三  2001
  上海   female   李四  2001

## 查询数据

`df = pd.read_csv('beijing_tianqi_2018.csv')`

![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-08%20232242.png)

1. df.loc 方法；根据行、列的标签值查询
   
   > 1）使用单个 label 值查询数据
   > 
   > 2）使用值列表批量查询
   > 
   > 3）使用数值区间进行范围查询
   > 
   > 4）使用条件表达式查询
   > 
   > 5）调用函数查询
   > 
   > **.loc 既能查询，又能覆盖写入**
   
   - 使用单个 label 值查询数据
     
     ```python
     # 得到单个值
     df.loc['2018-01-03', 'bWendu']
     ```
     
     Out: 2
     
     ```python
     # 得到一个 Series
     df.loc['2018-01-03', ['bWendu', 'yWendu']]
     ```
     
     Out: 
     
     bWendu 2
     yWendu -5
     Name: 2018-01-03, dtype: objec
   
   - 使用值列表批量查询
     
     ```python
     # 得到 Series
     df.loc[['2018-01-03', '2018-01-04', '2018-01-05'], 'bWendu']
     ```
     
     Out: 
     
     ymd
     2018-01-03    2
     2018-01-04    0
     2018-01-05    3
     Name: bWendu, dtype: int32
     
     ```python
     # 得到 DataFrame
     df.loc[['2018-01-03', '2018-01-04', '2018-01-05'], ['bWendu', 'yWendu']]
     ```
     
     Out: ![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-08%20233119.png)
   
   - 使用数值区间进行范围查询
     
     > 区间既包含开始，也包含结束
     
     ```python
     # 行 index 按区间
     df.loc['2018-01-03':'2018-01-05', 'bWendu']
     ```
     
     Out:
     
     ymd
     2018-01-03    2
     2018-01-04    0
     2018-01-05    3
     Name: bWendu, dtype: int32
     
     ```python
     # 列 index 按区间
     df.loc['2018-01-03', 'bWendu':'fengxiang']
     ```
     
     Out:
     
     bWendu        2
     yWendu       -5
     tianqi       多云
     fengxiang    北风
     Name: 2018-01-03, dtype: object
     
     ```python
     # 行和列都按区间查询
     df.loc['2018-01-03':'2018-01-05', 'bWendu':'fengxiang']
     ```
     
     Out:
     
     ![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-08%20233605.png)
   
   - 使用条件表达式查询
     
     bool 列表的长度得等于行数或者列数
     
     *简单条件查询，最低温度低于 -10 度的列表*
     
     `df.loc[df['yWendu'] < -10, :]`
     
     Out:
     
     ![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-08%20233809.png)
     
     *复杂条件查询，查一下我心中的完美天气*
     
     > 注意；组合条件用 **&** 符号合并，每个条件判断都得带括号
     
     ```python
     # 查询最高温度小于30度，并且最低温度大于15度，并且是晴天，并且天气为优的数据
     df.loc[(df['bWendu'] <= 30) & (df['yWendu'] >= 15) & (df['tianqi'] == '晴') & (df['aqiLevel'] == 1), :]
     ```
     
     Out:
     
     ![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-09%20090736.png)
     
     ```python
     # 查看异常值情况
     Age_error = long_train.loc[(long_train['Age'] == '-1') | (long_train['Age'] == '0') | (long_train['Age'] == '-')]
     Age_error
     ```

2. df.iloc 方法；根据行、列的数字位置查询

3. df.where 方法

4. df.query 方法
   
   使用 df.query 可以简化查询
   
   `DataFrame.query(expr, inplace = False, **kwargs)`
   
   其中 expr 为要返回 boolean 结果的字符串表达式
   
   形如
   
   ```python
   df.query('a < 100')
   df.query('a < b & b < c') or df.query('(a < b) & (b < c)')
   ```
   
   df.query 中可以使用 @var 的方式传入外部变量
   
   ```python
   # 查询最低温度低于 -10 度的列表
   df.query('yWendu < 3').head(3)
   ```
   
   Out:
   
   ![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-09%20091749.png)
   
   ```python
   # 查询最高温度小于30度，并且最低温度大于15度，并且是晴天，并且天气为优的数据
   df.query("bWendu <= 30 & yWendu >= 15 &tianqi == '晴' & aqiLevel == 1")
   ```
   
   Out:
   
   ![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-09%20090736.png)
   
   ```python
   # 查询温差大于15度的日子
   df.query('bWendu-yWendu >= 15').head()
   ```
   
   Out:
   
   ![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-09%20092209.png)

5. 调用函数查询
   
   ```python
   # 直接写 lambda 表达式
   df.loc[lambda df : (df['bWendu'] <= 30) & (df['yWendu'] >= 15), :]
   ```
   
   Out: 
   
   ![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-09%20090736.png)
   
   64 rows × 8 columns
   
   ```python
   # 编写自己的函数，查询9月份空气质量好的数据
   def query_my_data(df):
       return df.index.str.startswith('2018-09') &(df['aqiLevel'] == 1)
   df.loc[query_my_data, :]
   ```
   
   Out:
   
   ![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-09%20092649.png)

## 排序的方法

**df.sort_values(self, by, axis = 0, ascending = True)**

- by: 按哪一个进行排序

- ascending: True 为升序排序

## 新增数据列

1. 直接赋值
   
   ```python
   # 清理温度列，变成数字类型
   df.loc[:, 'bWendu'] = df['bWendu'].str.replace('℃', '').astype('int32')
   df.loc[:, 'yWendu'] = df['yWendu'].str.replace('℃', '').astype('int32')
   df.head
   ```
   
   Out:
   
   ![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-09%20132554.png)
   
   ```python
   # 计算温差
   # 注：df['bWendu'] 其实是一个 Series，后面的减法返回的是 Series
   # df.loc[:, 'wencha'] = df['bWendu'] - df['yWendu']
   df['wencha1'] = df['bWendu'] - df['yWendu']
   df.head()
   ```
   
   Out:
   
   ![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-09%20132829.png)

2. df.apply 方法

3. df.assign 方法

4. 按条件选择分组分别赋值

5. 新增一行
   
   ```python
   datal = {'city':'兰州', 'name':'李红', 'year':'2005, 'sex':'female'}
   print(df5.append(datal, ignore_index = True))
   ```
   
   Out:
   
   name year city sex
   0 张三 2001 NaN NaN
   1 李四 2001 NaN NaN
   2 王五 2003 NaN NaN
   3 小明 2002 NaN NaN
   4 李红 2005 兰州 female
   
   ```python
   def df_creat(data):
       X = ['IndepenVar_' + '{}'.format(m) for m in range(len(data[0]))]
       df = pd.DataFrame(columns = X)
       for i in range(len(data)):
           row = data[i]
           df.loc[i] = row
       # print(df)
       return df
   ```

6. 增加一列并赋值
   
   ```python
   df5['score'] = [85, 78, 96, 80]
   print(df5)
   ```

7. insert【原地添加】
   
   `Dataframe.insert(loc, column, value, allow_duplicates=False)`
   
   在Dataframe的指定列中插入数据
   
   - loc:  int型，表示第几列；若在第一列插入数据，则 loc=0
   
   - column: 给插入的列取名，如 column='新的一列'
   
   - value：数字，array，series等都可（可自己尝试） 
   
   - allow_duplicates: 是否允许列名重复，选择Ture表示允许新的列名与已存在的列名重复。

8. `append`
   
   ```python
   df = np.array([])
   df = np.append(df, np.array([temp_1]))
   ```

## 删除数据列

删除行数据

`.drop()`

删除列数据

`.drop('', axis = 1, inplace = True)`

```python
# 删除异常值所在行
for i in Age_error.index:
    long_train.drop(labels = i, inplace = True)
```

## 数据读写

1. <mark>**<font color = red>读取数据</font>**</mark>
   
   ```python
   pd.read_excel(io, sheetname = 0, header = 0, skiprows = None, index_col = None,
                 names = None, arse_cols = None, date_parser = None, na_values = None,
                 thousands = None, convert_float = True, has_index_names = None,
                 converters = None, dtype = None, true_values = None,
                 false_values = None, engine = None, squeeze = False, **kwds)
   ```
   
   - `io`: excel 路径
   
   - `sheet_name`: None, string, int, 默认为0
     
     string 用于工作表名称
     
     int 用于索引工作表位置
     
     string, int 列表用于请求多个工作表
     
     ```python
     #参数为None时，返回全部的表格，是一个表格的字典；
     sheet = pd.read_excel('example.xls',sheet_name= None)
     #当参数为list = [0，1，2，3]此类时，返回的多表格同样是字典
     sheet = pd.read_excel('example.xls',sheet_name= 0)
     sheet = pd.read_excel('example.xls',sheet_name= [0,1])
     #同样可以根据表头名称或者表的位置读取该表的数据
     sheet = pd.read_excel('example.xls',sheet_name= 'Sheet0')
     sheet = pd.read_excel('example.xls',sheet_name= ['Sheet0','Sheet1'])
     sheet = pd.read_excel('example.xls',sheet_name=[0,1,'Sheet3'])
     ```
   
   - `header`: 指定作为列名的行
     
     默认0，数据为列名行以下的数据；若数据不含列名，则设定 `header = None`
     
     ```python
     #数据不含作为列名的行
     sheet = pd.read_excel('example.xls',sheetname= 1,header = None)
     #默认第一行数据作为列名
     sheet = pd.read_excel('example.xls',sheetname= 1,header =0)
     ```
   
   - `index_col`: 指定列为索引列
     
     可以是工作表列名称，e.g. `index_col = '排名'`
     
     可以是整型或整型列表，`index_col = 0 or index_col = [0, 1]`
     
     ```python
     sheet = pd.read_excel('example.xls', index_col=1)
     sheet = pd.read_excel('example.xls', index_col='排名')
     ```
   
   - `usecols`: int or list，默认为 None
   
   - `dtype`: 列的数据类型，默认为 None
     
     字典类型 `{'列名1': 数据类型}`
     
     pandas 默认将文本类的数据读取为整型
     
     ```python
     sheet = pd.read_excel('example.xls', dtype={'month': str, 'cd': int})
     sheet = pd.read_excel('example.xls', dtype={'a'：np.float64，'b'：np.int32})
     ```
   
   - `converters`: 在某些列中转换值的函数命令
     
     键可以是 int or list；值是接受一个输入参数的函数
     
     ```python
     sheet = pd.read_excel('example.xls', converters={'num': int, 'age': str})
     ```
   
   - `skiprows`: 跳过特定行
     
     `skiprows = n`，挑过前 n 行
     
     `skiprows = [a, b, c]`，跳过第 a 行，第 b 行，第 c 行【索引从0开始】
   
   - `nrows`: 需要读取的行数
   
   - `names`: 自定义最终的列名

2. <mark>**<font color = red>写入数据</font>**</mark>
   
   批量写入多个 sheet
   
   ```python
   # 重点1：writer不能在下面的for循环中
   writer = pd.ExcelWriter('filename.xlsx')
   counter=0
   for sy in range(1,10):
     counter=counter+1
     data = pd.DataFrame(c) 
     data.to_excel(writer, 'sheet'+str(counter))
   
   writer.save()        # 重点2：save不能在for循环里
   writer.close()
   ```

## 统计方法

1. 汇总类统计
   
   ```python
   # 提取所有数字列统计结果
   df.describe()
   ```
   
   Out:
   
   ![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-10%20134054.png)
   
   ```python
   # 查看单个 Series 的数据
   df['bWendu'].mean()
   ```
   
   Out: 18.665753424657535
   
   `df['bWendu'].max()`
   
   Out: 38

2. 唯一去重和按值计数
   
   - 唯一性去重
     
     `df['fengxiang'].unique()`
     
     Out: array(['东北风', '北风', '西北风', '西南风', '南风', '东南风', '东风', '西风'], dtype=object)
     
     `df['fengli'].unique()`
     
     Out: array(['1-2级', '4-5级', '3-4级', '2级', '1级', '3级'], dtype=object)
   
   - 按值计数
     
     `df['fengxiang'].value_counts()`
     
     Out:
     
     南风     92
     西南风    64
     北风     54
     西北风    51
     东南风    46
     东北风    38
     东风     14
     西风      6
     Name: fengxiang, dtype: int64
     
     `df['fengli'].value_counts()`
     
     Out:
     
     1-2级    236
     3-4级     68
     1级       21
     4-5级     20
     2级       13
     3级        7
     Name: fengli, dtype: int64

3. 相关系数和协方差
   
   - 协方差
     
     **衡量同向反向程度**，如果协方差为正，说明 X, Y 同向变化，协方差越大说明同向程度越高；如果协方差为负，说明 X, Y 反向运动，协方差越小说明反向程度越高
   
   - 相关系数
     
     **衡量相似度程度**，当他们的相关系数为 1 时，说明两个变量变化时的正向相似度最大，当相关系数为 -1 时，说明两个变量变化的反向相似度最大

| function | explanation | function   | explanation |
| -------- | ----------- | ---------- | ----------- |
| max      | 最大值         | mad        | 平均绝对误差      |
| min      | 最小值         | kurt       | 峰度          |
| mean     | 平均值         | skew       | 偏度          |
| median   | 中位数         | shift      | 移动          |
| quantile | 分位值         | diff       | 插值          |
| mode     | 众数          | pct_change | 差值百分比       |
| var      | 方差          | corr       | 相关系数        |
| std      | 标准差         | cov        | 协方差         |
| sem      | 平均值标准误差     | rank       | 排名          |

> 此函数不一定是 pandas 里的

## 缺失值的处理

- isnull、notnull: 检测是否是空值，可用于 DataFrame 和 Series

- dropna: 删除缺失值
  
  - axis: 0 or 'index', 1 or 'columns'; default = 0
  
  - how: 如果等于 any 则任何值为空都删除；如果等于 all 则所有值都为空才删除
  
  - inplace: 如果为 True 则修改当前  DataFrame, 否则返回新的 DataFrame
  
  ```python
  # short_data 缺失值
  short_isnull = short_data.isnull().values
  # list(short_null).count(True)
  list(short_isnull)
  # 删除缺失值所在行
  short_data.dropna(how = 'any', inplace = True)
  ```

- fillna: 填充空值
  
  - value: 用于填充的值，可以是单个值，或者字典 (key 是列名，value 是值)
  
  - method: 等于 ffill 使用前一个不为空的值填充 forword fill; 等于 bfill 使用后一个不为空的值填充 backword fill
  
  - axis: 0 or 'index'; 1 or 'columns'
  
  - inplace: 如果为 True 则修改当前 DataFrame, 否则返回新的 DataFrame

- 替换法【使用 `fillna`】
  
  - 用一个特定的值替换缺失值
  
  - 特征可分为**数值型**和**类别型**，两者出现缺失值时的处理方法也是不同的
    
    - <u>数值型</u>
      
      通常利用其均值，中位数和众数等描述其集中趋势的统计量来代替
    
    - <u>类别型</u>
      
      则选择众数来替换

- 插值法
  
  常用的插值法有线性插值，多项式插值和样条插值等
  
  - **线性插值**是一种较为简单的插值方法，针对<u>**已知**</u>的值求出线性方程，通过求解线性方差得到缺失值
  
  - **多项式插值**是利用已知的值拟合一个多项式，使得现有的数据满足这个多项式，再利用这个多项式求解缺失值，常见的多项式插值法有<u>拉格朗日插值，牛顿插值</u>
  
  - **样条插值**是以可变样条来作出一条经过一系列点的光滑曲线的插值方法，插值样条由一些多项式组成，每一个多项式都是由相邻两个数据点决定，*这样可以保证两个相邻多项式及其导数在连接处连续*
  
  ```python
  from scipy.interpolate import interpld, lagrange, spline
  x = np.array([1, 2, 3, 6, 9, 23])
  y = pn.array([3, 6, 4, 1, 7, 33])
  # 线性插值
  model = interpld(x, y, kind = 'linear')
  model([3, 5])
  # 拉格朗日
  # 样条插值
  spline(x, y, xnew = [4, 5])
  ```

## 重复值处理

数据重复会导致方差变小，数据分布会发生较大变化；

`df.duplicated()`  检测重复值

`df.drop_duplicates(inplace = True)`  删除重复值

```python
# 检测重复值
duplicate = short_data['user_id'].duplicated()
print('user_id 中重复值有: ', list(duplicate).count(True))
print('=' * 50)
# 删除重复值所在行
print('重复值所在行:\n', duplicate.loc[duplicate == True])
for i in duplicate.loc[duplicate == True].index:
    short_data.drop(labels = i, inplace = True)

print('=' * 50)
print('删除后 user_id 中重复值有: ', list(short_data['user_id'].duplicated()).count(True))
```

## 异常值处理

1. **<font color = cornflowerblue>数据中个别值数值明显偏离其余的数值，有时也称为离群点</font>**
   
   检测异常值就是检验数据中是否有录入错误以及含有不合法的数据

2. **<font color = cornflowerblue>不良影响</font>**
   
   异常值的存在对数据分析十分危险，如果计算分析过程中的数据有异常，那么会对结果产生不良影响
   
   从而导致分析结果产生偏差乃至错误 

3. **<font color = cornflowerblue>检测方法</font>**
   
   - <u>3 $\sigma$ 原则</u>
     
     - 又称为拉依达法则；
       
       该法则就是先假设一组检测数据只含有随机误差，对原始数据进行计算处理得到**标准差**，然后按**一定的概率**确定一个区间，任务误差超过这个区间就属于异常值
     
     - 这种判别处理方式**仅适用于**对正态或近似正态分布的样本数据进行处理；
       
       ![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-12-03%20040704.png)
       
       表中 $\sigma$ 表示标准差，$\mu$ 表示均值，$x = \mu$ 为图形对称轴
     
     - 数据的数值分布几乎全部集中在区间三内，超出这个范围的数据仅占不到 0.3% ，故根据小概率原理，可以认为超出 3$\sigma$ 的部分数据为异常数据
   
   - <u>箱线图</u>

## 数据转换

1. **<mark><font color = red>哑变量处理</font></mark>**
   
   - 数据分析模型中有相当一部分的算法模型都要求输入的特征为数值型，但实际数据中特征的类型不一定只有数值型，还会存在相当一部分的类别型
     
     这部分的特征需要经过哑变量处理才可以放入模型之中
   
   - <u>pandas 库中的 `get_dummies` 函数进行处理</u>
     
     ```python
     pd.get_dummies(data, prefix = None, prefix_sep = '_', dummy_na = False,
                    columns = None, Sparse = False, drop_first = False)
     ```
     
     - data: 接收 array, DataFrame, Series 数据
     
     - prefix: 接收 string, string 的列表或者 string 的 dict; 表示哑变量的化后的列名前缀
     
     - prefix_sep: 接收 string, 表示前缀的连接符; 默认为 '_'
     
     - dummy_na: 接收 boolean；表示是否为 Nan 值添加一列
     
     - columns: 接收类似 list 的数据；表示 DataFrame 中需要编码的列名
     
     - sparse: 接收 boolean; 表示虚拟列是否是稀疏的
     
     - drop_first: 接收 boolean; 表示是否通过从 k 个分类级别中删除第一级来获得 k-1 个分类级别；

2. **<mark><font color = red>离散化连续性数据</font></mark>**
   
   - <u>**等宽法**</u>
     
     `pd.cut()`
   
   - <u>**等频法**</u>
     
     - `cut` 函数虽然不能够直接实现等频离散化，但是可以通过定义将相同数量的记录放进每个区间
     
     - 等频率法离散化的方法相比较于等宽法离散化而言，避免了类分布不均匀的问题，但同时却**也有可能将数值非常接近的两个值**分到不同的区间以满足每个区间中固定的数据个数
     
     ```python
     def samfreq(data, k);
         w = data.quatile(np.arange(0, 1 + 1 / k, 1 / k))
         return pd.cut(data, w)
     
     samefreq(data['amounts'], k = 5).value_counts()
     ```
   
   - <u>**基于聚类分析方法**</u>
     
     - 一维聚类的方法包含两个步骤
       
       1. 将连续型数据用聚类算法 如 K-Means 算法等 进行聚类
       
       2. 处理聚类得到的簇，将合并到一个簇的连续性数据做同一标记
     
     - 聚类分析的离散化方法需要指定簇的个数，用来决定产生的区间数
     
     - K-Means 聚类分析的离散化方法可以很好地根据现有特征的数据分布状况进行聚类；
       
       但由于 K-Means 算法本身的缺陷，用该方法进行离散化时依旧需要指定离散化后类别的数目；
       
       此时需要配合聚类算法评价方法，找出最优的聚类簇数目

## 转换字典

`to_dict(orient = None)`

- `orient = 'dict'`: 函数默认，形式 `{column: {index: value}}`

- `orient = 'list'`: `{column: {[values]}}`

- `orient = 'series'`: `{column: Series(value)}`

- `orient = 'split'`: `{'index': [index], 'columns': [columns], 'data': [values]}`

- `orient = 'records'`: `[{column: value}...{column: value}]`

- `orient = 'index'`:  `{index: {column: value}}`

## 字符串函数

1. 字符拼接函数
   
   **df['a'].str.cat()**
   
   2个任意类型的变量，通过拼接，新生成3个字符型变量
   
   - df['a'] 表示一列，相当于 Excel 中的一列
   
   - .str, 调取 str 属性，将前面的内容转化为字符型
   
   - .cat, 来自 'concat', 合并字符串
     
     ```python
     dict = {'first name' : ['Steph', 'LeBron'],
             'last name' : ['Curry', 'James']}
     df = pd.DataFrame(dict) # 通过字典数据，构建 DataFrame
     df['full name'] = df['first name'].str.cat(df['last name'], sep = ' ')
     df
     ```
     
     Out:
     
     ![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-27%20170638.png)

2. 字符查找函数
   
   **df['a'].str.contains('abc')**
   
   ```python
   df['full name'] = df['first name'].str.cat(df['last name'], sep = ' ')
   print(df)
   df['full name'].str.contains('Le')
   ```
   
   Out: first name last name full name
   0 Steph Curry Steph Curry
   1 LeBron James LeBron James
   
   Out[5]:
   
   0 False
   1 True
   
   Name: full name, dtype: bool

3. 字符串查找函数
   
   **df['a'].str.startswith('L')/endswith()**
   
   ```python
   print(df['full name'].str.startswith('L'))
   print(df['full name'].str.startswith('l'))
   # 字母不区分大小写；endswith 原理类似
   df
   ```
   
   Out:
   
   ![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-27%20203128.png)

4. 计算字变量取值的字符串长度**df['a'].str.len()**
   
   ```python
   print(df['full name'])
   df['full name'].str.len()
   # 长度包括空格
   ```
   
   Out: 0 Steph Curry
   1 LeBron James
   Name: full name, dtype: object
   
   Out[7]:
   
   0 11
   1 12
   Name: full name, dtype: int64

5. 计算某个变量取值某个字符出现的个数**df['a'].str.count('s')**
   
   `df['full name'].str.count('s')`
   
   Out: 0 0
   1 1
   Name: full name, dtype: int64

6. 转换大小写函数**df['a'].str.upper()/lower()**
   
   ```python
   print(df['full name'].str.uper())
   print(df['full name'].str.lower())
   ```
   
   Out:
   
   0 STEPHCURRY
   1 LEBRONJAMES
   Name: full name, dtype: object
   0 stephcurry
   1 lebronjames
   Name: full name, dtype: object

7. 截取字符串函数
   
   **df['a'].str.get('0')**
   
   ```python
   # 倒数第一个字符
   print(df['full name'].str.get(-1))
   # 第一个字符
   print(df['full name'].str.get(0))
   ```
   
   Out:
   
   0 y
   1 s
   Name: full name, dtype: object
   0 S
   1 L
   Name: full name, dtype: object

8. 字符串分割函数
   
   **df['a'].str.split(sep = ' ')**
   
   ```python
   # 以空格为分隔符
   print(df['full name'].str.split(' '))
   # 先把空格替换为下划线
   print(df['full name'].str.replace(' ', '_'))
   # 表示用空格来对特殊字符进行分隔
   print(df['full name'].str.split(':'))
   ```
   
   ```python
   # +expand 选项，自动分割，生成新的变量
   df['full name'].str.replace(' ', '_')
   df[['first name', 'last name']] = df['full name'].str.split(expand = True)
   df[['first name', 'last name]]
   # 这里将 'full name' 变量分割之后，新生成了两个变量：'first name', 'last name'
   ```

9. 在字符串左边/右边添加特定字符**df['a'].str.pad('20, fillchar = 'A')**
   
   ```python
   # 总长度为20
   # 注意，不能添加字符串string
   print(df['full name'].str.pad(20, fillchar = 'A))
   print(df['full name'].str.pad(20, side = 'right', fillchar = 'B'))
   ```
   
   Out:
   
   0 Steph Curry
   1 LeBron James
   Name: full name, dtype: object
   0 AAAAAAAAASteph Curry
   1 AAAAAAAALeBron James
   Name: full name, dtype: object
   
   0 Steph CurryBBBBBBBBB
   1 LeBron JamesBBBBBBBB
   Name: full name, dtype: object

10. 字符串替换函数
    
    **df['a'].str.replace(' ', '_')**

11. 清除两边的特殊字符**df['a'].str.strip()/lstrip()/rstrip()**
    
    ```python
    # strip 函数区分大小写；也可以清楚字符串两边的 '\n' 和 '\0' (空格) 等
    # 清除任意的 'S', 'L', 'r' 字符
    df['full name'].str.strip('SLr')
    ```
    
    Out:
    
    0 teph Curry
    1 eBron James
    Name: full_name, dtype: object

12. 基于正则表达式，查找特定字符**df['a'].str.findall('[A-Z]')**
    
    ```python
    # 查找所有的大写字母
    df['full name'].str.findall('[A-Z]')
    ```
    
    Out:
    
    0 [S, C]
    1 [L, B, J]
    Name: full name, dtype: object

13. 抽取特定字符**df['a'].str.extract(r'(["LS"]')**
    
    `df['full name'].str.extract('([a-zA-Z]+)', expand = False)`
    
    Out:
    
    0 Steph
    1 LeBron
    Name: full name, dtype: object

## 分组聚合

`DataFrame.groupby(by = None, axis = 0, level = None, as_index = True, sort = True, group_keys = True, squeeze = NoDefault.no_default, observed = False, dropna = True)`

- by: mapping, function, label, or list of labels

- axis: '0' or 'index', '1' or 'columns'; default '0'

- level: int, level name, or sequence of such, default  None

- as_index: bool, default True

- sort: bool, default True

- group_keys: bool, default True

- squeeze: bool, default False; reduce the dimensionality of the return type if possible, otherwise return a consistent type. (Deprecated since version 1.1.0.)

- observed: bool, default False; This only applies if any of the groupers are Categoricals.  If True: only show observed values for categorical groupers. If False: show all values for categorical groupers.

- dropna: bool, default True; If True, and if group keys cantain NA values, NA values together with row/column will be dropped. If False, Na values will also be treated as the key in groups. (New in version 1.1.0.)

- Returns: DataFraneGroupBy

```python
fpath = 'beijing_tianqi_2018.csv'
df = pd.read_csv(fpath)
df.loc[:, 'bWendu'] = df['bWendu'].str.replace('C', '').astype('int32')
df.loc[:, 'yWendu'] = df['yWendu'].str.replace('C', '').astype('int32')
# 新增一列
df['month'] = df['ymd'].str[:7]
df.head()
```

Out:

![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-30%20091433.png)

```python
# 查看每个月的最高温度
max_tem = df.groupby('month')['bWendu'].max()
data
```

Out:

month
2018-01     7
2018-02    12
2018-03    27
2018-04    30
2018-05    35
2018-06    38
2018-07    37
2018-08    36
2018-09    31
2018-10    25
2018-11    18
2018-12    10
Name: bWendu, dtype: int32

```python
# 查看每个月的最高温度、最低温度、平均空气质量指数
group_data = df.groupby('month').agg({'bWendu':np.max, 'yWendu':np.min, 'aqi':np.mean})
group_data
```

Out:

![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-30%20092157.png)

**两列数据以上分组聚合**

```python
df = df.groupby([df.index, '日期']).sum()
```

```python
# 对地区索引进行分组聚合操作
df= df['销售额'].groupby([df['国家'], df['日期']]).sum()
```

1. **<font color = cornflowerblue>常规分组</font>**
   
   ```python
   df = pd.DataFrame({'key1':['a','a','b','b','a'],'key2':['one','two','one','two','one'],'data1':np.random.randint(1,10,
        ...: 5),'data2':np.random.randint(1,10,5)})
   
   In [443]: df
   Out[443]:
     key1 key2  data1  data2
   0    a  one      4      6
   1    a  two      7      2
   2    b  one      1      3
   3    b  two      4      1
   4    a  one      6      6
   
   In [444]: df['data1'].groupby(df['key1']).mean()
   Out[444]:
   key1
   a    5.666667
   b    2.500000
   Name: data1, dtype: float64
   
   In [449]: df.groupby('key1').sum()
   Out[449]:
         data1  data2
   key1
   a        17     14
   b         5      4
   # 把不是数字的‘key2’组直接丢掉
   
   In [450]: df.groupby('key1').sum()['data1']
   Out[450]:
   key1
   a    17
   b     5
   Name: data1, dtype: int32
   
   In [453]: mean = df.groupby(['key1','key2']).sum()['data1']
   
   In [454]: mean
   Out[454]:
   key1  key2
   a     one     10
         two      7
   b     one      1
         two      4
   Name: data1, dtype: int32
   
   # 通过unstack转化成DafaFrame
   In [455]: mean.unstack()
   Out[455]:
   key2  one  two
   key1
   a      10    7
   b       1    4
   ```

2. **<font color = cornflowerblue>自定义分组键</font>**
   
   ```python
   In [445]: key = [1,2,1,1,2]
   
   In [446]: df['data1'].groupby(key).mean()
   Out[446]:
   1    3.0
   2    6.5
   Name: data1, dtype: float64
   ```

3. **<font color = cornflowerblue>多层索引分组</font>**
   
   ```python
   In [447]: df['data1'].groupby([df['key1'],df['key2']]).sum()
   Out[447]:
   key1  key2
   a     one     10
         two      7
   b     one      1
         two      4
   Name: data1, dtype: int32
   
   In [448]: df['data1'].groupby([df['key1'],df['key2']]).size()
   Out[448]:
   key1  key2
   a     one     2
         two     1
   b     one     1
         two     1
   Name: data1, dtype: int64
   ```

4. **<font color = cornflowerblue>使用 groupby 的迭代器协议</font>**
   
   ```python
   In [457]: for name, group in df.groupby('key1'):
        ...:     print (name)
        ...:     print (group)
        ...:
   a
     key1 key2  data1  data2
   0    a  one      4      6
   1    a  two      7      2
   4    a  one      6      6
   b
     key1 key2  data1  data2
   2    b  one      1      3
   3    b  two      4      1
   
   # 转换成字典
   In [458]: dict(list(df.groupby('key1')))['a']
   Out[458]:
     key1 key2  data1  data2
   0    a  one      4      6
   1    a  two      7      2
   4    a  one      6      6
   ```
   
   ```python
   # for 遍历将各个地区分开成多个数据表
   for group_name, group_data in data_region_volumeM:
       if group_name == 'Eastern':
           df_EasternVolume = group_data
       elif group_name == 'Middle':
           df_MiddleVolume = group_data
       elif group_name == 'Northern':
           df_NorthernVolume = group_data
       elif group_name == 'Southern':
           df_SouthernVolume = group_data
       else:
           df_WesternVolume = group_data
   ```
   
   ```python
   data_country_volumeY_group = data_country_volumeY.groupby('国家')
   for group_name, group_data in data_country_volumeY_group:
   
       group_data = group_data.set_index('日期')
       temp_1 = (group_data.loc[['2018'], ['销售额']].values - group_data.loc[['2017'], ['销售额']].values) / (group_data.loc[['2017'], ['销售额']].values * 0.1)
       values_17_18 = np.append(values_17_18, np.array([temp_1]))
   
       temp_2 = (group_data.loc[['2019'], ['销售额']].values - group_data.loc[['2018'], ['销售额']].values) / (group_data.loc[['2018'], ['销售额']].values * 0.1)
       values_18_19 = np.append(values_18_19, np.array([temp_2]))
   
       temp_3 = (group_data.loc[['2020'], ['销售额']].values - group_data.loc[['2019'], ['销售额']].values) / (group_data.loc[['2019'], ['销售额']].values * 0.1)
       values_19_20 = np.append(values_19_20, np.array([temp_3]))
   ```

## 离散间隔

`pd.cut(x, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False)`

- bins：int, sequence of scalars, 或 IntervalIndex分级依据。

- int：定义equal-width个bin的数量，范围为x。范围x每侧扩展.1％以包括最小和最大值x。

- 标量序列：定义面元边以允许宽度不均匀。没有扩大范围x已经完成了。

- IntervalIndex：定义要使用的确切bin。请注意，IntervalIndex用于bins必须不重叠。

- right：bool, 默认为 True指示是否bins是否包含最右边。如果right == True(默认)，然后bins [1, 2, 3, 4]表示(1,2]，(2,3]，(3,4]。当bins是一个IntervalIndex。

- labels：array 或 False, 默认为 None指定返回的垃圾箱的标签。必须与生成的垃圾箱长度相同。如果为False，则仅返回垃圾箱的整数指示符。这会影响输出容器的类型(请参见下文)。在以下情况下将忽略此参数bins是一个IntervalIndex。如果为True，则会引发错误。

- retbins：bool, 默认为 False是否归还垃圾箱。当将垃圾箱作为标量提供时很有用。

- precision：int, 默认为 3存储和显示垃圾箱标签的精度。

- include_lowest：bool, 默认为 False第一个间隔是否应为left-inclusive。

- duplicates：{default ‘raise’, ‘drop’}, 可选参数如果bin边不是唯一的，则引发ValueError或丢弃非唯一的。

```python
import random
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
#用随机数产生一个二维数组。分别是年龄的性别。
df=pd.DataFrame({'Age':np.random.randint(0,70,100),
                'Sex':np.random.choice(['M','F'],100),
                })
#用cut函数对于年龄进行分段分组，用bins来对年龄进行分段，左开右闭
age_groups=pd.cut(df['Age'],bins=[0,18,35,55,70,100])
# print(age_groups)
print(df.groupby(age_groups).count())
```

![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-11-14%20223951.png)

## 堆叠索引拆分

```python
df = df.unstack(0)
df
```

![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-11-11%20013707.png)

行索引有两级，里层的索引和外层的索引

`stack` 和 `unstack` 默认操作最里层的数据，当我们想操作外层的数据时，传入一个层级序号或名称来拆分一个不同的层级

分不清层级序号可以输入层级名称

```python
result.unstack(0) = result.unstack('日期')
result.unstack(1) = result.unstack('国家')
```

## 合并数据

### join

默认情况下他是把**行**索引相同的数据合并到 一起

```python
df1 = pd.DataFrame(np.ones((2, 4)), index = ['A', 'B'], columns = list('abcd'))
df1
```

Out:

![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-30%20144641.png)

```python
df2 = pd.DataFrame(np.zeros((3, 3)), index = ['A', 'B', 'C'], columns = list('xyz'))
df2
```

Out:

![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-30%20144724.png)

`df1.join(df2)`

Out:

![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-30%20144829.png)

`df2.join(df1)`

Out:

![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-30%20144910.png)

### merge

按照列索引进行合并

`DataFrame.merge(right, how = 'inner', on = None, left_on = None, right_on = None, left_index = False, right_index = False, sort = False, suffixes = ('_x', '_y'), copy = True, indicator = False, validate = None)`

![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-05-30%20150358.png)

## 时间序列

- Pandas 功能1
  
  解析时间格式字符串、np.datetime64、datetime.datetime 等多种时间序列数据
  
  ```python
  dti = pd.to_datetime(['1/1/2018', np.datetime64('2018-01-01'),
                          datetime.datetime(2018, 1, 1)])
  dti
  ```
  
  Out:
  
  DatetimeIndex(['2018-01-01', '2018-01-01', '2018-01-01'], dtype='datetime64[ns]', freq=None)

- Pandas 功能2
  
  生成 DatetimeIndex, TimedeltaIndex, PeriodIndex 等定频日期与时间段序列
  
  ```python
  dti = pd.date_range('2018-01-01', period = 7, freq = 'B')
  dti
  ```
  
  Out:
  
  DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04','2018-01-05', '2018-01-08', '2018-01-09'],dtype='datetime64[ns]', freq='B')

- Pandas 功能3
  
  按指定频率重采样，并转换为时间序列
  
  ```python
  idx = pd.date_range('2018-01-01', periods = 5, freq = 'H')
  ts = pd.Series(range(len(idx)), index = idx)
  print(ts)
  ts.resample('0.5H').backfill()
  ```
  
  Out:
  
  2018-01-01 00:00:00    0
  2018-01-01 01:00:00    1
  2018-01-01 02:00:00    2
  2018-01-01 03:00:00    3
  2018-01-01 04:00:00    4
  Freq: H, dtype: int32
  
  Out[9]:
  
  2018-01-01 00:00:00    0
  2018-01-01 00:30:00    1
  2018-01-01 01:00:00    1
  2018-01-01 01:30:00    2
  2018-01-01 02:00:00    2
  2018-01-01 02:30:00    3
  2018-01-01 03:00:00    3
  2018-01-01 03:30:00    4
  2018-01-01 04:00:00    4
  Freq: 30T, dtype: int32

- Pandas 功能4
  
  用绝对或相对时间差计算日期与时间
  
  ```python
  t = pd.Timestamp('2018-01-05')
  t.day_name()
  ```
  
  Out: 'Friday'
  
  ```python
  ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
  ts
  ```
  
  Out: Timestamp('2020-03-14 15:32:52.192548651')
  
  ```python
  s = ts + pd.Timedelta('1 day')
  print(s)
  m = ts + pd.offsets.BDay()
  print(m)
  ```
  
  Out:
  
  2020-03-15 15:32:52.192548651
  2020-03-16 15:32:52.192548651

Pandas 支持4种常见时间概念

1. 日期时间 (Datetime)
   
   带时区的日期时间，类似于标准库的 datetime.datetime

2. 时间差 (Timedelta)
   
   绝对时间周期，类似于标准库的 datetime.timedelta

3. 时间段 (Timespan)
   
   在某一时点以指定频率定义的时间跨度

4. 日期偏移 (Dateoffset)
   
   与日历运算对应的时间段

> 一般情况下，时间序列主要是 Series 或 DataFrame 的时间型索引，可以用时间元素进行操控
> 
> `pd.Series(range(3), index = pd.date_range('2000', freq = 'D', periods = 3))`
> 
> Out:
> 
> 2000-01-01    0
> 2000-01-02    1
> 2000-01-03    2
> Freq: D, dtype: int32

### 时间戳

时间戳是最基本的时间序列数据，用于把数值与时点关联在一起

`pd.Timestamp(datetime.datetime(2012, 5, 1))`

`pd.Timestamp('2012-05-01')`

`pd.Timestamp(2012, 5, 1)`

Out: Timestamp('2012-05-01 00:00:00')

> 以上三种方式输出结果相同，是等价的

**freq** 缩写

| 缩写      | 偏移量类型              | 说明            |
| ------- | ------------------ | ------------- |
| D       | Day                | 每日历日          |
| B       | BusinessDay        | 没工作日          |
| H       | Hour               | 每小时           |
| T 或 min | Minute             | 没分            |
| S       | Second             | 每秒            |
| L 或 ms  | Milli              | 每毫秒（即每千分之一秒）  |
| U       | Micro              | 每微秒（即每百万分之一秒） |
| M       | MonthEnd           | 每月最后一个日历日     |
| BM      | BusinessMonthEnd   | 每月最后一个工作日     |
| MS      | MonthBegin         | 每月第一个日历日      |
| BMS     | BusinessMonthBegin | 每月第一个工作日      |

#### 转换时间戳

**to_datetime** 函数用于转换字符串、纪元式及混合的日期 Series 或日期列表；转换的是 Series 时，返回的是具有相同的索引的 Series，日期时间列表则会被转换为 DatetimeIndex

`pd.to_datetime(pd.Series(['Jul 31, 2009', '2010-01-10', None]))`

Out:

0   2009-07-31
1   2010-01-10
2          NaT
dtype: datetime64[ns]

`pd.to_datetime(['2005/11/23', '2010.12.31'])`

Out: DatetimeIndex(['2005-11-23', '2010-12-31'],dtype='datetime64[ns]', freq=None)

解析欧式日期（日-月-年），要用 **dayfirst** 关键字参数

```python
print(pd.to_datetime(['04-01-2012 10:00'], dayfirst = True)
print(pd.to_datetime(['14-01-2012', '01-14-2012'], dayfirst = True)
```

Out:

DatetimeIndex(['2012-01-04 10:00:00'], dtype='datetime64[ns]', freq=None)

DatetimeIndex(['2012-01-14', '2012-01-14'], dtype='datetime64[ns]', freq=None)

提供**格式**参数

`pd.to_datetime('2010/11/12', format = '%Y/%m/%d')`

Out: Timestamp('2010-11-12 00:00:00')

**多列组合日期时间**

[0.18.1    版更新]

把 DataFrame 里的整数或字符串列组合成 Timestamp Series

```python
df = pd.DataFrame({'year' : [2015, 2016],
                    'month' : [2, 3],
                    'day' : [4, 5],
                    'hour' : [2, 3]})
pd.to_datetime(df)
```

Out:

0   2015-02-04 02:00:00
1   2016-03-05 03:00:00
dtype: datetime64[ns]

#### 纪元时间戳

把整数或浮点数纪元时间转换为 Timestamp 与 DatetimeIndex

```python
pd.to_datetime([1349720105, 1349806505, 1349892905,
                1349979305, 1350065705], unit = 's')
```

Out:

DatetimeIndex(['2012-10-08 18:15:05', '2012-10-09 18:15:05',
               '2012-10-10 18:15:05', '2012-10-11 18:15:05',
               '2012-10-12 18:15:05'],
              dtype='datetime64[ns]', freq=None)

#### 时间戳序列

时间戳是定频的，用 **date_range(), bdate_range()** 函数即可创建 DatetimeIndex

```python
start = datetime.datetime(2011, 1, 1)
end = datetime.datetime(2012, 1, 1)
index = pd.date_range(start, end)
index
```

Out:

DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03', '2011-01-04',
               '2011-01-05', '2011-01-06', '2011-01-07', '2011-01-08',
               '2011-01-09', '2011-01-10',
               ...
               '2011-12-23', '2011-12-24', '2011-12-25', '2011-12-26',
               '2011-12-27', '2011-12-28', '2011-12-29', '2011-12-30',
               '2011-12-31', '2012-01-01'],
              dtype='datetime64[ns]', length=366, freq='D')

`pd.date_range(start, end, freq = 'W')`

Out:

DatetimeIndex(['2011-01-02', '2011-01-09', '2011-01-16', '2011-01-23',
               '2011-01-30', '2011-02-06', '2011-02-13', '2011-02-20',
               '2011-02-27', '2011-03-06', '2011-03-13', '2011-03-20',
               '2011-03-27', '2011-04-03', '2011-04-10', '2011-04-17',
               '2011-04-24', '2011-05-01', '2011-05-08', '2011-05-15',
               '2011-05-22', '2011-05-29', '2011-06-05', '2011-06-12',
               '2011-06-19', '2011-06-26', '2011-07-03', '2011-07-10',
               '2011-07-17', '2011-07-24', '2011-07-31', '2011-08-07',
               '2011-08-14', '2011-08-21', '2011-08-28', '2011-09-04',
               '2011-09-11', '2011-09-18', '2011-09-25', '2011-10-02',
               '2011-10-09', '2011-10-16', '2011-10-23', '2011-10-30',
               '2011-11-06', '2011-11-13', '2011-11-20', '2011-11-27',
               '2011-12-04', '2011-12-11', '2011-12-18', '2011-12-25',
               '2012-01-01'],
              dtype='datetime64[ns]', freq='W-SUN')

`pd.date_range(start, end, freq = 'BM')`

Out:

DatetimeIndex(['2011-01-31', '2011-02-28', '2011-03-31', '2011-04-29',
               '2011-05-31', '2011-06-30', '2011-07-29', '2011-08-31',
               '2011-09-30', '2011-10-31', '2011-11-30', '2011-12-30'],
              dtype='datetime64[ns]', freq='BM')

**bdate_range()** 工作日

```python
index = pd.bdate_range(start, end)
index
```

Out:

DatetimeIndex(['2011-01-03', '2011-01-04', '2011-01-05', '2011-01-06',
               '2011-01-07', '2011-01-10', '2011-01-11', '2011-01-12',
               '2011-01-13', '2011-01-14',
               ...
               '2011-12-19', '2011-12-20', '2011-12-21', '2011-12-22',
               '2011-12-23', '2011-12-26', '2011-12-27', '2011-12-28',
               '2011-12-29', '2011-12-30'],
              dtype='datetime64[ns]', length=260, freq='B')

### 时间索引

DatetimeIndex 主要用作 pandas 对象的索引；DatetimeIndex 类为时间序列做了很多优化

- 预计算了各种偏移量的日期范围，并在后台缓存，让后台生成后续日期范围的速度非常快【仅需抓取切片】

- 在 pandas 对象上使用 shift 与 tshift 方法进行快速偏移

- 合并具有相同频率的重叠 DatetimeIndex 对象的速度非常快【这点对快速数据对齐非常重要】

- 通过 year, month 等属性快速访问日期字段

- snap 等正则函数与超快的 asof 逻辑

```python
'''
以年提取数据
'''
print(df['2019'])
```

Out:

![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-11-11%20132759.png)

```python
'''
以月提取数据
'''
print(df['2019-06'])
```

Out:

![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-11-11%20132856.png)

```python
'''
以天提取数据
'''
print(df['2019-06-10': '2019-06-18'])
```

Out:

![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-11-11%20133001.png)

### DateOffset 对象

```python
ts = pd.Timestamp('2016-10-30 00:00:00', tz = 'Asia/Hong_Kong')
print(ts)
print('**')
print(ts + pd.Timedelta(days = 1))
print('**')
print(ts + pd.DateOffset(days = 1))
print('**')
print(pd.Timestamp('2018-01-05').day_name())
```

Out:

Timestamp('2016-10-30 00:00:00+0800', tz='Asia/Hong_Kong')

**

Timestamp('2016-10-31 00:00:00+0800', tz='Asia/Hong_Kong')

**

Timestamp('2016-10-31 00:00:00+0800', tz='Asia/Hong_Kong')

**

'Friday'

```python
two_business_days = 2*pd.offsets.BDay()
pd.Timestamp('2018-01-05') + two_business_days
```

Out: Timestamp('2018-01-09 00:00:00')

```python
d_1 = pd.Timestamp('2018-01-05')
print(d_1 + pd.offsets.Week())
print('-####')
d_2 = datetime.datetime(2008, 8, 18, 9, 0)
print(d_2 + pd.offsets.Week())
print(d_2 + pd.offsets.Week(weekday = 4))
print(d_2 - pd.offsets.Week())
```

Out:

Timestamp('2018-01-12 00:00:00')

-####

2008-08-25 09:00:00
2008-08-22 09:00:00

Timestamp('2008-08-11 09:00:00')

### 移位与延迟

整体向前或向后移动时间序列中的值

实现这一操作的方法是 **shift()**，该方法适用于所有 pandas 对象

```python
ts = pd.Series(range(len(rng)), index = rng)
print(ts.head())
print(ts.shift(-1))
```

Out:

2011-01-31    0
2011-02-28    1
2011-03-31    2
2011-04-29    3
2011-05-31    4
Freq: BM, dtype: int32

2011-01-31    1.0
2011-02-28    2.0
2011-03-31    3.0
2011-04-29    4.0
2011-05-31    NaN
Freq: BM, dtype: float64

```python
# freq: DateOffset, timedelta, time rule string; 可选参数，默认值= None
# 只适用于时间序列，若此参数存在，则会按照参数移动时间序列，数据值不发生变化
print(ts)
print(ts.shift(5, freq = pd.offsets.Day()))
```

Out:

2011-01-31    0
2011-02-28    1
2011-03-31    2
2011-04-29    3
2011-05-31    4
Freq: BM, dtype: int32

2011-02-05    0
2011-03-05    1
2011-04-05    2
2011-05-04    3
2011-06-05    4
dtype: int32

### Period 时期

Pandas 的 Period 可用定义一个时期，或者说具体的一个时段；有这个时段的起始时间 start_time, 终止时间 end_time 等属性信息，其参数 freq 和之前的 date_range 里的 freq 参数类似，可用取 'S', 'D' 等

```python
T = pd.Period('2018-12-15', freq = 'A')
print(T.start_time, T.end_time, T + 1, T)
```

Out: 2018-01-01 00:00:00 2018-12-31 23:59:59.999999999 2019 2018

```python
print(pd.Period('2013-1-9 11:22:33', freq = 'S'))
print(pd.Period('2013-1-9 11:22:33', freq = 'T'))
print(pd.Period('2013-1-9 11:22:33', freq = 'H'))
print(pd.Period('2013-1-9 11:22:33', freq = 'D'))
print(pd.Period('2013-1-9 11:22:33', freq = 'M'))
print(pd.Period('2013-1-9 11:22:33', freq = 'A'))
```

Out:

2013-01-09 11:22:33 \
2013-01-09 11:22 \
2013-01-09 11:00 \
2013-01-09 \
2013-01 \
2013

```python
df.index = pd.to_datetime(df.index).to_period('Y')  # 年
```

### 重采样

将时间序列从**一个频率转化为另一个频率进行**处理的过程，将高频率数据转化为低频率数据为**降采样**，低频率转化为高频率为**升采样**

pandas 提供了一个 **resample** 的方法来帮助我们实现频率转化

![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-06-05%20192634.png)

```python
'''
将数据以 W 星期，M 月，Q 季度，QS 季度的开始第一天开始，A 年，
10A 十年，10AS 十年聚合日期第一天开始的形式进行聚合
'''
print(df.resample('M').sum())
```

Out:

![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-11-11%20133257.png)

```python
'''
具体某列的数据聚合
'''
# 星期聚合，以0填充 nan
print(df.High.resample('W').sum().fillna(0))
```

Out:

![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-11-11%20133429.png)

```python
'''
某两列聚合
'''
print(df[['High', 'Low']].resample('W').sum().fillna(0))
```

Out:

![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-11-11%20133543.png)

```python
'''
某个时间段内，以 W 聚合
'''print(df['2019-06-10':'2019-06-18'].resample("M").sum().fillna(0))
```

Out:

![](D:\HANSHAN\Data%20Analysis\Pandas\Picture\屏幕截图%202022-11-11%20133728.png)

# Matplotlib

## Matplotlib 简介

一款用于数据可视化的 Python 软件包，支持跨平台运行，它能够根据 Numpy ndarray 数组来绘制 2D 图像，它使用简单、代码清晰易懂，深受广大技术爱好者喜爱

Matplotlib 提供了一套面向绘图对象变成的 API 接口，能够很轻松地实现各种图像的绘制，并且它可以配合 Python GUI 工具（如 PyQt, Tkinter 等）在应用程序中嵌入图形；同时 Matplotlib 也支持以脚本的形式嵌入到 IPython shell, Jupyter Notebook, web 应用服务器中使用

### 架构组成

由三个不同的层次结构组成，分别是脚本层、美工层和后端层

![](D:\HANSHAN\Data%20Analysis\Matplotlib\Picture\matplotlib架构图.jpg)

- 脚本层
  
  Matplotlib 结构中的最顶层；编写的绘图代码大部分代码都在该层运行，它的主要工作是负责生成图形与坐标系

- 美工层
  
  结构中的第二层，它提供了绘制图形的元素时的各种功能，例如，绘制标题，轴标签，坐标刻度等

- 后端层
  
  结构中的最底层，它定义了三个基本类，首先是 FigureCanbas（图层画布类）, 它提供了绘图所需的画布；其次是 Renderer（绘图操作类）, 它提供了在画布上进行绘图的各种方法；最后是 Event（事件处理类）, 它提供了用来处理鼠标和键盘事件的方法

### 图形组成

如图组成

<img title="" src="file:///D:/HANSHAN/Data Analysis/Matplotlib/Picture/Matplotlib图像构成.jpg" alt="" data-align="inline">

- Figure: 整个图形，可以把它理解成一张画布，它包括了所有的元素，比如标题、轴线等

- Axes: 绘制 2D 图像的实际区域，也称为轴域区，或者绘图区

- Axis: 坐标系中的垂直轴与水平轴，包含轴的长度大小、轴标签和刻度标签

- Artist: 在画布上看到的所有元素都属于 Artist 对象，比如文本对象（title, xlabel, ylabel）、Line2D 对象

## Matplotlib  figure 图形对象

在 matplotlib 中，面向对象编程的核心思想是创建图形对象 (figure object)；通过图形对象来调用其它的方法和属性，这样有助于我们更好地处理多个画布；在这个过程中，pyplot 负责生成图形对象，并通过该对象来添加一个或多个 axes 对象（即绘图区域）

matplotlib 提供了 matplotlib.figure 图形类模块，它包含了创建图形对象的方法；通过调用 pyplot 模块中 figure() 函数来实例化 figure 对象

## Matplotlib subplot() 函数

plt 模块提供了一个 subplot() 模块，它可以均等地划分画布，该函数的参数格式如下

`plt.subplot(nrows, ncols, index)`

nrows 与 cols 表示要划分几行几列的子区域，nrows*nclos 表示子图数量，index 的初始值为1，用来选定具体的某个子区域

> 例如，subplot(233) 表示在当前画布的右上角创建一个两行散列的绘图区域，同时选择在第3个位置绘制子图

## Matplotlib subplots() 函数

subplots() 既创建了一个包含子图区域的画布，又创建了一个 figure 图形对象

而 subplot() 只是创建了一个包含子图区域的画布

`fig, ax = plt.subplots(nrows, ncols)`

nrows 与 ncols 表示两个整数参数，它们指定子图所占的行列数

函数的返回值是一个元组，包括一个图形对象和所有的 axes 对象

其中 axes 对象的数量等于 nrows*ncols，且每个 axes 对象均可通过所有制访问（从1开始）

```python
fig,a =  plt.subplots(2,2)
x = np.arange(1,5)

#绘制平方函数
a[0,0].plot(x,x*x)
a[0,0].set_title('square')

#绘制平方根图像
a[0,1].plot(x,np.sqrt(x))
a[0,1].set_title('square root')

#绘制指数函数
a[1,0].plot(x,np.exp(x))
a[1,0].set_title('exp')

#绘制对数函数
a[1,1].plot(x,np.log10(x))
a[1][1].set_title('log')
plt.show()
```

Out:

![](D:\HANSHAN\Data%20Analysis\Matplotlib\Picture\屏幕截图%202022-06-07%20150249.png)

## Matplotlib subplot2grid() 函数

能够在画布的特定位置创建 axes 对象 (即绘图区域)，且使用不同数量的行、列来创建跨度不同的绘图区域

与 subplot() 和 subplots() 函数不同，**subplot2grid()** 函数以非等分的形式对画布进行切分，并按照绘图区域的大小来展示最终绘图结果

`plt.subplot2grid(shape, location, rowspan, colsoan)`

- shape: 把该参数值规定的网格区域作为绘图区域

- location: 在给定的位置绘制图形，初始位置 (0, 0) 表示第1行第1列

- rowspan/colspan: 这两个参数用来设置让子区跨越几行几列

## PyLab

面向 matplotlib 的绘图库接口，其语法和 MATLAB 十分相近；它和 Pyplot 模块都能实现 matplotlib 的绘图功能；PyLab 是一个单独的模块，随 matplotlib 软件包一起安装

```python
# 导包方式
import pylab
from pylab import *
```

## 3D 绘图

3D 绘图程序包，比如 mpl_toolkits.mplot3d，通过调用该程序包的一些接口可以绘制 3D 散点图、3D 曲面图、3D 线框图等

首先创建一个三维绘图区域，plt.axes() 函数提供了一个参数 projection，将其参数值设置为 '3d'

![](D:\HANSHAN\Data%20Analysis\Matplotlib\Picture\屏幕截图%202022-06-07%20172143.png)

![](D:\HANSHAN\Data%20Analysis\Matplotlib\Picture\屏幕截图%202022-06-07%20172214.png)

![](D:\HANSHAN\Data%20Analysis\Matplotlib\Picture\屏幕截图%202022-06-07%20172257.png)

# Sklearn

sklearn（全称 Scikit-Learn）是基于 Python 语言的机器学习工具；它建立在 NumPy, SciPy, Pandas 和 Matplotlib 之上，里面的 API 的设计非常好，所有对象的接口简单，很适合新手上路

sklearn 里面有六大任务模块：分别是分类、回归、聚类、降维、模型选择和预处理

1. 分类：Classification

2. 回归：Regression

3. 聚类：Clustering

4. 降维：Dimensionality Reduction

5. 模型选择：Model Selection

6. 预处理：Preprocession

> - `from sklearn.linear_model import SomeClassifier`
> 
> - `from sklearn.linear_model import SomeRegressor`
> 
> - `from sklearn.cluster import SomeModel`
> 
> - `from sklearn.decomposition import SomeModel`
> 
> - `from sklearn.model_selection import SomeModel`
> 
> - `from sklearn.preprocession import SomeModel`
> 
> SomeClassifier, SomeRegressor, SomeModel 其实都叫作估计器/预估器 (estimator)，sklearn 里【万物皆预估器】

# Python

## random

| methods                | explanation                          |
| ---------------------- | ------------------------------------ |
| random()               | 返回 $0 \le n \le 1$ 之间的随机实数 $n$       |
| choice(seq)            | 从序列 seq 中返回随机的元素                     |
| getrandbits(n)         | 以长整型形式返回 $n$ 个随机位                    |
| shuffle(seq[, random]) | 原地指定 seq 序列                          |
| sample(seq, n)         | 从序列 seq 中选择 $n$ 个随机且独立的元素            |
| randint(10， 100)       | 随机生一个整数 int 类型，可以指定这个整数的范围，同样有上限和下限值 |

## format

fotmat 作为 Python 的的格式字符串函数，主要通过字符串中的花括号 {}，来识别替换字段，从而完成字符串的格式化

```python
print("我叫{}，今年{}岁。".format("小蜜", 18))
#我叫小蜜,今年18岁。
#花括号的个数决定了，参数的个数。但是花括号的个数可以少于参数。
print("我喜欢{}和{}".format("乒乓球", "羽毛球", "敲代码"))
#我喜欢乒乓球和羽毛球。
"""
花括号多于参数的个数，则会报错。
"""
```

通过数字参数传入位置参数  

传入参数注意以下事项

- 数字必须是大于0的整数
- 带数字的替换字段可以重复
- 数字形式的简单字段名相当于把字段当成一个序列形式。通过索引的形式进行一一取值

```python
#通过数字索引传入参数
print("名字{0},家住{1}").format("橙留香", "水果村")
#带数字的替换1字段可以重复

​```python
print("我爱{0}。\n他爱{1}。\n{0}爱{1}".format("灰太狼", "红太狼")
"""
我爱灰太狼
他爱红太狼
灰太狼爱红太狼
"""

"""
数字形式的简单字段名相当于把字段当成一个序列形式。通过索引的形式进行一一取值
"""
print("小明喜欢{1},{2}和{0}".foramt("海绵宝宝", "机器猫", "海贼王", "火影忍者", "龙珠"))
#小明喜欢机器猫，海贼王，和海绵宝宝
```

