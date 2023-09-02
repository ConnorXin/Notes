# 筹集款项

```python
def collect_contributions(n):
    if n <= 100:
        return 100
    else:
        friends = 10
        sum = 0
        for i in range(friends):
            sum += collect_contributions(n/10)
    return sum
```

```python
collect_contributions(1000)·
```

    1000

```python
def collect_contribution(n):
    if n < 100:
        return 100
    else:
        friend = 10
        sum = 0
        for i in range(friend):
            sum += collect_contribution(n/10)
    return sum
```

# 斐波那契数列

```python
def fib_rec(n):
    if n <= 1:
        f = n
    else:
        f = fib_rec(n-1) + fib_rec(n-2)
    return f
```

```python
fib_rec(24)
```

    46368

```python
def fib_rec(n):
    if n <= 1:
        f = n
    else:
        f = fib_rec(n-1) + fib_rec(n-2)
    return f
```

# 回文判断

```python
def isPal(s):
    if len(s) <= 1:
        return True
    else:
        return s[0] == s[-1] and isPal(s[1:-1])
```

```python
s = 'doggod'
result = isPal(s)
print(result)
```

    True

```python
s = 'Jinx'
result = isPal(s)
print(result)
```

    False

# x的n次幂

```python
def power(x,n):
    if n == 1:
        return x
    if n%2 == 0:
        result = power(x,n/2)
#power(x,n/2)
    else:
        result = power(x,(n+1)/2)
#power(x,(n-1)/2)
    return result
```

```python
power(3,2)
```

    3

# 全排列

```python
def permutation(str):
    lenstr = len(str)
    if lenstr < 2:
        return str
    else:
        result = []
        for i in range(lenstr):
            ch = str[i]    #索引每一个输入字符，存到变量ch中
            rest = str[0:i] + str[i+1:lenstr]  #将去除该字符后剩余的串存储于变量rest中
            for s in permutation(rest):  #递归函数
                result.append(ch + s)
        return result
```

```python
def p(str):
    lenstr = len(str)
    if lenstr < 2:
        return str
    else:
        result = []
        for i in range(str):
            ch = str[i]
            rest = str[0:i] + str[i+1:lenstr]
            for s in p(rest):
                result.append(ch + s)
        return result
```

```python
arr1 = [3,6,9,12,15]
arr2 = [2,4,6,7,13,14]
i,j = 0,0
li =[]
while i < len(arr1) and j < len(arr2):
    if arr1[i] <= arr2[j]:
        li.append(arr1[i])
        i += 1
    else:
        li.append(arr2[j])
        j += 1
li += arr1[i:] + arr2[j:]
print(li)
```

    [2, 3, 4, 6, 6, 7, 9, 12, 13, 14, 15]

# 算法 (Algorithm)


## 算法的定义

算法就是按照一定步骤解决问题的办法。这个定义里面蕴含了算法的两个重要属性：

1. 算法一般包括一系列**有限**的步骤，这些步骤能快速完成。
2. 算法要能**正确**给出具体问题的解。

## 算法的性质

- 有穷性
- 确定性
- 可行性(可以机械地一步一步执行基本操作步骤)

## 算法的效率

算法的效率主要体现两方面：一是算法运行的时间（*时间复杂度*）；二是算法执行过程中所占用的存储空间（*空间复杂度*）

1. 时间复杂度
   
   - **时间频度**
     代表一个算法中的语句执行次数
   - **时间复杂度**
     表示的并不是算法真实的执行时间 ，而是表示代码执行时间的增长变化趋势
     常见的时间复杂度：
     O(1)<O(logn)<O(n)<O(nlogn)<O(n^2)<O(n^2logn)<O(n^3)
   
   快速地判断算法复杂度(适用于绝大多数简单情况)
   
   - 确定问题规模n
   - 循环减半过程：logn
   - k层关于n的循环：n^k
     复杂情况：根据算法执行过程判断

2. 空间复杂度 对一个算法所需存储空间的量度
   
   ### 常用算法时间和空间复杂度汇总

| 排序算法 | 平均时间复杂度  | 最优时间复杂度  | 最差时间复杂度  | 空间复杂度    |
| ---- | -------- | -------- | -------- | -------- |
| 选择排序 | O(n^2)   | O(n^2)   | O(n^2)-  | O(1)     |
| 插入排序 | O(n^2)   | O(n)     | O(n^2)   | O(1)     |
| 冒泡排序 | O(n^2)   | O(n^2)   | O(n^2)   | O(1)     |
| 快速排序 | O(nlogn) | O(nlogn) | O(n^2)   | O(nlogn) |
| 堆排序  | O(nlogn) | O(nlogn) | O(nlogn) | O(1)     |
| 归并排序 | O(nlogn) | O(nlogn) | O(nlogn) | O(n)     |

# 递归

> 一个函数或是过程的定义中，包含了调用自身的语句，无论是直接或是简接。

创建递归函数时，通常有三个主要结构需要考虑：**边界条件**、**递归前进阶段**、**递归返回阶段**

*实例*

当你的朋友听到你的这个安排,应该会比较容易接受这个任务。因为,你没有要他们直接掏腰包出钱,而且还告诉他们该如何去完成这项任务。假如王某某是你的这10个朋友中的其中一位,那么他会按照你同样的步骤去执行他的10万元的筹款计划。第一,他会找到他的10个朋友;第二,王某某的这10个朋友筹款额度是1万元;第三,王某某也同样告诉他的这10个朋友,应该用与他自己相同的策略去筹款。

也许到这儿,读者会非常怀疑这个办法是否真的能筹集到100万元。因为,大家似乎都是在说着一个故事,然后依次去传递故事,但并没有人真正掏钱,那么如何能完成目标呢?这里我们需要做一个额外的限定,即当筹款的目标款数小于等于100元时,就不再继续往下传递这个故事,而是需要接受这个任务的人从自己的口袋拿出那100元,并把钱送给向他发出募集请求的人。

***递归程序***

```python
def collect_contributions(n):   #n为需要筹集的款数
    if n <= 100:
        return 100 #需要此人捐出100元
    else:
        #寻找10个朋友
        friends = 10
        sum = 0
        for i in range(friends):
         #从这10个朋友中分别募集n/10元
            sum += collect_contributions(n/10)
    return sum  #返回从10个朋友募集到的资金
collect_contributions(1000)
```



# 分治算法

## 归并排序是分治算法的代表性应用之一

归并的过程

分成的两部分需要有序的（一次归并）

    def merge(li, low, mid, high):
        i = low 
        j = mid +1
        ltmp = []
        while i <= mid and j <= high: #只要左右两边都有数
            if li[i] < li[j]:
                ltmp.append(li[i])
                i += 1
            else:
                ltmp.append(li[j])
                j += 1
        # while执行完，肯定有 一部分没数了
        while i <= mid:
            ltmp.append(li[i])
            i += 1
        while j <= high:
            ltmp.append(li[j])
            j += 1
        li(low:high+1] = ltmp

归并排序的实现



# 排序算法
内置排序函数：sort()
## 冒泡排序
> 基本思想：
- 列表每两个相邻的数，如果前面比后面大，则交换这两个数
- 一趟排序完成后，则无序区减少一个数，有序区增加一个数

代码

1. 升序


    def bubble_sort(li):
        for i in range(len(li)-1):
            for j in range(len(li)-i-1):
                if li[j] > li[j+1]:
                    li[j],li[j+1] = li[j+1],li[j]

2. 降序


    def bubble_sort(li):
        for i in range(len(li)-1):
            for j in range(len(li)-i-1):
                if li[j] < li[j+1]:
                    li[j],li[j+1] = li[j+1],li[j]

时间复杂度：O(n^2)
## 选择排序
> 从无序的数组中，每次选择最小或最大的数据，从无序数组中放到有序数组的末尾，以达到排序的效果。

> 递增排序开始时，先遍历未排序的数组，找到最小的元素。然后，把最小的元素从未排序的数组中删除，添加到有序数组的末尾。

代码

    def select_sort_simple(li):
        li_new = []
        for i in range(len(li)):
            min_val = min(li)
            li_new.append(min_val)
            li.remove(min_val)
        return li_new

更好的代码

    def select_sort(li):
        for i in range(len(li)-1):
            min_loc = i
            for j in range(i+1,len(li)):
                if li[j] < li[min_loc]:
                    min_loc = j
            li[i],li[min_loc] = li[min_loc],li[i]

时间复杂度 O(n^2)
## 插入排序
代码

    def insert_sort(li):
        for i in range(1,len(li)): # i 表示摸到的牌的下标
            tmp = li[i]
            j = i - 1  # j 指的是手里的牌的下标
            while j >= 0 and li[j] > tmp:
                li[j+1] = li[j]
                j -= 1
            li[j+1] = tmp

时间复杂度：O(n^2)
## 快速排序
基本代码框架

    def quick_sort(data,left,right):
        if left < right:
            mid = partition(data,left,right)   # 返回第一个元素归位后的下标
            quick_sort(data,left,mid - 1)   #mid左边进行排序
            qucik_sort(data,mid + 1,right)    #mid右边进行排序

代码

    def partition(li,left,right):
        tmp = li[left]
        while left < right: 
            while left < right and li[right] >= tmp:  #从右边找比tmp小的数
                right -= 1   #往左走一步
            li[left] = li[right]  #把右边的值写到左边空位上
            while left < right and li[left] <= tmp:
                left += 1
            li[right] = li[left]  #把左边的值写到右边空位上
        li[left] = tmp  #把tmp归位

时间复杂度 O(nlogn)
## 堆排序
> 树 是一种数据结构

>树是由n个节点组成的集合；如果n=0，那这是一颗空树；如果n>0，那存在1个节点作为树的根节点，其他节点可以分为m个集合，每个集合本身又是一颗树

> 二叉树 度不超过2的树

> 满二叉树 一个二叉树，如果每一个层的结点数都达到最大值，则这个二叉树就是满二叉树

> 完全二叉树 叶节点只能出现在最下层和次下层，并且最下面一层的结点都集中在该层最左边的若干位置的二叉树

> 堆 一种特殊的完全二叉树结构

> 大根堆 一颗完全二叉树，满足任一节点都比其孩子节点大

> 小根堆 一颗完全二叉树，满足任一节点都比其孩子节点小

堆排序过程

1. 建立堆
2. 得到堆顶元素,为最大元素
3. 去掉堆顶,将堆最后一个元素放到堆顶,此时可通过一次调整重新使堆有序
4. 堆顶元素为第二大元素
5. 重复步骤3,直到堆变空

代码

    def sift(li,low,high): #三个参数分别表示列表,堆的根节点位置,堆的最后一个元素的位置
        i = low
        j = 2 * i + 1
        de

# 查找算法
> 在一些数据元素中，通过一定的方法找出与给定关键字相同的数据元素的过程

> 列表查找（线性表查找）：从列表中查找指定元素

> 内置列表查找函数：index()
## 顺序查找
> 也叫线性查找，从列表第一个元素开始，顺序进行搜索，直到找到元素或搜索到列表最后一个元素为止

代码

    def linear_search(li,val):
        for ind,v in enumerate(li):
            if v == val:
                return ind
        else:
            return None

时间复杂度  O(n)

## 二分查找
> 又叫折半查找，从**有序**列表的初始候选区li[0,n]开始，通过对待查找的值与候选区中间值的比较，可以使候选区减少一半

代码
(列表已排好序)

    def binary_search(li, val):
        left = 0
        right = len(li) - 1
        while left <= right:  # 候选区有值
            mid = (left + right) // 2
            if li[mid] == val:
                return mid
            elif li[mid] > val:  # 待查找的值在mid左侧
                right = mid - 1
            else:  # li[mid] < val 待查找的值在mid右侧
                left = mid +1
        else:
            return None

时间复杂度 O(logn)

# 哈希算法
> 也是一种查找算法，可以说哈希算法是最快的查找算法。对于查找问题而言，哈希算法一直是首选算法

## 除法哈希算法
**哈希函数公式  h(x) = x mod m**
