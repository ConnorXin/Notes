# 数据采集简介

为什么要进行数据采集

> 当前，全球大数据进入加速发展时期。大数据时代，谁掌握了足够的数据，谁就有可能掌握未来，现在的数据采集就是将来的资产积累。
> 
> 数据采集是数据挖掘的**前站**。数据采集是大数据价值挖掘中重要的一环，其后的分析挖掘都是建立在数据采集的基础上。

## 网络爬虫简介

对于大数据技术来说，爬虫可以用来**收集数据**,这也是爬虫最直接、最常用的使用方法。

### 概念与原理

1. 通用网络爬虫

2. 聚焦网络爬虫

3. 增量式网络爬虫

4. 深层网络爬虫

### 法律和道德问题

以下两种数据是不能爬取的，更不能用于商业用途。

> * 个人隐私数据，如姓名、手机号码、年龄、血型、婚姻情况等，爬取此类数据将会触犯个人信息保护法。
> 
> * 明确禁止他人访问的数据，例如，用户设置过权限控制的账号、密码或加密过的内容。
> 
> 另外，还需要注意版权相关问题，有作者署名的受版权保护的内容不允许爬取后随意转载或用于商业用途。

**法律法规**

全国人民代表大会常务委员会在2016年11月7日通过了《中华人民共和国网络安全法》，2017年6月1日正式实施

**robots.txt协议**

网站域名/robots.txt

## 网络爬虫的基本流程

1. **准备工作**
   
   通过浏览器查看分析目标网页，学习编程基础规范。观察所要爬取的网页是属于静态网页还是动态网页，从而确定使用不同的爬取策略。

2. **获取数据(网页源代码)**
   
   通过HTTP库向目标站点发起请求，请求可以包含额外的header等信息，如果服务器能正确相应，会得到一个Response，便是所要获取的页面内容。

3. **解析内容**
   
   得到的内容是HTML、json等格式，可以用页面解析库、正则表达式等进行解析。

4. **保存数据**
   
   保存形式多样，可以存为文本，也可以保存到数据库，或者保存特定格式的文件。

## 网络爬虫技术

- 关于获取网页，这里主要介绍了Python的四个第三方模块，一个是urlib，一个是requests，一个是selenium(webdriber,chrome)，另一个是爬虫框架Scrapy(选讲)。

- 关于解析网页内容，这里主要介绍了3种方式——正则表达式，XPath和BeautifulSoup。

两种网页获取方式和3种网页解析方式可以自由搭配，随意使用。

## 注意事项

- **注意事项一**
  
  在网络爬虫代码编写时，访问网站时通常会需要证书验证。为了保证每次网络爬虫程序的正常运行，所有网络爬虫的程序的前面都加上下面两行代码：
  
  ```python
  #全局取消证书验证
  import ssl
  ssl.__create__default__https__context = ssl.__create__unverified__context
  ```

- 注意事项二
  
  由于实际网站的网页会经常发生变动，所以此讲义中的代码很可能会遇到这样一种情形：前一段时间爬取网站信息可以顺利地爬取到数据，但间隔一段时间以后，发现同样的代码再也无法完整获取到所要的信息，甚至运行时会出现错误。这些问题的出现均是因为网页源代码的一些细节发生变动，或者网页结构发生变动。



# 字符串

## 字符串简介

在Python中，字符串属于**不可变有序序列**，使用*单引号、双引号、三(单\双)引号*作为定界符

> 如果需要判断一个变量是否为字符串，可以使用内置方法 isinstance() 或 type()。除了支持 Unicode 编码的 str 类型之外 ，Python 还支持字节串类型 bytes, str 类型字符串可以通过 encode() 方法使用指定的字符串编码格式编码成为 bytes 对象，而 bytes 对象则可以通过 decode() 方法使用正确的编码格式解码成为 str 字符串。

`type('中国')`   Out: <class 'str'>

`type('中国'.encode('gbk')) #编码成字节串，采用GBK编码格式`

Out: <class 'bytes'>

`bytes #bytes也是Python的内置类`   Out: <class 'bytes'>

`isinstance('中国',str)`   Out: True

`type('中国') == str`    Out: True

`type('中国'.encode()) == bytes`   Out: True

`'中国'.encode() #默认使用UTF-8进行编码`  Out: b'\xe4\xb8\xad\xe5\x9b\xbd'

`_.decode() #默认使用UTF-8进行编码`  Out: '中国'

`bytes('董付国','gbk')`  Out: b'\xb6\xad\xb8\xb6\xb9\xfa'

`str(_,'gbk')`  Out: '董付国'

## 字符串编码格式简介

最早的字符串编码是美国标准信息交换码 ASCII，仅对 10 个数字、26
个大写英文字母、26 个小写英文字母及一些其他符号进行了编码。
ASCII 码采用一个字节来对字符进行编码，最多只能表示 256 个符号。

随着信息技术的发展和信息交换的需要，各国的文字都需要进行编码，
不同的应用领域和场合对字符串编码的要求也略有不同，于是又分别设
计了多种不同的编码格式，常见的主要有 UTF-8、UTF-16、UTF-32、
GB2312、GBK、CP936、base64、CP437 等。UTF-8 对全世界所有国
家用到的字符进行了编码，以一个字节表示英语字符 (兼容 ASCII)，
以*3 个字节表示中文*，还有些语言的符号使用 2 个字节 (如俄语和希腊
语符号) 或 4 个字节。GB2312 是我国制定的中文编码，使用一个字节
表示英语，2 个字节表示中文；GBK 是 GB2312 的扩充，而 CP936 是
微软公司在 GBK 基础上开发的编码方式。<u>GB2312、GBK 和 CP936
都是使用 2 个字节表示中文。</u>

## 字符串常用操作

Python 字符串对象提供了大量方法用于字符串的检测、替换和排版等操作，另外还有大量内置函数和运算法也支持对字符串的操作。使用时需要注意的是，字符串对象是不可变的，所以**字符串对象提供的涉及字符串“修改”的方法都是返回修改后的新字符串，并不对原字符串做任何修改，无一例外**。

### find(), rfind(), index(), rindex(), count()

- find() 和 rind() 方法分别用来查找一个字符串在另一个字符串指定范围（默认是整个字符串）中首次和最后一次出现的位置，**如果不存在则返回-1**

- index() 和 rindex() 方法用来返回一个字符串在另一个字符串指定范围中首次和最后一次出现的位置，**如果不存在则抛出异常**

- count() 方法用来返回一个字符串在另一个字符串中出现的次数，**如果不存在则返回0**

```python
s = "apple, peach, banana, peach, pear"
s.find('peach') # 返回第一次出现的位置
```

Out: 6

`s.find('peach', 7) # 从指定位置开始查找`   Out: 19

`s.find('peach', 7, 20) #在指定范围中进行查找`   Out: -1

`s.rfind('p') # 从字符串尾部向前查找`   Out: 25

`s.index('p') # 返回首次出现的位置`  Out: 1

`s.index('pe')`   Out: 6

`s.index('pear')`   Out: 25

`s.index('ppp') # 指定子字符串不存在时抛出异常`  Out: ValueError: substring not found

`s.count('p') # 统计子字符串出现的次数`  Out: 5

`s.count('ppp') # 不存在时返回0`  Out: 0

### split(), rsplit()

字符串对象的 split() 和 rsplit() 方法分别用来以指定字符为分隔符，从字符串左端和右端开始将其分割成多个字符串，并返回包含分割结果的列表。

> ```python
> s = "apple, peach, banana, pear"
> s.split(',')
> ```
> 
> Out: ['apple','peach','banana','pear']
> 
> ```python
> s = '2014-10-31'
> t = s.split('-')
> t
> ```
> 
> Out: ['2014','10','31']
> 
> `list(map(int,t)) # 将分割结果转换为整数`  Out: [2014,10,31]

对于 split() 和 rsplit() 方法，如果不指定分隔符，则字符串中的任何空白符号（包括空格、换行符、制表符等）的连续出现都将被认为是分隔符，返回包含最终分割结果的列表。

> ```python
> s = 'hello world \n\n My name is Dong'
> s.split()
> ```
> 
> Out: ['hello','world','My','name','is','Dong']

另外，split() 和 rsplit() 方法允许指定最大分隔次数（注意，并不是必须必须分隔这么多次）

> ```python
> s = '\n\nhello\t\t world \n\n\n My name is Dong'
> s.split(maxsplit = 1) # 分隔1次
> ```
> 
> Out: ['hello','world \n\n\n My name is Dong']
> 
> `s.rsplit(maxsplit = 1) # 分隔1次`   
> 
> Out: ['\n\nhello\t\t world \n\n\n My name is','Dong']
> 
> `s.split(maxsplit = 2) # 分隔2次`
> 
> Out: ['hello','world','My name is Dong']
> 
> `s.split(maxsplit = 10) # 最大分隔次数可以大于实际可分隔次数`
> 
> Out: ['hello','world','My','name','is','Dong']

调用 split() 方法如果不传递任何参数，将使用任何空白字符作为分隔符，如果字符串存在连续的空白字符，split() 方法将作为一个空白字符对待。但是，明确传递参数指定 split() 使用的分隔符时，情况略有不同。

> ```python
> s = 'a,,,b,,ccc'
> s.split(',') # 每个都被作为独立的分隔符
> ```
> 
> Out: ['a','','','b','','ccc']
> 
> ```python
> s = 'a\t\t\tbb\t\tccc'
> s.split('\t')
> ```
> 
> Out: ['a','','','bb','','ccc']
> 
> `s.split() # 连续多个制表符被作为一个分隔符`
> 
> Out: ['a','bb','ccc']

### join()

字符串的 join() 方法用来将列表中多个字符串进行连接，并在相邻两个字符串之间插入指定字符，返回新字符串。

> ```python
> li = ['apple','peach','banana','pear']
> sep = ','
> sep.join(li)
> ```
> 
> Out: 'apple,peach,banana,pear'
> 
> `'/'.join(li)`
> 
> Out: 'apple/peach/banana/pear'

使用 split() 和 join() 方法可以删除字符串中多余的空白字符，如果有连续多个空白字符，只保留一个。

> ```python
> s = 'aaa      bb   c d e  fff'
> ' '.join(s.split())
> ```
> 
> Out: 'aaa bb c d e fff'

### replace()

字符串方法 replace() 用来替换字符串中指定字符或子字符串的所有重复出现，每次只能替换一个字符或一个字符串，把指定的字符串参数作为一个整体对待，类似于Word,Wos,记事本等文本编辑器的查找与替换功能。**该方法不修改原字符串，而是返回一个新字符串**。

> ```python
> s = 'abc....abc..ab'
> s.replace('abc','ABC')
> ```
> 
> Out: ABC....ABC..ab
> 
> `print(s.replace('ab','AB'))`  Out: ABc....ABc..AB

### strip(), rstrip(), lstrip()

这几个方法分别用来删除两端、右端或左端连续的空白字符或指定字符。

> ```python
> s = '   abc   '
> s.strip() # 删除空白字符
> ```
> 
> Out: 'abc'
> 
> `'\n\nhello world \n\n'.strip()`
> 
> Out: 'hello world'
> 
> `'aaaassddf'.strip('a')`
> 
> Out: 'ssddf'
> 
> `'aaaassddfaaaa'.rstrip('a')`
> 
> Out: 'aaaassddf'

这 3个函数的参数指定的字符串并不作为一个整体对待，而是在原字符串的两侧、右侧、左侧删除参数字符串中包含的所有字符，一层一层地从外往里扒。

> ```python
> s = 'aabbccddeeffg'
> s.strip('af') # 字母f不在字符串两侧，所以不删除
> ```
> 
> Out: 'bbccddeeffg'
> 
> `s.strip('gaf')`   Out: 'bbccddee'
> 
> `s.strip('gaef')`   Out: 'bbccdd'

### startswith(), endswith()

这两个方法用来判断字符串是否以指定字符串开始或结束，可以接受两个整数参数来限定字符串的检测范围。

> ```python
> s = 'Beautiful is better than ugly.'
> s.startswith('Be') # 检测整个字符串
> ```
> 
> Out: True
> 
> `s.startswith('Be',5) # 指定检测范围的起始范围`  Out: False
> 
> `s.startswith('Be',0,5) # 指定检测范围的起始和结束位置`   Out: True

另外，这两个方法还可以接受一个字符串元素作为参数来表示前缀或后缀。

例如，下面的代码可以列出指定文件夹下所有扩展名为bmp, jpg 或 gif 的图片。

> ```python
> import os
> [filename for filename in os.listdir(r'D:\\') if filename.endswith
> (('.bmp','.jpg','.gif'))]
> ```

## 中英文分词

Python 扩展库 jieba 和 snownlp 很好地支持了中英文分词，可以使用 pip 命令进行安装。在自然语言处理领域经常需要对文字进行分词，分词的准确度直接影响后续文本处理和挖掘算法的最终效果。

```python
import jieba
s = '分词的准确度直接影响了后续文本处理和挖掘算法的最终效果'
jieba.cut(s) # 使用默认词库进行分词
```

Out: <generator object Tokenizer.cut at 0x103951350>

`list(_)`

Out:['分词','的','准确度',’直接','影响','了','后续','文本处理','和','挖掘','算法','的','最终','效果']

```python
jieba.add_word('花纸杯')
list(jieba.cut('花纸杯'))
```

Out: ['花纸杯’]

# 文件操作

## 文件夹操作

### 1. os模块

Python 标准库的 os 模块除了提供使用操作系统功能和访问文件系统的简便之外, 还提供了大量文件与文件夹操作的方法. 在网络爬虫及数据分析中, 主要用到的就是 "文件与文件夹操作的方法" , 并用于保存爬取的内容.

| 命令                         | 解释                                                                                                                                                                   |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| path = os.getcwd()         | 查看当前工作目录                                                                                                                                                             |
| os.chdir(path)             | 把 path 设为当前工作目录                                                                                                                                                      |
| os.listdir(path)           | 查看 path 目录下的文件和文件夹目录列表                                                                                                                                               |
| **os.makedirs(file_path)** | **创建多级目录, 会根据需要自动创建中间缺失的目录, 其中 file_path 为: 1. 文件夹的绝对路径, 如 file_path = '/User/mac/Documents/图书馆/其它扫描'; 2. 文件夹的绝对路径, 如 os.makedirs('Test/test1') 会在当前工作目录下创建相应的文件夹.** |
| os.rename(src,dst)         | 重命名文件或目录, 若目标文件已存在则抛出异常. 而 os.replace(scr,dst) 也可以重命名文件或目录, 若目标文件已存在则直接覆盖.                                                                                           |

### 2. os.path 模块

os.path 模块提供了大量用于路径判断、切分、连接以及文件夹遍历的方法.

| 命令                        | 解释              |
| ------------------------- | --------------- |
| **os.path.exists(path)**  | **判断文件是否存在**    |
| os.path.getsize(filename) | 查看文件的大小         |
| os.path.isabs(path)       | 判断 path 是否为绝对路径 |
| isdir(path)               | 判断 path 是否为文件夹  |
| isfile(path)              | 判断 path 是否为文件   |

### 3. shutil 模块

shutil 模块也提供了大量的方法支持文件和文件夹操作

| 命令                                    | 解释                                                 |
| ------------------------------------- | -------------------------------------------------- |
| shutil.copytree('Test1','Test1/Test') | 递归复制文件夹, 如该代码为将文件夹 Test1 复制到它自己的文件里面, 并将其命名为 Test. |
| shutil.copy(src,sdt)                  | 复制文件, 新文件具有同样的文件属性, 如果目标文件已存在则抛出异常.                |
| shutil.move(src,dst)                  | 移动文件或递归移动文件夹, 也可以给文件和文件夹重命名.                       |
| **shutil.rmtree(path)**               | **递归删除文件夹.**                                       |

## txt 文本文件操作

对文本文件的操作流程通常分为以下几个步骤

- 打开文件并创建文件对象; 内置函数 open()

- 通过该文件对象对文件内容进行读取、写入、删除、修改等操作

- 关闭并保存文件内容; 内置函数 close()

> ***open() 函数***
> 
> Python 内置函数 open() 可以用指定模式打开指定文件并创建文件对象, 由于很多参数都有默认值, 在使用时只需给特定的参数传值即可.
> 
> **常用方法为** `open(file, mode = 'r')`
> 
> - 参数 file 指定要打开或创建的文件名称, 如果该文件不在当前目录中, 可以使用相对路径或绝对路径, 为了减少路径中分隔符的输入可以使用原始字符串.
> 
> - 参数 mode 的取值指定打开文件后的处理方式, 默认为 "文本只读模式" . 以不同方式打开文件时, 文件指针的初始位置略有不同. 以 "只读" 和 "只写" 模式打开时文件指针的初始位置是文件头, 以 "追加" 模式打开时文件指针的初始位置为文件尾. 以 "只读" 方式打开的文件无法进行任何写操作, 反之亦然.
> 
> **文件打开模式**
> 
> | 命令  | 解释                                        |
> | --- | ----------------------------------------- |
> | r   | 读模式 (默认模式, 可省略) , 如果文件不存在则抛出异常            |
> | w   | 写模式, 如果文件已存在, 先清空原有内容写模式. 如果文件不存在, 则创建该文件 |
> | x   | 创建新文件, 如果文件已存在则抛出异常                       |
> | a   | 追加模式, 不覆盖文件中原有内容                          |
> | +   | 读 写模式 (可与其他模式组合使用)                        |
> 
> **如果执行正常, open() 函数返回一个可迭代的文件对象, 通过该文件对象可以对文件进行读写操作, 如果指定文件不存在、访问权限不够、磁盘空间不够或其他原因导致创建文件对象失败则抛出异常. 下面代码分别以读 写方式打开了两个文件并创建了与之对应的文件对象**
> 
> ```python
> f1 = open('file1.txt','r')
> f2 = open('file2.txt','w')
> ```

> ***close() 函数***
> 
> 当对文件内容操作完以后, 一定要关闭文件对象, 这样才能保证所做的任何修改都被保存到文件中.
> 
> `f1.close()`
> 
> 需要注意的是, 即使写了关闭文件的代码, 也无法保证文件一定能够正常关闭. 例如如果在打开文件之后和关闭文件之前发生了错误导致程序崩溃, 这时文件就无法正常关闭. **在管理文件对象时推荐使用 with 关键字, 可以有效地避免这个问题**.
> 
> 在实际开发中, 读写文件应优先考虑使用上下文管理语句 with , 关键字 with 可以自动管理资源, 不论因为什么原因 (哪怕是代码引发了异常) 跳出 with 块总能保证文件被正确关闭, 可以在代码块执行完毕后自动还原进入该代码块时的上下文, 常用于文件操作、数据库连接、网络通信连接、多线程与多进程同步的锁对象管理等场合。
> 
> ```python
> with open(filename, mode, encoding) as fp:
>     ... # 文件操作
> ```
> 
> 它等价于
> 
> ```python
> fp = open(filename, mode, encoding)
> ... # 文件操作
> fp.close()
> ```

如果执行正常, open() 函数返回一个可迭代的文件对象, 通过该文件对象可以对文件进行读写操作.

**文件对象常用方法如下表**

| 命令               | 解释                                                                        |
| ---------------- | ------------------------------------------------------------------------- |
| close()          | 把缓冲区的内容写入文件, 同时关闭文件, 并释放文件对象                                              |
| **read([size])** | **从文本文件中读取 size 个字符的内容作为结果返回, 或从二进制文件中读取指定数量的字节并返回. 如果省略 size 则表示读取所有内容** |
| readline()       | 从文本文件中读取一行内容作为结果返回                                                        |
| **readlines()**  | **把文本文件中的每行文本作为一个字符串存入列表中, 返回该列表, 对于大文件会占用较多内存, 不建议使用**                   |
| tell()           | 返回文件指针的当前位置                                                               |
| write(s)         | 把字符串 s 的内容写入文件                                                            |
| writelines(s)    | 把字符串列表写入文本文件, 不添加换行符                                                      |

*需要特别说明的是, 文件读写操作相关的函数都会自动改变文件指针的位置. 例如, 以读模式打开一个文本文件, 读取10个字符, 会自动把文件指针移动到第 11 个字符的位置, 再次读取字符的时候总是从文件指针的当前位置开始, 写入文件的操作函数也具有相同的特点.*

> 将字符串写入文本文件, 然后再读取并输出
> 
> ```python
> s = 'Hello World\n文本文件的读取方法\n文本文件的写入方法\n'
> with open('sample.txt','w') as fp:
>     fp.write(s)
> with open('sample.txt','r') as fp:
>     print(fp.read())
> ```
> 
> 遍历并输出文本文件的所有行内容
> 
> ```python
> with open('sample.txt') as fp:
>     for line in fp:
>         print(line)
> ```

## Excel/Csv 文件操作 - Pandas

Pandas (Python Data Analysis Library) 是基于 Numpy 的数据分析模块, 提供了大量标准数据类型和高效操作大型数据集所需要的工具, 可以说 Pandas 是使得 Python 能够成为高效且强大的数据分析环境的重要因素之一.

在网络爬虫中, 主要用到的两个功能: 

- **创建 DataFrame 对象**

- **DataFrame 对象和 Excel 之间的交互**

导入库

`import pandas as pd`

1. 创建 DataFrame 对象
   
   ```python
   import pandas as pd
   df = pd.DataFrame()  # 建立空数据框
   name = ['张三', '李四', '王五'] 
   df['姓名'] = name    # 追加列 name
   height = [170, 175, 169]
   weight = [120, 135, 115]
   df['身高'] = height
   df['体重'] = weight
   print(df)           # 查看数据内容
   ```
   
   Out:   姓名   身高   体重
   0  张三  170  120
   1  李四  175  135
   2  王五  169  115

2. 将 df 对象保存为 Excel 文件
   
   ```python
   df.to_excel('Test.xlsx', sheet_name = 'Sheet1', header = True, index = True, startrow = 0, startcol = 0)
   ```
   
   ![](D:\HANSHAN\数据采集\Picture\屏幕截图%202022-03-24%20144836.png)
   
   如果不想让行号保存到 Excel 文件, 则使用下面代码
   
   ```python
   df.to_excel('Test.xlsx', sheet_name = 'Sheet1', header = True, index = False, startrow = 0, startcol = 0)
   ```
   
   ![](D:\HANSHAN\数据采集\Picture\屏幕截图%202022-03-24%20145139.png)

3. 读取 Excel 文件
   
   ```python
   df_new = pd.read_excel('Test.xlsx', sheet_name = 0, header = 0, skiprows = None)
   print(df_new)
   ```
   
   Out:    姓名   身高   体重
   0  张三  170  120
   1  李四  175  135
   2  王五  169  115

> 在大多数的情况下, 使用 Pandas 库进行表格数据的读取、操作和存储可以满足我们的需要. 如果需要对 Excel 文件进行更为细致的操作, 就需要借助于 openpyxl 库.

## 图片文件操作

网络图片的保存

<https://tieba.baidu.com/p/2460150866?red_tag=0829949537>

**使用 open() 函数保存图片**

```python
import requests
# 图片网络地址
img_url = 'https://imgsa.baidu.com/forum/w%3D580/sign=294db374d462853592e0d229a0ee76f2/e732c895d143ad4b630e8f4683025aafa40f0611.jpg'
img = requests.get(img_url)
f = open('test1.jpg','ab')  # 存储图片, 多媒体文件需要参数 b (二进制文件)
f.write(img.content)        # 多媒体存储 content
```

**使用 with 环境保存图片**

```python
import requests
# 图片地址
img_url = 'https://imgsa.baidu.com/forum/w%3D580/sign=294db374d462853592e0d229a0ee76f2/e732c895d143ad4b630e8f4683025aafa40f0611.jpg'
img = requests.get(img_url) # 获取图片内容
# 存储图片, 多媒体文件需要参数 b (二进制文件)
with open('test.jpg','ab') as f:
    f.write(img.content)    # 多媒体存储 content
```

# 正则表达式

获取了 Web 资源 (HTML 代码) 以后，接下来则需要在资源中提取重要的信息。对于 Python 爬虫来说，提取资源 (HTML 代码) 中信息的方式多种多样，在不借助第三方模块的情况下，正则表达式是一个非常强大的工具。

## 正则表达式概述

在网络爬虫中，学习正则表达式主要是为了从字符串中提取符合**给定模式**的 ”子字符串“，这个给定模式就是正则表达式。

> *从字符串中将文字提取出来*
> 
> ```python
> import re
> s = 'eeae123rrrrae456rr'
> re.findall('(\d{2,8})', s)
> ```
> 
> Out: ['123', '456']
> 
> *解释*
> 
> - "()" : 括号里面放置的是要提取的模式
> 
> - "\d" : 可以匹配任何数字
> 
> - "{2,8}" : 表示前面的字符或模式至少重复2次，而最多重复8次

简单来说，**正则表达式**就是描述字符串排列的一套规则，这些规则称为模式。正则表达式通常用来描述自定义的规则，所以正则表达式也称为模式表达式。

*比如，我们想找出一个网页中的所有电子邮件，其它信息需要过滤掉；那么此时，我们可以观察电子邮件；然后，写一个正则表达式来表示所有的电子邮件；随后，我们可以利用该正则表达式从网页中提取出所有满足该规则的字符串出来。*

## 正则表达式模块 re

Python 标准库 re 提供了正则表达式操作所需要的功能，既可以直接使用 re 模块中的方法处理字符串，也可以把模式编译成正则表达式对象再使用。

| 方法                       | 功能说明           |
| ------------------------ | -------------- |
| comoile(pattern)         | 创建模式对象         |
| findall(pattern, string) | 列出字符串中模式的所有匹配项 |

```python
import re
text = 'alpha, beta...gamma delta'  # 测试用的字符串
re.findall('[a-zA-Z]+', text)       # 提取所有的单词
```

Out: ['alpha', 'beta', 'gamma', 'delta']

## 正则表达式基本语法

正则表达式由**元字符**及其不同组合构成，通过巧妙地构造正则表达式可以匹配任意字符串，完成提取、查找和替换等复杂的字符串处理任务 。

> 常用的正则表达式元字符

| 元字符 | 解释                   |
| --- | -------------------- |
| ()  | 将位于 () 内的内容作为一个整体来对待 |

**注: 正则表达式使用圆括号 "()" 表示一个子模式，圆括号内的内容作为一个整体对待**

| 元字符 | 解释                       |
| --- | ------------------------ |
| .   | 匹配除换行符以外的任意单个字符          |
| *   | 匹配位于 * 之前的字符或子模式的0次或多次出现 |
| +   | 匹配位于 * 之前的字符或子模式的1次或多次出现 |

```python
import re
s = 'ab\nc'
print(re.findall('(.)', s))
print(re.findall('(.*)', s))
print(re.findall('(.+)', s))
```

Out: ['a', 'b', 'c']
['ab', '', 'c', '']
['ab', 'c']

| 元字符 | 解释                                       |
| --- | ---------------------------------------- |
| ?   | 匹配位于 ”?“ 之前的0个或1个字符或子模式，即问号之前的字符或子模式是可选的 |

**注: "?" 紧随任何其它限定符 (*, +, ?, {n}, {n,}, {n,m}) 之后时，表示 "非贪心" 匹配模式；"非贪心" 模式匹配尽可能短的字符串，而默认的 "贪心" 模式匹配尽可能长的字符串**。例如，在字符串 "oooo" 中，"o+?" 只匹配单个o, 而 "o+" 匹配所有的o

| 元字符    | 解释                                   |
| ------ | ------------------------------------ |
| []     | 匹配位于 [] 内的任意一个字符                     |
| -      | 在 [] 之内来表示范围                         |
| [a-z]  | 匹配指定范围的任意一个字符                        |
| [^xyz] | ^ 放在 [] 内表示反向字符集，匹配除 x, y, z 之外的任何字符 |
| [^a-z] | 匹配除小写英文字母之外的任何字符                     |

```python
import re
s = 'ab\nc'
print(re.findall('([a-z]*)', s))
print(re.findall('([a-z]+)', s))
```

Out: ['ab', '', 'c', '']
['ab', 'c']

| 元字符    | 解释                             |
| ------ | ------------------------------ |
| **\d** | **匹配任何数字，相当于 [0-9]**           |
| \D     | 与 \d 相反，相当于 [^0-9]             |
| \s     | 匹配任何空白字符，包括空格、制表符、换页符          |
| \S     | 与 \s 相反                        |
| \w     | 匹配任何字符、数字以及下划线，相当于 [a-zA-Z0-9] |
| \W     | 与 \w 相反                        |
| **\|** | **匹配位于 \| 之前或之后的字符**           |

```python
import re
s = 'eeae123rrrrae456rr'
print(re.findall('([a-z]+)', s))  # 提取所有连续的字符
print(re.findall('([0-9]+)', s))  # 提取所有连续的数字
print(re.findall('([0-9]+|[a-z]+)', s))  # 提取所有连续的字符和数字
```

Out: ['eeae', 'rrrrae', 'rr']
['123', '456']
['eeae', '123', 'rrrrae', '456', 'rr']

**元字符 | 可以连续使用**

```python
import re
s = 'eeae123rrrrae456rrABC'
re.findall('([0-9]+|[a-z]+|[A-Z]+)', s)
```

Out: ['eeae', '123', 'rrrrae', '456', 'rr', 'ABC']

| 元字符   | 解释                     |
| ----- | ---------------------- |
| {}    | 按 {} 中指定的次数进行匹配        |
| {m}   | 一个字符重复 m 次             |
| {m,n} | 一个字符重复 m 到 n 次 (含 n 次) |

```python
import re
s = 'eeae123rrrrae456rr'
print(re.findall('(r{4})', s)) 
print(re.findall('(r{2,3})', s))  
print(re.findall('(\d+[a-z]{4})', s)) 
```

Out: ['rrrr']
['rrr', 'rr']
['123rrrr']

## re 模块修饰符

re 模块中包含一些可选标志修饰符来控制匹配的模式

| 元字符      | 解释                                       |
| -------- | ---------------------------------------- |
| re.l     | 使匹配对大小写不敏感                               |
| re.L     | 做本地化识别 (locale-aware) 匹配                 |
| re.M     | 多行匹配，影响 ^ 和 $                            |
| **re.S** | **使匹配包括换行在内的所有字符**                       |
| re.U     | 根据 Unicode 字符集解析字符；这个标志影响 \w, \W, \b, \B |
| re.X     | 该标志通过给予更灵活的格式，以便将正则表达式写的更易理解             |

> **在爬虫中，re.S 是最常用的修饰符，它能够换行匹配**

# Web 前端基础

## HTTP 基本原理

当用户在浏览器中输入 "www.baidu.com" 网址访问百度首页时，用户的浏览器被称为**客户端**，而百度网站被称为**服务器**；这个过程实质上就算客户端向服务器发起请求，服务器接收请求后将处理后的信息 (也称响应) 传给客户端；这个过程是通过 HTTP 协议实现的。

**HTTP (Hyper Text Transfer Protocol)**，即**超文本传输协议**，是互联网上应用最为广泛的一种网络协议。

客户端向服务器端发起请求时，常用的请求方法如下表

| 方法       | 描述                                                                          |
| -------- | --------------------------------------------------------------------------- |
| **get**  | **请求指定的页面信息，并返回响应内容**                                                       |
| **post** | **向指定资源提交数据进行处理请求 (例如提交表单或者上传文件)，数据被包含在请求体中；post  请求可能会导致新的资源的建立、或已有资源的修改** |
| head     | 类似于 get 请求，只不过返回的响应中没有具体的内容，用于获取报文头部信息                                      |
| put      | 从客户端向服务器传送的数据取代指定的文档内容                                                      |
| delete   | 请求服务器删除指定的页面                                                                |
| options  | 允许客户端查看服务器的性能                                                               |

> **以下内容 Important !**
> 
> **服务器返回给客户端的状态码，可以分为 5 种类型，由它们的第一位数字表示**
> 
> *HTTP 状态码及其含义*
> 
> | 代码      | 含义                         |
> | ------- | -------------------------- |
> | **1**** | **信息，请求收到，继续处理**           |
> | **2**** | **成功，行为被成功地接受、理解和采纳**      |
> | **3**** | **重定向，为了完成请求必须进一步执行的动作**   |
> | **4**** | **客户端错误，请求包含语法错误或请求无法实现**  |
> | **5**** | **服务器错误，服务器不能实现一种明显无效的请求** |
> 
> ***例如，状态码为 200，表示请求成功完成；状态码为 404，表示服务器找不到给定的资源***

### 浏览器中的请求和响应

如使用谷歌浏览器访问某一网页，查看请求和响应的具体步骤如下

1. **在谷歌浏览器中输入网址**
   
   <https://mobie.douban.com/top250>
   
   按下 <Enter> 键，进入该网页

2. **单击鼠标右键，选择 “检查” 选项，审查页面元素**

3. **单击谷歌浏览器调试工具的 “Network” 选项，并手动刷新页面，单机调试工具中的 “Name” 栏目下的某一个选项，查看请求与响应的信息，如图**
   
   <img title="" src="file:///D:/HANSHAN/Reptile/Picture/屏幕截图 2022-03-30 173007.png" alt="" width="471">

> 从图中得知 General 概述关键信息如下
> 
> - Request URL: 请求的 URL 地址，也就是服务器的 URL 地址
> 
> - Request Method: 请求方式是 GET
> 
> - Status Code: 状态码是 200，即成功返回响应
> 
> - Remote Address: 服务器 IP 地址是 154.8.131.165，端口号是 443
> 
> 如果我们在浏览器中打开的是一个登录页面，输入 “账号” 与  ”密码“ 后，单击 ”登录“ 按钮将发送一个 POST 请求，此时浏览器的请求信息如图
> 
> <img title="" src="file:///D:/HANSHAN/Reptile/Picture/屏幕截图 2022-03-30 173508.png" alt="" width="434">

## HTML 语言

### 什么是 HTML

**HTML 是纯文本类型的语言，使用 HTML 编写的网页文件也是标准的纯文本文件**；我们可以用任何文本编辑器，例如 Windows 的记事本程序打开它，查看其中的 HTML 源代码，也可以在浏览器打开网页时，通过 ”查看“ -- ”源文件“ 命令查看网页中的 HTML 源代码；HTML 文件可以直接由浏览器解释执行，无须编译；当用浏览器打开网页时，浏览器读取网页中的 HTML 代码，分析其语法结构，然后根据解释的结果显示网页内容。

### 了解 HTML 结构

![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-03-30%20174119.png)

上图所示的代码中，第一行代码用于指定文档的类型；第二行和第十行代码为 HTML 文档的根标签，也就是 <html> 标签；第三行到第六行代码为头标签，也就是 <head> 标签；第七行到第九行代码为主体标签，也就是 <body> 标签。

### HTML 的基本标签

1. **文件开始标签 <html>**
   
   在任何的一个 HTML 文件里，最先出现的 HTML 标签都是 <html>，它用于表示该文件是以超文本标识语言 (HTML) 编写的。<html> 标签是成对出现的，首标签 <html> 和尾标签 </html> 分别位于文件的最前面和最后面，文件中的所有文件和 HTML 标签都包含在其中。
   
   ```html
   <html>
   文件的全部内容
   </html>
   ```
   
   该标签不带任何属性
   
   > 事实上，现在常用的 Web 浏览器 (例如  IE) 都可以自动识别 HTML 文件，并不要求有 html 标签，也不对该标签进行任何操作。但是，为了提高文件的适用性，使编写的 HTML 文件能适应不断变化的 Web 浏览器，还是应该养成使用这个标签的习惯。

2. **文件头部标签 <head>**
   
   习惯上，把 HTML 文件分为 **文件头**和**文件主体**两个部分；文件主体部分都是在 Web 浏览器窗口的用户区内看到的内容，而文件头部分用来规定该文件的标题 (出现在 Web 浏览器窗口的标题栏中) 和文件的一些属性
   
   <head> 是一个表示网页头部的标签；在由 <head> 标签所定义的元素中，并不放置网页的任何内容，而是放置关于 HTML 文件的信息，也就是说它并不属于 HTML 文件的主体；它包含文件的标题编码方式及 URL 等信息；这些信息大部分是用于提供索引、辨认或其他方面的应用
   
   写在 <head> 与 </head> 中间的文本，如果又写在 <title> 标签中，表示该网页的名称，并作为窗口的名称显示在这个网页窗口的最上方
   
   > *说明：如果 HTML 文件并不需要提供相关信息时，可以省略 <head> 标签*

3. **文件标题标签 <title>**
   
   每个 HTML 文件都需要有一个文件名称；在浏览器中，文件名称作为窗口名称显示在该窗口的最上方；这对浏览器的收藏功能很必要的，如果浏览者认为某个网页对自己很有用，今后想经常阅读，可以选择 IE 浏览器 ”收藏“ 菜单中的 ”添加到收藏夹“ 命令将它保存起来，供以后调用；网页的名称要写在 <title> 和 </title> 之间，并且 <title> 标签应包含在 <head> 与 </head> 标签之中
   
   HTML 文件的标签是可以嵌套的，即在一对标签中可以嵌入另一对子标签，用来规定母标签所含范围的属性或其中某一部分内容，嵌套在 <head> 标签中使用的主要有 <title> 标签

4. **元信息标签 <meta>**
   
   meta 标签提供的信息是用户不可见的，它不显示在页面中，一般用来定义页面信息的名称、关键字、作者等；在 HTML 中，<meta> 标签不需要设置结束标签，在一个尖括号内就是一个 meta 内容，而在一个 HTML 头页面中可以有多个 <meta> 标签；<meta> 标签的属性有两种：name 和 http-equiv，其中 name 属性主要用于描述网页，以便于搜索引擎机器人查找、分类。
   
   特别要注意的是，HTML 文档的编码也包含在该标签中，如
   
   ```html
   <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
   ```
   
   表明该 HTML 文档的编码格式为 “utf-8”

5. **页面的主体标签 <body>**
   
   网页的主体部分以 <body> 标签标志它的开始，以 </body> 标志它的结束；在网页的主体标签中有很多的属性设置，如表

6. **内容标签**
   
   不同级别的标题
   
   ```html
   <h1>一级标题</h1>
   <h2>二级标题</h2>
   <h3>三级标题</h3>
   ```
   
   段落
   
   `<p>这是一个段落</p>`
   
   超链接，链接地址由 href 属性指定
   
   `<a href="http:www.baidu.com">点这里</a>`
   
   图片，图片地址由 src 属性指定
   
   `<image src="http://www.***" width="200" height="300" />`
   
   table 标签用来创建表格
   
   tr 标签用来创建行
   
   td 标签用来创建单元格
   
   ```html
   <table border="1">
   <tr>
       <td>第1行第1列</td>
       <td>第1行第2列</td>
   </tr>
   <tr>
       <td>第2行第1列</td>
       <td>第2行第2列</td>
   </tr>
   </table>
   ```
   
   ul 标签用来创建无序列表
   
   uo 标签用来创建有序列表
   
   li 标签用来创建其中的列表项
   
   ```html
   <ul id="colors" name="myColor">
       <li>红色</li>
       <li>绿色</li>
       <li>蓝色</li>
   </ul>
   ```
   
   div 标签可以用来创建一个块，其中可以包含其它标签
   
   ```html
   <div id="reddiv" style="background-color:red">
       <p>第1段</p>
       <p>第2段</p>
   </div>
   ```

## CSS 层叠样式表

CSS 是 Cascading Style Sheets (层叠样式表) 的缩写，它是一种标记语言，用于为 HTML 文档定布局；例如，CSS 涉及字体、颜色、边距、高度、宽度、背景图像、高级定位等方面；运用 CSS 样式可以让页面变得美观，就像化妆前和化妆后的效果一样

关于 CSS，在实际的网络爬虫用到的知识并不多，这里只用知道是做什么的就可以了；关于 CSS 的具体内容，可参考相关书籍进行更为全面的学习

## JavaScript 动态脚本语言

通常，我们所说的 **Web 前端就是指 HTML、CSS 和 JavaScript 三项技术**：

- **HTML: 定义网页的内容**

- CSS: 描述网页的样式

- JavaScript: 描述网页的行为

JavaScript 是一种可以嵌入在 HTML 代码中由客户端浏览器运行的脚本语言；在网页中使用 JavaScript 代码，不仅可以实现网页特效，还可以响应用户请求实现动态交互的功能；例如，在用户注册页面中，需要对用户输入信息的合法性进行验证，包括是否填写了 “邮箱” 和 “手机号”，填写 的信息格式是否正确等；

通常情况下，在 Web 页面中使用 JavaScript 有以下两种方法，一种是在页面中直接嵌入 JavaScript 代码，另一种是链接外部 JavaScript 文件。

1. **在页面中直接嵌入 JavaScript 代码**
   
   **在 HTML 文档中可以使用 <script>...</script> 标签将 JavaScript 脚本嵌入到其中；** 在 HTML 文档中可以使用多个 <script> 标签，每个 <script> 标签中可以包含多个 JavaScript  的代码集合；
   
   <script> 标签常用的属性及说明如表
   
   | 属性值      | 说明                            |
   | -------- | ----------------------------- |
   | language | 设置所使用的脚本语言及版本                 |
   | src      | 设置一个外部脚本文件的路径位置               |
   | type     | 设置所使用的脚本语言，此属性已代替 language 属性 |
   | defer    | 此属性表示当 HTML 文档加载完毕后再执行脚本语言    |
   
   如图
   
   ![](D:\Study\数据采集\Picture\屏幕截图%202022-03-31%20145454.png)
   
   > **注意：<script> 标签可以放在 Web 页面的 <head> </head> 标签中，也可以放在 <body> </body> 标签中**

2. **链接外部 Javascript 文件**
   
   在 Web 页面中引入 JavaScript 代码的另一种方法是采用链接外部 JavaScript 文件的形式；如果 脚本代码比较复杂或是同一段代码可以被多个页面所使用，则可以将这些脚本代码放置在一个单独的文件中 (保存文件的扩展名为 js)，然后在需要使用改代码的 Web 页面中链接该 JavaScript 文件即可；在 Web 页面中链接外部 JavaScript 文件的语法格式如下
   
   `<script language="javascript" src="your-JavaScript.js"></script>`

## 网页观察

在网页爬取信息时，通常会直接爬取多个网页，而这多个网页的网址信息会呈现出某种规律，这种规律可以通过直接观察得到；例如其中一页的网址

<https://movie.douban.com/top250?start=25&filter=>

参数 "?start=25"，不同的值指示不同的网页；去掉参数 "filter="，即网址为

<https://movie.douban.com/top250?start=25>

可以发现网页是不变的；可以看到这个参数 "?start=25" 是至关重要的

# Urllib 请求模块

在实线网络爬虫的爬取工作时，就必须使用到网络请求，只有进行了网络请求才可以对响应结果中的数据进行提取；utllib 模块是 Python 自带的网络请求模块，无需安装，导入即可使用

【提取网页源代码】

## urllib 简介

- **urllib.request: 用于实现基本 HTTP 请求的模块**

- urllib.error: 异常处理模块，如果在发送网络请求时出现了错误，可以捕获异常进行异常的有效处理

- urllib.parse: 用于解析 URL 的模块

- urllib.robotparser: 用于解析 robots.txt 文件，判断网站是否可以爬取信息

## 使用 urllib.request.urlopen() 方法发送请求

urllib.request 模块提供了 urlopen() 方法发送请求

`urllib.request.urlopen(url, data = None, [timeout,]*, cafile = None, capath = None, cadefault = False, context = None)`

- url: 需要访问网站的 url 完整地址

- data: 默认 = None，通过该参数确认请求方式
  
  - **= None, 表示请求方式为 get, 否则请求方式为 post**
  
  - 在发送 post 请求时，参数 data 需要以字典形式的数据作为参数值，并且需要将字典类型的参数值转换为字节类型的数据才可以实现 post 请求

- **timeout: 以秒为单位，设置超时停止访问**

- cafile, capath: 指定一组 HTTPS 请求受信任的 CA 证书，cafile 指定包含 CA 证书的单个文件，capath 指定证书文件目录

- cadefault: CA 证书默认值

- context: 描述 SSL 选项的示例

### 发送 get 请求

使用 urlopen() 方法实现一个网络请求时，所返回的是一个 "http.client.HTTPResponse" 对象

```python
#全局取消证书验证
import ssl
ssl.__create__default__https__context = ssl.__create__unverified__context

# 导入 request 子模块
import urllib.request
# 发送网络请求
response = urllib.request.urlopen('https://www.baidu.com/')

print('响应数据类型为：', type(response))
```

Out: 响应数据类型为：<class 'http.client.HTTPResponse'>

```python
# 在 HTTPResponse 对象中包含可以获取信息的方法以及属性
#全局取消证书验证
import ssl
ssl.__create__default__https__context = ssl.__create__unverified__context

import urllib.request
url = 'https://www.baidu.com/'
# 发送网络请求
response = urllib.request.urlopen(url = url)
print('相应状态码为：', response.states)
print('响应头所有信息为：', response.getheaders())
print('响应头指定信息为：', response.getheaders('Accept-Ranges'))
# 读取 HTML 代码并进行 UTF-8 解码
print('官网 HTML 代码如下', response.read().decode('utf-8'))
```

Out:

![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-09%20234103.png)

![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-09%20234138.png)

### 发送 post 请求

urlopen() 方法默认发送的是 get 请求，如果需要发送 post 请求，可以为其设置 data 参数，该参数是 bytes 类型，所有需要使用 bytes() 方法将参数值进行数据类型转换

```python
#全局取消证书验证
import ssl
ssl.__create__default__https__context = ssl.__create__unverified__context

# 导入 request 子模块
import urllib.request
# 导入 parse 子模块
import urllib.parse
# post 请求测试地址
url = 'https://www,httpbin.org/post'
# 将表单数据转换为 bytes 类型，并设置编码方式为 utf-8
data = bytes(urllib.parse.urlencode({'hello': 'python'}), encoding = 'utf-8')
# 发送网络请求
response = urllib.request.urlopen(url = url, data = data)

print('相应状态码为：', response.states)
print('响应头所有信息为：', response.getheaders())
print('响应头指定信息为：', response.getheaders('Accept-Ranges'))
# 读取 HTML 代码并进行 UTF-8 解码
print('HTML 代码如下', response.read().decode('utf-8'))
```

Out:

![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-10%20092400.png)

![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-10%20092416.png)

### 设置网络超时

timeout 参数用于设置请求超时，该参数以秒为单位，表示如果在请求时，超出了设置的时间还没有得到响应时就抛出异常

```python
#全局取消证书验证
import ssl
ssl.__create__default__https__context = ssl.__create__unverified__context

import urllib.request
url = 'https://www.baidu.com/'
response = urllib.request.urlopen(url = url, timeout = 0.01)
# 读取 HTML 代码并进行 utf-8 解码
print(response.read().decode('utf-8'))

# 由于代码中的超时时间设置为0.01秒，时间较短，所以将显示超时异常
```

> 根据网络环境的不同，可以将超时时间设置为一个合理的时间，如2秒、3秒
> 
> 如果遇到了超时异常，爬虫程序将在此处停止；所以在实际开发中开发者可以将超时异常捕获，然后再处理下面的爬虫任务

## 复杂的网络请求

urlopen() 方法能够发送一个最基本的网络请求，但这并不是一个完整的网络请求；如果要构建一个完整的网络请求，还需要在ing求中添加 Headers, Cookies 以及代理 IP 等内容

Request 类则可以构建一个多种功能的请求对象

`urllib.request.Request(url, data = None, headers = {}, orihin_req_host = None, unverifiable = False, method = None)`

- url: 网站完整地址

- data: 默认 = None，通过该参数确认请求方式
  
  - **= None, 表示请求方式为 get, 否则请求方式为 post**
  
  - 在发送 post 请求时，参数 data 需要以字典形式的数据作为参数值，并且需要将字典类型的参数值转换为字节类型的数据才可以实现 post 请求

- headers: 设置请求头部信息，该参数为字典类型；**添加请求头信息最常见的用法就是修改 User-Agent 来伪装成浏览器；例如：header = {'User-Agent': 'Mozilla/5.0(Macintosh; Inter MacOs X 10_13_6)AppleWebKit/537.36(KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'}, 表示伪装 Google 浏览器进行网络请求**

- origin_req_host: 用于设置请求的 host 名称或者是 IP

- unverifiable: 用于设置网页是否需要验证，默认 False

- method: 设置请求方法，如 get, post；默认 get

**设置请求头参数是为了模拟浏览器向网页后台发送网络请求，这样可以避免服务器的反爬措施**

使用 urlopen() 方法发送网络请求时，其本身并没有设置请求头参数，所以向 'https://www.httpbin.org/post' 请求测试地址发送请求时，返回的信息中 headers 将显示如下所示默认值

```python
#全局取消证书验证
import ssl
ssl.__create__default__https__context = ssl.__create__unverified__context

# 导入 request 子模块
import urllib.request
# 导入 parse 子模块
import urllib.parse
# post 请求测试地址
url = 'https://www,httpbin.org/post'
# 将表单数据转换为 bytes 类型，并设置编码方式为 utf-8
data = bytes(urllib.parse.urlencode({'hello': 'python'}), encoding = 'utf-8')
# 发送网络请求
response = urllib.request.urlopen(url = url, data = data)
# 读取 HTML 代码并进行 UTF-8 解码
print(response.read().decode('utf-8'))
```

Out:

![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-10%20095715.png)

> 1. method one
>    
>    所以在设置请求头信息前，需要在浏览器中找到一个有效的请求头信息，以 Google 浏览器为例；执行以下操作
>    
>    - 打开 Google 浏览器，右键选择【检查】选项
>    
>    - 然后选择 Network 选项，接着在浏览器地址栏中任意打开一个网页，在请求列表中选择一项请求信息
>    
>    - 在 Headers 选项中找到请求头信息
> 
> 2. method two
>    
>    在地址栏输入以下代码
>    
>    `chrome://version/`
>    
>    ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-10%20100339.png)

```python
'''
如果需要设置请求头信息，首先通过 Request 类构造一个带有 headers 请求头信息的 Request 对象，
然后为 urlopen() 方法传入 Request 对象，再进行网络请求的发送
'''
#全局取消证书验证
import ssl
ssl.__create__default__https__context = ssl.__create__unverified__context

# 导入 request 子模块
import urllib.request
# 导入 parse 子模块
import urllib.parse
# post 请求测试地址
url = 'https://www,httpbin.org/post'

# 浏览器模拟头部
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36'}

# 将表单数据转换为 bytes 类型，并设置编码方式为 utf-8
data = bytes(urllib.parse.urlencode({'hello': 'python'}), encoding = 'utf-8')
# 创建 Request 对象
r = urllib.request.Request(url = url, data = data, headers = headers, method = 'POST')
# 发送网络请求
response = urllib.request.urlopen(r)
# 读取 HTML 代码并进行 UTF-8 解码
print(response.read().decode('utf-8'))
```

Out:

![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-10%20101032.png)

![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-10%20101118.png)

## 常用 urllib 获取网页源码的模板

在 Python 网络爬虫获取数据时，我们最常用的是如下方式的 get() 获取方式

```python
# 爬取的基础网址
base_url = 'https://movie.douban.com/top250?start='
# 代理服务，模拟网页查找
# Google 浏览器输入 chrome://version/, 找到【用户代理】
header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36'}
# 发送请求
request = urllib.request.Request(url = base_url, headers = header)
response = urllib.request.urlopen(request)
# 获取网页内容
html = response.read().decode('utf-8')
print(html)
```

## 案例: 使用 urllib 下载图片

```python
#全局取消证书验证
import ssl
ssl.__create__default__https__context = ssl.__create__unverified__context

# 导入 re 模块，直接调用来实现正则匹配
import re
import urllib.request
# 下载图片的地址
url = 'http://tieba.baidu.com/p/2460150866'
# 使用 urlopen() 打开，read() 读取并设置解码格式
response = urllib.request.urlopen(url).read().decode('utf-8')
imglist = re.findall('src="(.+?\.jpg)"pic_ext', response)
# 建立文件夹用来保存图片
import os
os.mkdir('image_download')
# 定义变量 x 并初始化用来计数图片的张数
x = 1
# 遍历
for imgurl in imglist:
    # urlretrieve() 方法直接将远程数据下载到本地
    urllib.request.urlretrieve(imgurl, './image_download/image{}.jpg'.format(x))
    print('第', x, '张')
    x += 1
print('下载完毕')
```

# Requests 请求模块

requests 是 Python 中实现 HTTP 请求的一种方式，requests 是第三方模块，该模块在实现 HTTP 请求时要比 urllib, urllib3 模块简化很多，操作更加人性化

## 请求方式

### get 请求

最常用的 HTTP 请求方式分别为 get 和 post, 在使用 requests 模块实现 get 请求时可以使用两种方式实现，一种是带参数，另一种不带参数

1. 不带参数【不需要账号密码】
   
   ```python
   # 导入 requests 模块
   import requests
   # 发送网络请求
   url = 'https://www.baidu.com'
   response = requests.get(url)
   print('相应状态码为：', response.status_code)
   print('请求的网络地址为：', response.url)
   print('头部信息为：', response.headers)
   print('cookie 信息为：', response.cookies)
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-11%20092359.png)

2. 带参数
   
   - 为 get 请求指定参数时，可以直接将参数添加在请求地址 url 的后面，然后用 ? 进行分割
     
     一个 url 地址中有多个参数，参数之间用 & 进行连接
     
     ```python
     import requests
     url = 'http://httpbin.org/get?name=Jack&ange=30'
     response = requests.get(url)
     print(response.text)
     ```
     
     Out:
     
     ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-11%20094555.png)
     
     > 对网站 'http://httpbin.org/get' 进行操作，该网站可以作为练习网络请求的一个站点使用，该网站可以模拟各种请求操作
   
   - 配置 params 参数
     
     requests 模块提供了传递参数的方法，允许使用 params 关键字参数以一个字符串字典来提供这些参数
     
     ```python
     # 想传递 key1 = value1 和 key2 = value2 到 httpbin.org/get
     import requests
     # 定义请求参数
     data = {'name': 'Michael', 'age': '36'}
     url = 'http://httpbin.org/get'
     response = requests.get(url, params = data)
     print(response.text)
     ```
     
     Out:
     
     ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-11%20095400.png)

### 对响应结果指定编码格式

当相应状态码为200时，说明本次网络请求已经成功，此时可以获取请求地址所对应的网页源码

为了正确获取网页 html 源码的内容，需要正确指定 html 源码中指定的编码格式；**通过属性 encoding 来指定**

```python
import requests
url = 'https://www.baidu.com'
response = requests.get(url)
# 对响应结果进行 utf-8 编码
response.encoding = 'utf-8'
# 以文本形式打印网页源码
print(response.text)
```

Out:

![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-11%20093243.png)

> 注意
> 
> 在没有对响应内容进行 utf-8 编码时，网页源码中的中文信息可以会出现乱码

### 爬取二进制数据

get() 函数不仅可以获取网页中的源码信息，还可以获取二进制文件

但是在获取二进制文件时，需要使用 Response.content 属性获取 bytes 类型的数据，然后将数据保存在本地文件中

```python
# 下载百度首页中的 LOGO 图片
import requests
url = 'https://dss0.bdstatic.com/5aV1bjqh_Q23odCf/static/superman/img/logo/logo_redBlue-0a7c20fcaa.png'
response = requests.get(url)
# 打印二进制数据
print(response.content)
# 写入二进制文件
with open('百度_LOGO.png', 'wb') as f:
    f.write(response.content)
```

### post 请求

post 请求方式也叫做提交表单，表单中的数据内容就是对应的请求参数；使用 requests 模块实现 post 请求时需要设置请求参数 data

```python
import request
# 字典类型的表单参数
data = {'1': 'one',
        '2': 'two'}
url = 'http://httpbin.org/post'
response = requests.post(url, params = data)
print(response.text)
```

Out:

![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-11%20095911.png)

## 复杂的网络请求

在使用 requests 模块实现网络请求时，不只有简单的 get 与 post；还有复杂的请求头、Cookies 以及网络超时等

不过，requests 模块将这一系列复杂的请求方式进行了简化，只要在发送请求时设置对应的参数即可实现复杂的网络请求

有时在请求一个网页内容时，会发现无论通过 get, post 以及其它请求方式，都会出现403错误；这种现象多数为服务器拒绝了您的请求，那是因为这些网页为了放置恶意采集数据，所使用的反爬设置

此时可以通过模拟浏览器的头部信息来进行访问

```python
# 添加请求头模拟浏览器头部信息访问网页信息
header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36'}
import requests
url = 'https://www.baidu.com'
response = requests.get(url, headers = headers)
print(response.status_code)  # 打印状态码
```

### 验证 Cookies

在爬取某些数据时，需要进行网页的登录，才可以进行数据的抓取工作；Cookies 登录就像很多网页中的自动登录功能一样，可以让用户第二次登录时在不需要验证账号和密码的情况下进行登录

在使用 requests 模块实现 Cookies 登录时

- 首先需要在浏览器的开发者工具页面中找到可以实现登录的 Cookies 信息

- 然后将 Cookies 信息处理并添加至 RequestsCookieJar 对象中

- 最后将 RequestCookieJar 对象作为网络请求的 Cookies 参数发送网络请求即可

> 例 
> 通过验证 Cookies 模拟豆瓣登录
> 
> Step_1
>  在 Google 浏览器中打开豆瓣网页地址 <https://www.douban.com/>, 并输入自己的账号密码登录
> 
> Step_2
> 右键【检查】，选择【Network】选项
> 
> Step_3
> 在 'name' 框中选择其中一个，在【Headers】选项中选择 Request Headers 选项，获取登录后的 Cookies 信息（选中后右键【Copy value】）
> 
> Step_4
> 导入相应的模块，将复制的 Cookie 信息粘贴到下面的 ”此处填写登录后网页的 Cookie 信息“ 中
> 然后创建 RequestsCookieJar() 对象并对 Cookie 信息进行处理
> 最后将处理后的 RequestsCookieJar() 对象作为网络请求参数，实现网页的登录请求
> 
> ```python
> header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36'}
> import requests
> url = 'https://www.douban.com'
> # 创建 RequestsCookieJar 对象, 用于设置 Cookies 信息
> cookies_jar = requests.cookies.RequestsCookieJar()
> for cookie in cookies.split(';'):
>     key, value = cookie.split('=', 1)
>     cookies_jar.set(key, value)
> response = requests.get(url, headers = headers, cookies = cookies_jar)
> print(response.status_code)
> print(response.text)
> ```
> 
> Out:
> 
> ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-11%20102511.png)
> 
> ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-11%20102527.png)

### 网络超时

在访问一个网页时，如果该网页长时间未响应，系统就会判断该网页超时，无法打开网页

```python
# 模拟一个网络超时的现象
import requests
url = 'https://www.baidu.com'
try:
    response = requests.get(url, timeout = 0.01)
    print(response.status_code)
except Exception as E:
    print('异常为：\n', str(E))
```

Out:
异常为：
 HTTPSConnectionPool(host='www.baidu.com', port=443): Max retries exceeded with url: / (Caused by ConnectTimeoutError(<urllib3.connection.VerifiedHTTPSConnection object at 0x110b03b90>, 'Connection to www.baidu.com timed out. (connect timeout=0.01)'))

### 网络异常

针对网络异常信息，requests 模块提供了三种常见的网络异常类捕获异常

```python
import requests
from requests.exceptions import ReadTimeout, HTTPError, RequestException
url = 'https://www.baidu.com'
try:
    response = requests.get(url, headers = headers, timeout = 0.001)
    print(response.status_code)
except ReadTimeout:
    print('timeout')
    # 超时异常
except HTTPError:
    print('httperror')
    # HTTP 异常
except RequestException:
    print('reqerror')
    # 请求异常
```

## 案例: 使用 request 下载图片

```python
#全局取消证书验证
import ssl
ssl.__create__default__https__context = ssl.__create__unverified__context

# 导入 re 模块，直接调用来实现正则匹配
import re
import requests
# 下载图片的地址
url = 'http://tieba.baidu.com/p/2460150866'

# 使用 get() 获取 url
r = requests.get(url)
imglist = re.findall('src="(.+?\.jpg)"pic_ext', r.text)
import os
# 建立文件夹来保存图片
os.mkdir('image_download')
# 定义变量 x 并初始化用来计数图片的张数
x = 0
for imgurl in imglist:
    # 获取获得的从 imglist 中遍历得到的 imgurl
    imgres = requests.get(imgurl)
    with open('./image_download/image{}.jpg'.format(x), 'wb') as f:
        f.write(imgres.content)
        x += 1
        print('第', x, '张')
print('下载完毕')
```

# Xpath 解析模块

虽然正则表达式处理字符串的能力很强，但是在编写正则表达式的时候代码还是比较麻烦的，如果不小心写错一处，那么将无法匹配页面中所需要的数据，因为网页中包含大量的节点，而节点中又包含 id, class 等属性

如果在解析页面中的数据时，通过 Xpath 来**定位**网页中的数据，将会更加简单有效

## Xpath 模块

XML 路径语言 (XML Path Language, Xpath) 是一门在 XML 文档中查找信息的语言；Xpath 最初被设计用来搜寻 XML 文档，但同样使用于 HTML 文档的搜索；所以在爬虫中可以使用 Xpath 在 HTML 文档中进行可用信息的抓取

Xpath 功能非常强大，不仅提供非常简洁明了的路径选择表达式，还提供了超过100个内建函数，用于字符串、数值、时间比较、序列处理、逻辑值等，几乎所有定位的节点都可以用 Xpath 来选择

Xapth 于1999年11月16日称为 W3C 标准，被设计为 XSLT, XPointer 以及其他 XML 解析软件使用，Xpath 使用路径表达式在 XML 或 HTML 中选取节点，最常用的路径表达式如表

| 表达式      | 描述          |
| -------- | ----------- |
| nodename | 选取此节点的所有子节点 |
| /        | 从当前节点选取子节点  |
| //       | 从当前节点选取子孙节点 |
| .        | 选取当前节点      |
| ..       | 选取当前节点的父节点  |
| @        | 选取属性        |
| *        | 选取所有节点      |

[Xpath more](https://www.w3.org/TR/xpath/all)

## Xpath 的解析操作

Python 中可以支持 Xpath 提取数据的解析模块有很多，这里主要介绍 lxml 模块，该模块可以解析 HTML 与 XML，并且支持 Xpath 解析

因为 lxml 模块的底层是通过 C 语言所编写，所以在解析效率方面是非常优秀的

解析 HTML

**HTML() 方法**

lxml.etree 子模块提供了一个 HTML() 方法，该方法可以实现解析字符串类型的 HTML 代码

`lxml.etree.HTML(text, parser = None, *, base_url = None)`

- text: 接收 str; 表示需要转换为 HTML 的字符串

- parser: 接收 str; 表示选择的 HTML 解析器

- base_url: 接收 sr; 表示文档的原始 url, 用于查找外部实体的相对路径

```python
# 使用 HTML() 方法将网页内容初始化，并打印初始化后的网页内容
import requests
url = 'http://www.baidu.com'
ua = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36'}
rqg = requests.get(url.headers = ua)
# 导入 etree 模块
from lxml import etree
# 初始化 HTML
html_str = rqg.content.text
html_xpath = etree.HTML(html_str)
# 输出修正后的 HTML
result = etree.tostring(html_xpath, encoding = 'utf-8')
print('修正后的HTML:', result.decode('utf-8'))

'''
首先调用 HTML() 对 requests 库请求回来的网页 "html_str" 进行初始化，这样就成功构造了一个
Xpath 解析对象 "html_xpath"; 若 HTML 中的节点没有闭合，etree 模块也可提供自动补全功能；
调用 tostring 方法即可输出修正后的 HTML 代码，但是结果 "result" 为 bytes 类型，需要使用
decode() 方法将其转成 str 类型
'''
```

Out:

![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-12%20135810.png)

## 信息的提取

1. 获取所有节点
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-12%20140406.png)
   
   ```python
   # 使用 '//*' 的方式获取 HTML 代码中所有节点信息
   # 导入 etree 子模块
   from lxml import etree
   # 解析 HTML 字符串
   html = etree.HTML(html_str)
   # 获取所有节点
   node_all = html.xpath('//*')
   print('数据类型：', type(node_all))
   print('数据长度：', len(node_all))
   print('数据内容：', node_all))
   # 通过列表推导式打印所有节点名称，通过节点对象 .tag 获取节点名称
   print('节点名称：', [i.tag for i in node_all])
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-12%20141156.png)
   
   ```python
   # 需要获取 HTML 代码中所有指定名称的节点，在 '//' 后面添加节点的名称
   # 导入 etree 子模块
   from lxml import etree
   # 解析 HTML 字符串
   html = etree.HTML(html_str)
   # 获取所有节点
   li_all = html.xpath('//li')
   print('所有li节点：', li_all)
   print('指定li节点：', li_all[0])
   print('指定li节点的html代码：', etree.tostring(li_all[0], encoding = 'utf-8').decode('utf-8'))
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-12%20141612.png)

2. 获取子节点
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-12%20140406.png)
   
   ```python
   # 需要获取一个节点中的直接子节点使用 '/'
   from lxml import etree
   # 解析 HTML 字符串
   html = etree.HTML(html_str)
   # 获取所有 a 节点
   a_all = html.xpath('//li/a')
   print('所有 a 节点: ', a_all)
   a_html = [etree.tostring(i, encoding = 'utf-8').decode('utf-8') for i in a_all]
   for i in a_html:
       print(i)
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-12%20142058.png)
   
   可以看到属性 title 中的带有双引号的字符串并没有解码成功；为了保证这里解码成功，需要在 tostring() 方法中指定参数 "method = 'html'"
   
   ```python
   from lxml import etree
   # 解析 HTML 字符串
   html = etree.HTML(html_str)
   # 获取所有 a 节点
   a_all = html.xpath('//li/a')
   print('所有 a 节点: ', a_all)
   a_html = [etree.tostring(i, encoding = 'utf-8', method = 'html').decode('utf-8') for i in a_all]
   for i in a_html:
       print(i)
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-12%20142410.png)
   
   '/' 获取直接的子节点
   
   '//' 获取子孙节点
   
   ```python
   # 获取 ul 节点中所有子孙节点 
   a_all = html.xpath('//ul//a')
   print('所有a节点：', a_all)
   a_html = [etree.tostring(i, encoding = 'utf-8', method = 'html').decode('utf-8') for i in a_all]
   for i in a_html:
       print(i)
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-12%20144746.png)

3. 获取父节点
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-12%20140406.png)
   
   ```python
   # 获取父节点，使用 '..' 来实现
   from lxml import etree
   html = etree.HTML(html_str)
   a_all_parent = html.xpath('//a/..')
   print('所有a节点的父节点：', a_all_parent)
   a_html = [etree.tostring(i, encoding = 'utf-8', method = 'html').decode('utf-8') for i in a_all_parent]
   for i in a_html:
       print(i)
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-12%20145523.png)
   
   > 在实际使用中，通常情况下都是通过层层的子孙标签来进行信息的获取，通过父节点提取信息的情况较少

4. 获取文本
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-12%20140406.png)
   
   ```python
   # 获取文本，使用 text() 方法
   from lxml import etree
   html = etree.HTML(html_str)
   # 获取所有 a 节点的文本
   a_all_text = html.xpath('//a/text()')
   print(a_all_text)
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-12%20150113.png)

## 属性匹配

1. 单个属性匹配
   
   ```python
   # 需要更精确地获取某个节点中的内容，使用 '[@...]' 事项节点属性匹配
   # '...' 表示属性匹配的条件
   # 获取所有 class = "level" 中的 div 节点
   html_str = '''
   <div class="video_scroll">
       <div class="level">什么是Java</div>
       <div class="level">Java的版本</div>
   </div>
   '''
   from lxml import etree
   html = etree.HTML(html_str)
   # 获取所有 div 节点的文本
   div_all_text = html.xpath('//div[@class="lever"]/text()')
   print(div_all_text)
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-12%20151605.png)
   
   > '[@...]' 不仅可以用于 class 属性的匹配，还可以用于 id, href 等属性的匹配

2. 多个属性匹配
   
   ```python
   # 一个节点中出现多个属性，需要同时匹配多个属性，才能更精确获取指定节点中的数据
   html_str = '''
   <div class="video_scroll">
       <div class="level" id="one">什么是Java</div>
       <div class="level">Java的版本</div>
   </div>
   '''
   from lxml import etree
   html = etree.HTML(html_str)
   # 获取所有 div 节点的文本
   div_all_text = html.xpath('//div[@class="level" and @id="one"]/text()')
   print(div_all_text)
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-12%20152252.png)
   
   ```python
   # or
   div_all_text = html.xpath('//div[(@class="level" and @id="one") or @class="level"]/text()')
   print(div_all_text)
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-12%20151605.png)

3. 获取标签属性值
   
   ```python
   # '@' 不仅可以实现通过属性匹配节点，还可以直接获取属性所对应的值
   from lxml import etree
   html_str = '''
   <div class="video_scroll">
       <li class="level" id="one">什么是Java</li>
   </div>
   '''
   html = etree.HTML(html_str)
   # 获取 li 节点中的 class 属性值
   li_class = html.xpath('//div/li/@class')
   print(li_class)
   # 获取 li 节点钟的 id 属性值
   li_id = html.xpath('//div/li/@id')
   print(li_id)
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-12%20153304.png)

## 按序进行信息获取

如果同时匹配了多个节点，但**只需要其中的某一个节点时**，可以使用指定编号索引的方式获取对应的节点内容，不过 **Xpath 中的索引是从1开始的**，所以需要注意不要与 Python 中的列表索引混淆

给定以下代码

![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-12%20154548.png)

```python
from lxml import etree
html = etree.HTML(html_str)
# 获取所有 a 节点中的 title 属性值
a_all_title = html.xpath('//div/ul/li/a/@title')
print(a_all_title)
# 获取第1个 li/a 节点中的 title 属性值
a_title_1 = html.xpath('//div/ul/li[1]/a/@title')
print(a_title_1)
# 获取第4个 li/a 节点中的 title 属性值
a_title_4 = html.xpath('//div/ul/li[4]/a/@title')
print(a_title_4)
```

Out:

![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-12%20155158.png)

```python
# 还可使用 Xpath 中提供的函数来获取指定节点的内容
html = etree.HTML(html_str)
a_all_title = html.Xpath('//div/ul/li/a/@title')
print(a_all_title)
# 获取第1个 li/a 节点中的 title 属性值
a_title_1 = html.xpath('//div/ul/li[position()=1]/a/@title')
print(a_title_1)
```

Out:

![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-12%20155528.png)

```python
a_title_4 = html.xpath('//div/ul/li[position()=4]/a/@title')
print(a_title_4)
# 获取位置大于第1个 li/a 节点中 title 属性值
a_title_4 = html.xpath('//div/ul/li[position()>1]/a/@title')
print(a_title_4)
# 获取最后一个
a_title_4 = html.xpath('//div/ul/li[liat()]/a/@title')
print(a_title_4)
# 获取倒数第1个
a_title_4 = html.xpath('//div/ul/li[last()-1]/a/@title')
print(a_title_4)
```

Out:

![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-12%20160030.png)

## 案例: 爬取动物庄园小说

```python
# 首先抓取第一章内容
# 全局取消证书验证
import ssl
ssl.__create__default__https__context = ssl.__create__unverified__context
# 模拟浏览器头部
header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36'}
import requests
from lxml import etree
import time

i = 0
# 主页网址 'http://www.kanunu8.com/book3/6879'
https://www.kanunu8.com/book3/6879/131780.html
url = 'https://www.kanunu8.com/book3/6879/{}.html'.format(131779+i)
# 获取网页源代码
res_source = requests.get(url, headers = header)
'''
观察 <meta http-equiv="Content-Type" content="text/html; charset=gbk">
可以发现该网页编码格式为 'gbk'
'''
# 解码网页内容
res = res_source.content.decode('gbk')
# 提取小说内容
# 通过观察网页源代码可以发现，小说内容均在 p 标签中
# 可以通过如下方式提取小说内容
html = etree.HTML(res)  # 解析 HTML 字符串
content = html.xpath('//p/text()')
content = ''.join(content)
# 将提取的内容保存到 txt 文件中
with open('chapter{}.txt'.format(i+1), 'a') as fp:
    fp.write(content)
print('第', i+1, '章下载完成')

# 抓取所有章节
for i in range(10):
    # 主页网址 'http://www.kanunu8.com/book3/6879'
    https://www.kanunu8.com/book3/6879/131780.html
    url = 'https://www.kanunu8.com/book3/6879/{}.html'.format(131779+i)
    # 获取网页源代码
    res_source = requests.get(url, headers = header)
    '''
    观察 <meta http-equiv="Content-Type" content="text/html; charset=gbk">
    可以发现该网页编码格式为 'gbk'
    '''
    # 解码网页内容
    res = res_source.content.decode('gbk')
    # 提取小说内容
    # 通过观察网页源代码可以发现，小说内容均在 p 标签中
    # 可以通过如下方式提取小说内容
    html = etree.HTML(res)  # 解析 HTML 字符串
    content = html.xpath('//p/text()')
    content = ''.join(content)
    # 将提取的内容保存到 txt 文件中
    with open('chapter{}.txt'.format(i+1), 'a') as fp:
        fp.write(content)
    print('第', i+1, '章下载完成')
    time.sleep(1)
```

# Bs4 解析模块

Beautiful Soup 是一个可以从 HTML 或 XML 文件中提取数据的 Python 库；它提供了一些简单的函数用来处理导航、搜索、修改分析树等功能

通过解析文档，Beautiful Soup 库可为用户提供需要抓取的数据，非常简便，仅需少量代码就可以写出一个完整的应用程序

Beautiful Soup 自动将输入文档转换为 Unicode 编码，输出文档转换为 utf-8 编码；开发者不需要考虑编码方式，除非文档没有指定一个编码方式，这时，Beautiful Soup 就不能自动识别编码方式了；此时，开发者仅仅需要说明一下原始编码方式就可以了

## 解析器

Beautiful Soup 支持 Python 标准库中包含的 HTML 解析器，但它也支持许多第三方 Python 解析器，其中包含 lxml 解析器

| 解析器                 | 用法                                              | 优点                               | 缺点             |
| ------------------- | ----------------------------------------------- | -------------------------------- | -------------- |
| Python 标准库          | BeautifulSoup(markup 【源代码】, "html.parser"【解析器】) | Python 标准库，执行速度适中                | 文档容错能力差        |
| **lxml 的 HTML 解析器** | **BeautifulSoup(markup, "lxml")**               | **速度快，文档容错能力强**                  | **需要安装 C 语言库** |
| lxml 的 XML 解析器      | BeautifulSoup(markup, "lxml-xml")               | 速度快，唯一支持 XML 的解析器                | 需要安装 C 语言库     |
| html5lib            | BeautifulSoup(markup, "html5lib")               | 最好的容错性，以浏览器的方式解析文档生成 HTML5 格式的文档 | 速度慢，不依赖外部扩展    |

## 创建 Beautiful Soup 类对象

要使用 Beautiful Soup 库解析网页

1. 首先需要创建 BeautifulSoup 对象

2. 通过将字符串或 HTML 文件传入 Beautiful Soup 库的构造方法可以创建一个 BeautifulSoup 对象

```python
'''
使用 bs4.BeautifulSoup() 方法，可以创建 Beautiful Soup 类对象
常见方式如下
'''
# 1.'lxml' 进行解析
soup = bs4.BeautifulSoup(html, 'lxml')
# 2.'html.parser' 进行解析
soup = BeautifulSoup(html, 'html.parser')
# 3.'html5lib' 进行解析
soup = BrautifulSoup(html, 'html5lib')
```

> 返回的 soup 以树形结构呈现，具有明显的层级结构
> 
> **注意：这里的 html 为解码好的 html 内容**

![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20084731.png)

```python
from bs4 import BeautifulSoup
# 创建 Beautiful Soup 对象，并制定解析器为 lxml
soup = BeautifulSoup(html_doc, 'lxml')
# 输出数据类型
print(type(soup))
# 输出解析后的 html 代码
print(soup)
```

Out:

![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20085110.png)

![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20085207.png)

```python
# 输出格式化后的 html 代码
print(soup.prettify())
```

Out:

![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20085341.png)

> 得到 soup 这种树形结构之后，就可以针对 soup 对象提取我们想要的信息

## 获取节点内容

使用 Beautiful Soup 可以直接调用节点的名称，然后再调用对应的 string 属性便可以获取到节点内的文本信息

在单个节点结构层次非常清晰的情况下，使用这种方式提取节点信息是非常快的

1. 获取节点对应的代码
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20135336.png)
   
   ```python
   from bs4 import BeautifulSoup
   # 创建 Beautiful Soup 对象，并指定解析器为 lxml
   soup = BeautifulSoup(html_doc, 'lxml')
   print('head节点：\n', soup.head)
   print('body节点：\n', soup.body)
   print('title节点：\n', soup.title)
   print('p节点：\n', soup.p)
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20135512.png)

2. 获取节点属性
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20135751.png)
   
   每个节点都会包含有多个属性，例如：class, id等
   
   如果已经选择了一个指定的节点名称，那么只需要调用 attrs 即可获取这个节点下的所有属性
   
   ```python
   from bs4 import BeautifulSoup
   # 创建 Beautiful Soup 对象，并指定解析器为 lxml
   soup =BeautifulSoup(html_doc, 'lxml')
   print('meta节点中的属性：\n', soup.meta.attrs)
   print('link节点中的属性：\n', soup.link.attrs)
   print('div节点中的属性：\n', soup.div.attrs)
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20135929.png)
   
   在以上运行结果可以发现，attrs 的返回值为**字典类型**，字典中的元素分别是属性名称与对应的值；所以中 attrs 后面添加 [] 并在括号内添加属性名称即可获取指定属性对应的值，如 `soup.meta.attrs['http-equiv']`

3. 获取节点包含的文本内容
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20135751.png)
   
   ```python
   # 实现文本内容只需要在节点名称后面添加 string 属性即可
   from bs4 import BeautifulSoup
   # 创建 Beautiful Soup 对象，并指定解析器为 lxml
   soup =BeautifulSoup(html_doc, 'lxml')
   print(soup.title.string)
   print(soup.h3.string)
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20140319.png)

4. 嵌套获取节点内容
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20135751.png)
   
   ```python
   # HTML代码中的每个节点都会出现嵌套的可能，而使用 Beautiful Soup 获取每个节点的内容时，可以通过 '.' 直接获取下一个节点中的内容（当前节点的子节点）
   from bs4 import BeautifulSoup
   # 创建 Beautiful Soup 对象，并指定解析器为 lxml
   soup =BeautifulSoup(html_doc, 'lxml')
   print(soup.head.title)
   print(soup.head.title.string)
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20140443.png)

## 使用 find_all() 与 find() 方法获取内容

HTML 中获取比较复杂的内容时，可以使用 find_all() 与 find() 方法；调用这些方法，然后传入指定的参数即可灵活的获取节点中的内容

### find_all() 方法

Beautiful Soup 提供了一个 find_all() 方法

该方法可以获取所有符合条件的内容

`find_all(name = None, attrs = {}, recursive = True, text = None, limit = None, **kwargs)`

> 在 find_all() 方法中，常用参数时 name, attrs 以及 text
> 
> 返回的内容时一个列表

- name
  
  指定节点名称，指定该参数以后将返回一个可迭代对象（类似列表），所有符合条件的内容均为对象中的一个元素
  
  ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20140734.png)
  
  ```python
  from bs4 import BeautifulSoup
  # 创建 Beautiful Soup 对象，并指定解析器为 lxml
  soup =BeautifulSoup(html_doc, 'lxml')
  print(soup.find_all(name = 'p'))
  print(type(soup.find_all(name = 'p')))
  ```
  
  Out:
  
  ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20141030.png)
  
  > 'bs4.element.ResultSet' 类型的数据与 Pytjon 中发列表类似，如果想获取可迭代对象中的某条数据可以使用切片的方式进行
  > 
  > 如获取所有 p 节点中的第一个
  > `soup.find_all(name = 'p')[0]`
  
  ```python
  '''
  因为 'bs4.element.ResultSet' 数据中的每一个元素都是 'bs4.element.Tag'
  类型，所以可以直接对某一个元素进行嵌套获取
  '''
  print(type(soup.find_all(name = 'p')[0]))
  print(soup.find_all(name = 'p')[0].find_all(name = 'a'))
  ```
  
  Out:
  
  ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20141205.png)

- attrs
  
  ```python
  '''
  attrs 参数表示通过指定属性进行数据的获取工作，填写 attrs 参数时，
  需要填写字典类型的参数值
  '''
  from bs4 import BeautifulSoup
  # 创建 Beautiful Soup 对象，并指定解析器为 lxml
  soup =BeautifulSoup(html_doc, 'lxml')
  print(soup.find_all(name = 'p', attrs = {'class': 'p-1'}))
  print(soup.find_all(name = 'p', attrs = {'value': '1'}))
  ```
  
  Out:
  
  ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20141318.png)

### find() 方法

find_all() 方法可以获取所有符合条件的节点内容，而 find() 方法只能获取第一个匹配的节点内容，使用方法与 find_all() 相同

## 层级选择

Beautiful Soup 模块提供了层级选择来获取节点内容，如果时 Tag 或者时 BeautifulSoup 对象都可以直接调用 **select()** 方法，然后填写指定参数即可通过层级选择获取到节点中的内容

> 常用的层级选择
> 
> - 直接填写字符串类型的节点名称
> - .class: 指定 class 属性值
> - #id: 指定 id 属性值

![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20141527.png)

1. 通过标签名查找
   
   ```python
   from bs4 import BeautifulSoup
   # 创建 Beautiful Soup 对象，并指定解析器为 lxml
   soup =BeautifulSoup(html_doc, 'lxml')
   print(soup.select('title'))
   print(soup.select('a'))
   print(soup.select('p'))
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20141622.png)

2. 通过类名查找
   
   ```python
   from bs4 import BeautifulSoup
   # 创建 Beautiful Soup 对象，并指定解析器为 lxml
   soup =BeautifulSoup(html_doc, 'lxml')
   print(soup.select('.sister'))
   # 获取 p 标签下 class 类名为 title 的标签
   print(soup.select('p.title'))
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20141732.png)

3. 通过 id 名查找
   
   ```python
   from bs4 import BeautifulSoup
   # 创建 Beautiful Soup 对象，并指定解析器为 lxml
   soup =BeautifulSoup(html_doc, 'lxml')
   print(soup.select('#link1'))
   # 寻找所有 id 为 link1 或 link2 的标签
   print(soup.select('#link1,#link2'))
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20141904.png)

4. "标签名+类名+id 名" 的查找组合
   
   ```python
   # 组合查找是将标签名与类名、id 名进行的组合查找
   # 例如查找 p 标签中，id 等于 link1 的内容，二者需要用空格分开
   print(soup.select('p #link1'))
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20142446.png)

5. "标签名+属性" 查找
   
   查找时还可以加入属性元素，**属性需要用中括号括起来，注意属性和标签属于同一节点，所以中间不能加空格，否则会无法匹配到**
   
   ```python
   from bs4 import BeautifulSoup
   # 创建 Beautiful Soup 对象，并指定解析器为 lxml
   soup =BeautifulSoup(html_doc, 'lxml')
   # 查找 a 标签下存在 herf 属性标签
   print(soup.select('a[href]'))
   # 寻找 href 属性值是以 'http://example.com/' 开头的 a 标签
   print(soup.select('a[href^="http://example.com/"]'))
   # 寻找 href 属性值是以 tillie 为结尾的 a 标签
   print(soup.select('a[href$="tillie"]'))
   # 寻找 href 属性值中存在字符串 '.com/el' 的标签 a
   print(soup.select('a[href*=".com/el"]'))
   print(soup.select('a[href="http://example.com/elsie"]'))
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20143720.png)
   
   > 若某个标签具有多个属性，可以使用如下方式同时使用多个属性
   > 
   > `soup.select('标签[属性1][属性2]')`

6. 直接子标签查找【层级查找】
   
   ```python
   # 非常类似于 XPath 的层级查找
   from bs4 import BeautifulSoup
   # 寻找 head 标签下子节点 title 标签
   print(soup.select("head > title"))
   # 寻找 body 标签下子节点标签
   print(soup.select('body > p > a'))
   # 寻找 p 标签子节点中 id='link1' 的标签
   print(soup.select('p > #link1'))
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20144156.png)
   
   ```python
   # 更复杂地，在层级查找时，在每一层级上，标签和属性可以联合使用
   # 创建 Beautiful Soup 对象，并指定解析器为 lxml
   soup =BeautifulSoup(html_doc, 'lxml')
   print(soup.select('p > a[href].sister'))
   print(soup.select('p > a[href]#link1'))
   ```
   
   Out:
   
   ![](D:\HANSHAN\Reptile\Picture\屏幕截图%202022-06-13%20144353.png)

## 案例: 爬取动物庄园小说

```python
# 全局取消证书验证
import ssl
ssl.__create__default__https__context = ssl.__create__unverified__context
# 模拟浏览器头部
header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36'}
import requests
from bs4 import BeautifulSoup
import time

for i in range(10):
    # 主页网址 'http://www.kanunu8.com/book3/6879'
    https://www.kanunu8.com/book3/6879/131780.html
    url = 'https://www.kanunu8.com/book3/6879/{}.html'.format(131779+i)
    # 获取网页源代码
    res_source = requests.get(url, headers = header)
    '''
    观察 <meta http-equiv="Content-Type" content="text/html; charset=gbk">
    可以发现该网页编码格式为 'gbk'
    '''
    # 解码网页内容
    res = res_source.content.decode('gbk')
    # 提取小说内容
    # 通过观察网页源代码可以发现，小说内容均在 p 标签中
    # 可以通过如下方式提取小说内容
    soup = BeautifulSoup(res, 'lxml')  # 解析 HTML 字符串
    content = soup.select('p')[0].text
    content_filter = ''.join(content)
    # 将提取的内容保存到 txt 文件中
    with open('chapter{}.txt'.format(i+1), 'a') as fp:
        fp.write(content_filter)
    print('第', i+1, '章下载完成')
    time.sleep(1)
```
