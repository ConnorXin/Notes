# 大数据应用的主要计算模式

- <font color = cornflowerblue>批处理计算</font>
  
  - 针对大规模数据的批量处理；主要技术有 MapReduce, Spark 等

- <font color = cornflowerblue>流计算</font>
  
  - 针对流数据的实时计算处理；主要技术有 Spark, Storm, Flink, Flume, Dstream 等

- <font color = cornflowerblue>图计算</font>
  
  - 针对大规模图结构数据的处理；主要技术有 GraphX, Gelly, Giraph, PowerGraph 等

- <font color = cornflowerblue>查询分析计算</font>
  
  - 大规模数据的存储管理和查询分析；主要技术有 Hive, Impala, Dremel, Cassandra

<mark>Hadoop 大数据生态圈</mark>

![](./屏幕截图%202022-11-17%20222631.png)

# 华为云大数据服务

数据开发、测试、应用一站式服务

![](./屏幕截图%202022-11-18%20004608.png)

**<mark>华为云 MRS 服务综述</mark>**

MapReduce 服务 (MapReduce Service / MRS) 是一个在华为云上部署的管理 Hadoop 系统的服务，一键即可完成部署 Hadoop 集群

MRS 提供租户完全可控的一站式企业级大数据集群云服务，完全兼容开源接口，结合华为云计算、存储优势及大数据行业经验，为客户提供高性能、低成本、灵活易用的全栈大数据平台，轻松运行 Hadoop, Spark, HBase, Kafka, Storm 等大数据组件，并具备在后续根据业务需要进行定制开发的能力，帮助企业快速构建海量数据信息处理系统，并通过对海量信息数据实时与非实时的分析挖掘，发现全新价值点和企业商机；

**<mark>MRS 服务应用场景</mark>**

1. <font color = cornflowerblue>海量数据离线分析场景</font>
   
   - 低成本
     
     利用 OBS 实现低成本存储
   
   - 海量数据分析
     
     利用 Hive 实现 TB / PB 级的数据分析
   
   - 可视化的导入导出工具
     
     通过可视化导入导出工具 Loader，将数据导出到 DWS，完成 BI 分析
   
   ![](./屏幕截图%202022-11-18%20010543.png)

2. <font color = cornflowerblue>海量数据存储场景</font>
   
   - **实时**
     
     利用 Kafka 实现海量汽车的消息实时接入
   
   - 海量数据存储
     
     利用 HBase 实现 PB 级别海量数据存储，并实现毫秒级数据查询
   
   - 分布式数据查询
     
     利用 Spark 实现海量数据的分析查询
   
   ![](./屏幕截图%202022-11-18%20010827.png)

3. <font color = cornflowerblue>低时延实时数据分析场景</font>
   
   - 实时数据采集
     
     利用 Flume 实现实时数据采集，并提供丰富的采集和存储连接方式
   
   - 海量的数据源接入
     
     利用 Kafka 实现万级别的电梯数据的实时接入
   
   ![](./屏幕截图%202022-11-18%20011048.png)

# HDFS 分布式文件系统

## 概述及应用场景

Hadoop 分布式文件系统 HDFS 是一种旨在商品硬件上运行的分布式文件系统，核心作用就是为了存储数据；

HDFS 具有高度的容错能力，旨在部署在低成本硬件上；

HDFS 提供对应用程序数据的高吞吐量访问，并且适用于具有大数据集的应用程序

HDFS 放宽了一些 POSIX 要求，以实现对文件系统数据的流式访问；

HDFS 最初是作为 Apache Nutch Web 搜索引擎项目的基础结构而构建的；

HDFS 是 Apache Hadoop Core 项目的一部分；

**应用场景**

![](./屏幕截图%202022-11-17%20232913.png)

## 相关概念

1. <font color = cornflowerblue>**计算机集群结构**</font>
   
   分布式文件系统把文件分布存储到多个计算机节点上，成千上万的计算机节点构成计算机集群；
   
   目前的分布式文件系统所采用的计算机集群都是由普通硬件构成的，这就大大降低了硬件上的开销；
   
   ![](./屏幕截图%202022-11-17%20232955.png)

2. <font color = cornflowerblue>基本系统架构</font>
   
   ![](./屏幕截图%202022-11-17%20233828.png)
   
   - 元数据所在节点称之为 namenode;
   
   - 工作节点 slave01 等称之为 datanode;

3. <font color = cornflowerblue>Block-块</font>
   
   HDFS 默认一个块128MB，一个文件被分成多个块，以块作为存储单位；
   
   块的大小远远大于普通文件系统，开源最小化寻址开销；
   
   抽象的块概念可以带来**以下几个明显的好处**
   
   - 支持大规模文件存储
   
   - 简化系统设计
   
   - 适合数据备份
   
   | NameNode                     | DataNode                          |
   | ---------------------------- | --------------------------------- |
   | 存储源数据                        | 存储文件内容                            |
   | 元数据保存在内存中                    | 文件内容保存在磁盘                         |
   | 保存文件，block, datanode 之间的映射关系 | 维护了 block id 到 datanode 本地文件的映射关系 |
   
   > ***NameNode***
   > 
   > 元数据的存储分为两个部分
   > 
   > 1. FsImage
   >    
   >    ![](./屏幕截图%202022-11-18%20012445.png)
   >    
   >    存储的是数据的目录结构以及文件信息
   > 
   > 2. EditLog
   >    
   >    存储的是用户操作的信息，操作日志，即针所有针对文件的创建、删除、重命名等操作
   > 
   > ***DataNodes***
   > 
   > 数据节点是分布式文件系统 HDFS 的工作节点，负责数据的存储和读取，会根据客户端或者是 NameNode 的调度来进行数据的存储和检索，并且向名称节点定期发送自己所存储的块列表；
   > 
   > 每个数据节点中的数据会被保存在各自节点的本地 Linux 文件系统中；

## 体系架构

![](./屏幕截图%202022-11-18%20013244.png)

**<mark>HDFS 单名称节点体系结构的局限性</mark>**

HDFS 只设置唯一一个 NameNode，虽然大大简化了系统设计，但也带来了一些明显的局限性，

- 命名空间的限制
  
  名称节点是保存在内存中的，因此，名称节点能够容纳的对象个数收到内存空间大小的限制

- 性能的瓶颈
  
  所有的操作都要经过 NameNode 进行保存；
  
  整个分布式文件系统的吞吐量，受限于单个 NameNode 的吞吐量

- 隔离问题
  
  由于集群中只有一个 NameNode，只有一个命名空间，因此，无法对不同应用程序进行隔离

- 集群的可用性
  
  一旦这个唯一的 NameNode 发生故障，会导致整个集群变得不可用

---

> 1. **HDFS 命名空间管理**
>    
>    - HDFS 的命名空间包含目录、文件和块
>    
>    - HDFS 使用的是传统的分级文件体系，因此，用户可以像使用普通文件系统一样，创建、删除目录和文件，在目录间转移文件，重命名文件等
>    
>    - NameNode 维护文件系统命名空间；对文件系统命名空间或其属性的任何更改均由 NameNode 记录
> 
> 2. **通信协议**
>    
>    - 所有的 HDFS 通信协议都是构建在 TCP / IP 协议基础之上的
>    
>    - 客户端通过一个可配置的端口向 NameNode 主动发起 TCP 连接，并使用客户端协议与 NameNode 进行交互
>    
>    - NameNode 与 DataNodes 之间则使用数据节点协议进行交互
>    
>    - 客户端与 DataNodes 的交互通过 RPC (Remote Procedure Cell) 来实现；在设计上，NameNode 不会主动发起 RPC，而是响应来自客户端和 DataNodes 的 RPC 请求
> 
> 3. **客户端**
>    
>    - 客户端是用户操作 HDFS 最常用的方式，HDFS 在部署时都提供了客户端
>    
>    - HDFS 客户端是一个库，包含 HDFS 文件系统接口，这些接口隐藏了 HDFS 实现中大部分复杂性
>    
>    - 严格来说，客户端并不算是 HDFS
>    
>    - 客户端可以支持打开、读取、写入等常见操作，并且提供了类似 Shell 的命令行方式来访问 HDFS 中的数据
>    
>    - HDFS 也提供了 Java API，作为应用程序访问文件系统的客户端编程接口【编写 Java 代码同样可以远程操作】

## 关键特性

1. <mark><font color = cornflowerblue>HDFS 高可用性 (HA)</font></mark>

2. <mark><font color = cornflowerblue>元数据持久化</font></mark>

3. <mark><font color = cornflowerblue>数据副本机制</font></mark>

4. <mark><font color = cornflowerblue>HDFS 数据完整性保障</font></mark>

## 数据读写流程

**<mark>写入数据</mark>**

![](./屏幕截图%202022-11-18%20021921.png)

**<mark>读入数据</mark>**

![](./屏幕截图%202022-11-18%20022054.png)

# ZooKeeper 分布式协调服务

# Hive

## 概述

Hive 是基于 Hadoop 的数据仓库软件，可以查询和管理 PB 级别的分布式数据；

Hive 构建在基于静态批处理的 Hadoop 之上，Hadoop 通常都有较高的延迟并且在作业和调度的时候需要大量的开销；

**<mark>特性</mark>**

- 灵活方便 ETL (extract / transform / load)

- 支持 Tez, Spark 等多种计算引擎

- 可以直接访问 HDFS 文件以及 HBase

- 易用易编程

<mark>**应用场景**</mark>

1. 数据挖掘
   
   - 用户行为分析
   
   - 兴趣分区
   
   - 区域展示

2. 非实时分析
   
   - 日志分析
   
   - 文本分析

3. 数据汇总
   
   - 每天 / 每周用户点击数
   
   - 流量统计

4. 数据仓库
   
   - 数据抽取
   
   - 数据加载
   
   - 数据转换

**<mark>Hive 与传统数据仓库比较</mark>**

|      | Hive                              | 传统数据仓库                                       |
| ---- | --------------------------------- | -------------------------------------------- |
| 存储   | HDFS，理论上有无限拓展的可能                  | 集群存储，存在容量上限，而且伴随着容量的增长，计算速度急剧下降，不适合大数据量的数据应用 |
| 执行引擎 | 默认执行引擎 Tez                        | 可以选择更加高效的算法来执行查询，也可以进行更多的优化措施来提高速度           |
| 使用方法 | HQL                               | SQL                                          |
| 灵活性  | 元数据存储独立于数据存储之外，从而解耦合元数据与数据        | 低，数据用途单一                                     |
| 分析速度 | 计算依赖于集群规模，易拓展，在大数据量情况下，远远快于普通数据仓库 | 在数据容量较小时非常快速，数据量较大时，急剧下降                     |
| 索引   | 效率较低                              | 高效                                           |
| 易用性  | 需要自己开发出模型，灵活度较高，但是易用性较低           | 继承一整套成熟的报表的解决方案，可以较为方便的进行数据分析                |
| 可靠性  | 数据存储在 HDFS ，可靠性不高，容错性高            | 可靠性较低，一次查询失败需要重新开始；数据容错依赖于硬件 Raid            |
| 依赖环境 | 依赖硬件较低，可适应一般的普通机器                 | 依赖于高性能的商业服务器                                 |
| 价格   | 开源产品                              | 商用比较昂贵                                       |

# HBase

## 基本介绍

一个高可靠性、高性能、面向列、可伸缩的分布式存储系统

- 适用于存储大表数据（表的规模可以达到数十行以及数百万列），并且对大表数据的读、写访问可以达到实时级别

- 利用 Hadoop HDFS 作为其文件存储系统，提供实时读写的分布式数据库系统

- 利用 ZooKeeper 作为协同服务

<mark>应用场景</mark>

![](./屏幕截图%202022-11-19%20010134.png)

## 数据模型

简单来说，应用程序是以表的方式在 HBase 存储数据的；

表是由行和列构成的，所有的列是从属于某一个列族的；

行和列的交叉点称之为 cell，cell 是版本化的；cell 的内容是不可分割的字节数组；

表的行键也是一段字节数组，所以任何东西都可以保存进去，不论是字符串或者数字；

HBase 的表是按 key 排序的，排序方式是针对字节的；

所有的表都必须要有逐渐 - key

![](./屏幕截图%202022-11-19%20014713.png)

- <mark>表</mark>: HBase 采用表来组织数据，表由行和列组成，列划分为若干个列族

- <mark>行</mark>: 每个 HBase 表都由若干行组成，每个行由行键 (row key) 来标识

- <mark>列族</mark>: 一个 HBase 表被分组成许多 “列族” (Column Family) 的集合，它是基本的访问控制单元

- <mark>单元格</mark>: 在 HBase 表中，通过行、列族和列限定符确定一个单元格 cell，单元格中存储的数据没有数据类型，总被视为字节数组 byte[]

- <mark>时间戳</mark>: 每个单元格都保存着同一份数据的多个版本，这些版本采用时间戳进行索引

## 体系架构

![](./屏幕截图%202022-11-19%20015820.png)

**主要包括两个主要的功能组件**

- 一个 HMaster 主服务器
  
  > - 主服务器 HMaster 负责管理和维护 HBase 表的分区信息，维护 HRegionServer 列表，分配 Region，负载均衡
  > 
  > - 客户端并不是直接从 HMaster 主服务器上读取数据，而是在获得 Region 的存储位置信息后，直接从 HRegionServer 上读取数据
  > 
  > - 客户端并不依赖 HMaster，而是通过 **Zookeeper 来获得 Region 位置信息**，大多数客户端甚至从来不和 HMaster 通信，这种设计方式使得 HMaster 负载很小
  
  <mark>主要负责表和 Region 的管理工作</mark>
  
  1. 管理用户对表的增加、删除、修改、查询等操作
  
  2. 实现不同 HRegionServer 之间的负载均衡
  
  3. 在 Region 分裂或合并后，负责重新调整 Region 的分布
  
  4. 对发生故障失效的 HRegionServer 上的 Region 进行迁移

- HRegionServer服务器
  
  是 HBase 最核心的模块
  
  - 负责维护分配给自己的 Region
  
  - 响应用户的读写请求

## 关键流程

1. **<font color = cornflowerblue>用户读写数据过程</font>**
   
   - 写入数据时，被分配到相应 HRegionServer 去执行
     
     用户数据首先被写入到 Hlog 中，再写入 MenStore 中，最终写到磁盘上形成 StoreFile
   
   - 读取数据时，HRegionServer 会首先访问 MenStore 缓存，如果找不到，再去磁盘上面的 StoreFile 中寻找
   
   > 由于是基于内存，因此读写效率很高

2. **<font color = cornflowerblue>缓存的刷新</font>**
   
   - 系统会**周期性**地把 MenStore 缓存里的内容刷写到磁盘的 StoreFile 文件中，清空缓存，并在 Hlog 里面写入一个标记
   
   - 每次刷写都**生成一个新的 StoreFile 文件**，因此，每个 Store 包含多个 StoreFile 文件
     
     ![](./屏幕截图%202022-11-19%20135732.png)
   
   - 每个 HRegionServer 都有**一个自己的 HLog 文件**，每次启动都检查该文件，缺人最近一次执行缓存刷新操作之后是否发生新的写入操作；如果发现更新，则先写入 MenStore，再刷写到 StoreFile，开始为用户提供服务

3. **<font color = cornflowerblue>StoreFile 的合并</font>**
   
   - 每次刷写都生成一个新的 StoreFile，数量太多，影响查找速度
   
   - 调用 `Store.compact()` 把多个合并成一个
   
   - 合并操作比较耗费资源，只有数量达到一个阈值才启动合并

4. **<font color = cornflowerblue>Store 工作原理</font>**
   
   - Store 是 **HRegionServer 的核心**
   
   - 多个 StoreFile 合并成一个
   
   - 单个 StoreFile 过大时，又触发分裂操作
   
   ![](./屏幕截图%202022-11-19%20140414.png)

5. **<font color = cornflowerblue>HLog 工作原理</font>**
   
   - 分布式环境必须考虑系统出错；HBase 采用 HLog **保证系统恢复**
   
   - HBase 系统为每个 HRegionServer 配置了一个 HLog 文件，它是一种**预写式日志 (Write Ahead Log)**
   
   - 用户更新数据必须首先写入日志后，才能写入 MenStore 缓存，并且，直到 MenStore 缓存内容对应的日志已经写入磁盘，该缓存内容才能被刷写到磁盘
   
   - Zookeeper 实时检测每个 HRegionServer 状态，当某个 HRegionServer 发生故障时，**Zookeeper 会通知 HMaster**
   
   - HMaster 首先会处理该故障 HRegionServer 上面遗留的 HLog 文件，这个遗留的文件包含了来自多个 Region 对象的日志记录
   
   - 系统会根据每天日志记录所属的 Region 对象对 HLog 数据**进行拆分**，分别放到相应 Region 对象的目录下，然后，在将失效的 Region 重新分配到可用的 HRegionServer 中，并把与该 Region 对象相关的 HLog 日志记录也发送给相应的 HRegionServer
   
   - HRegionServer 领取到分配给自己的 Region 对象与该 Region 对象相关的 HLog 日志记录以后，会重新做一遍日志记录中的各种操作，**把日志记录中的数据写入到 MenStore 缓存中**，然后刷新到磁盘的 StoreFile 文件中，完成数据恢复
   
   <mark>共用日志优点</mark>
   
   提高对表的写操作性能
   
   <mark>共用日志缺点</mark>
   
   恢复时需要分拆日志

## 突出特点

1. <mark><font color = cornflowerblue>多 HFile 的影响</font></mark>
   
   ![](./屏幕截图%202022-11-19%20142847.png)
   
   HFile 文件数目越来越多，读取时延也越来越大

2. <mark><font color = cornflowerblue>Compaction</font></mark>
   
   **目的**是为了减少同一个 Region 中同一个 ColumnFamily 下面的小文件 HFile 数目，从而提升读取性能
   
   *Compaction 分为 Minor, Major 两类*
   
   - Minor
     
     **小范围**的 Compaction；有最少和最大文件数目限制；通常会选择一些连续时间范围的小文件进行合并
   
   - Major
     
     涉及该 Region 该 ColumnFamily 下面的**所有的 HFile 文件**

3. <mark><font color = cornflowerblue>OpenScanner</font></mark>
   
   OpenScanner 的过程中，会创建两种不同的 Scanner 来读取 HFile, MemStore 的数据
   
   - HFile 对应的 Scanner 为 StoreFileScanner
   
   - MenStore 对应的 Scanner 为 MemStoreScanner

4. <mark><font color = cornflowerblue>BloomFilter</font></mark>
   
   用来优化一些随机读取的场景，即 **Get 场景**；
   
   它可以被用来快速判断一条用户数据在一个大的数据集合 *（该数据集合的大部分数据都没法被加载到内存中）* 中是否存在
   
   - BloomFilter 在判断一个数据是否存在时，拥有一定的误判率；但对于 “用户数据 xxx 不存在” 的判断结果是可信的
   
   - HBase 的 BloomFilter 的相关数据，被保存在 HFile 中

## 性能优化

1. **<font color = cornflowerblue>行键  Row Key</font>**
   
   行键是按照**字典序**存储，因此，设计行键时，要充分利用这个排序特点，将经常一起读取的数据存储到一块，将最近可能会被访问的数据放在一块
   
   > e.g.
   > 
   > 如果最近写入 HBase 表中的数据是最可能被访问的，**可以考虑将时间戳作为行键的一部分**，由于是字典序排序，所以可以使用 Long.MAX_VALUE - timestamp 作为行键，这样能保证新写入的数据在读取时可以被快速命中

2. **<font color = cornflowerblue>构建 HBase 二级索引</font>**
   
   - HBase 只有一个针对行键的索引
   
   - 访问 HBase 表中的行，只有**三种方式**
     
     - 通过单个行键访问
     
     - 通过一个行键的区间来访问
     
     - 全表扫描

## 常用 Shell 命令

1. `create`: 创建表

2. `list`: 列出 HBase 中所有的表信息

3. `put`: 向表、行、列指定的单元格添加数据

4. `scan`: 浏览表的相关信息

5. `get`: 通过表名、行、列、时间戳、时间范围和版本号来获得相应单元格的值

6. `enable / disable`: 使表有效或无效

7. `drop`: 删除表

# MapReduce

## 基本介绍

1. ***<mark><font color = cornflowerblue>概述</font></mark>***
   
   基于 Google 发布的 MapReduce 论文设计开发，基于分而治之的思想，用于大规模数据集的**并行计算和离线计算**，具有如下特点
   
   - <u>高度抽象的编程思想</u>
     
     程序员仅需描述做什么，具体怎么做交由系统的执行框架处理
   
   - <u>良好的扩展性</u>
     
     可通过添加节点以扩展集群能力
   
   - <u>高容错性</u>
     
     通过计算迁移或数据迁移等策略提高集群的可用性与容错性

## 功能与架构

1. **<font color = cornflowerblue>MapReduce 过程</font>**
   
   - MapReduce 计算过程可具体分为两个阶段，Map 阶段 和 Reduce 阶段；其中，Map 阶段输出的结果就是 Reduce 阶段的输入
   
   - Map 面对的是杂乱无章的互不相关的数据，它解析每个数据，从中**提取 key 和 value**，也就是提取了数据的特征
   
   - 到了 Reduce 阶段，数据是以 key 后面跟着若干个 value 来组织的，这些 value 有相关性，在次基础上我们可以做进一步的处理以便得到结果

2. **<font color = cornflowerblue>工作流程</font>**
   
   ![](./屏幕截图%202022-11-19%20200058.png)

3. **<font color = cornflowerblue>WordCount 程序功能</font>**
   
   ![](./屏幕截图%202022-11-19%20210137.png)
   
   <mark>Map</mark>
   
   ![](./屏幕截图%202022-11-19%20210201.png)
   
   <mark>Reduce</mark>
   
   ![](./屏幕截图%202022-11-19%20210213.png)

# YARN

## 基本介绍

1. <mark>***<font color = cornflowerblue>概述</font>***</mark>
   
   Apache Hadoop YARN (Yet Another Resource Negotiator)，中文名为 “另一种资源协调者”；
   
   它是一种新的 Hadoop 资源管理器，它是一个通用资源管理系统，可为上层应用提供统一的资源管理和调度，它的引入为集群在利用率、资源统一管理和数据共享等方面带来了巨大好处；
   
   ![](./屏幕截图%202022-11-19%20194913.png)

## 功能与架构

ResoueceManager (RM) 是整个集群可用资源的主节点，帮助 YARN 系统管理其上的分布式应用；它同 NodeManagers (NMs) 和 ApplicationMasters (AMs) 一起工作

- NodeManagers 从 ResourceManager 获取指令并管理本节点的可用资源

- ApplicationMasters 的职责是从 ResourceManager 谈判资源，并为 Node Managers 启动容器

## 资源管理和任务调度

1. **<font color = cornflowerblue>资源调度与分配</font>**
   
   - 在 Hadoop1.0 版本中资源调度通过 MRv1 来进行，存在着很好缺陷
   
   - 在 Hadoop2.0 中正式引入了 YARN 框架，以便更好地完成集群的资源调度与分配

2. **<font color = cornflowerblue>资源管理</font>**
   
   - 每个 NodeManager 可分配的内存和 CPU 的数量可用通过配置选项设置 (可在 YARN 服务配置页面配置)
     
     - yarn.nodemanager.resource.memory-mb
       
       可以分配给容器的物理内存的大小
     
     - yarn.nodemanager.vmem-pmem-ratio
       
       虚拟内存跟物理内存的比值
     
     - yarn.nodemanager.resource.cpu-vcore
       
       可分配给容器的 CPU 核数
   
   - 在 Hadoop3.x 版本中，YARN 资源模型已被推广为除了 CPU 和 Memory 以外，还包括 GPU 资源、软件 licenses 或本地附加存储器 (locally-attached storage) 之类的资源

3. **<font color = cornflowerblue>YARN 的三种资源调度器</font>**
   
   在 YARN 中，负责给应用分配资源的叫作 Scheduler (调度器)；在 YARN 中，根据不同的策略，共有三种调度器可供选择
   
   - ***FIFO Scheduler***
     
     把应用按提交的顺序排成一个对列，这是一个**先进先出对列**，在进行资源分配的时候，先给对列中最头上的应用进行分配资源，待最头上的应用需求满足后再给下一个分配，以此类推
   
   - ***Capacity Scheduler***
     
     允许多个组织共享整个集群，每个组织可以获得集群的一部分计算能力
   
   - ***Fair Scheduler***
     
     为所有的应用分配公平的资源 (对公平的定义可以通过参数来设置)

# Spark 基于内存的分布式计算

## 概述

1. **<mark><font color = Red>简介</font></mark>**
   
   - 2009年诞生于美国**加州大学伯克利分校 AMP 实验室**
   
   - Apache Spark 是一种基于内存的快速、通用、可扩展的大数据计算引擎
   
   - Spark 是一站式解决方案，集<u>批处理、实时流处理、交互式查询、图计算与机器学习</u>与一体

2. **<mark><font color = red>应用场景</font></mark>**
   
   - 批处理可用于 ETL (抽取、转换、加载)
   
   - 机器学习可用于自动判断淘宝的买家评论是好评还是差评
   
   - 交互式分析可用于查询 Hive 数据仓库
   
   - 流处理可用于页面点击流分析、推荐系统、舆情分析等实时业务

3. **<mark><font color = red>特点</font></mark>**
   
   - *“轻”*
     
     Spark 核心代码只有3万行
   
   - *“快”*
     
     Spark 对小数据集可达到亚秒级的延迟
   
   - *“灵”*
     
     Spark 提供了不同层面的灵活性
   
   - *“巧”*
     
     巧妙借力现有大数据组件

## 数据结构

1. **<font color = cornflowerblue>核心概念 RDD</font>**
   
   - ***RDD (Resilient Distributed Datasets)*** 即弹性分布式数据集，是一个只读的、可分区的分布式数据集
   
   - RDD **默认存储在内存**，当内存不足时，溢写到磁盘
   
   - RDD 数据以**分区的形式**在集群中存储
   
   - RDD 具有血统机制 (Lineage)，发生数据丢失时，可快速进行数据恢复

2. **<font color = cornflowerblue>RDD 的依赖关系</font>**
   
   ![](./屏幕截图%202022-11-19%20230828.png)
   
   - *Dependency* (依赖)
     
     - 窄依赖是指父 RDD 的每个分区最多被一个 RDD 的一个分区所用
     
     - 宽依赖是指父 RDD 的每个分区对应一个子 RDD 的多个分区
   
   - *Lineage* (血统): 依赖的链条
     
     - RDD 数据集通过 Lineage 记住了它是如何从其他 RDD 中演变过来的

3. **<font color = cornflowerblue>宽窄依赖的区别——算子</font>**
   
   - <u>窄依赖</u>指的是每一个父 RDD 的 Partition 最多被子 RDD 的一个 Partition 使用；如 map, filter, union
   
   - <u>宽依赖</u>指的是多个父 RDD 的 Partition 会依赖同一个父 RDD 的 Partition；如 groupByKey, reduceByKey, sortByKey

4. **<font color = cornflowerblue>宽窄依赖的区别</font>**
   
   <mark>容错性</mark>
   
   - 假如某个节点出故障
     
     - 窄依赖：只要重算和子 RDD 分区对应的父 RDD 分区即可
     
     - 宽依赖：极端情况下，所有的父 RDD 分区都要进行重新计算
   
   - 如下图所示，b1 分区丢失，则要重新计算 a1, a2, a3
     
     ![](./屏幕截图%202022-11-20%20003551.png)
   
   <mark>传输</mark>
   
   - <u>宽依赖</u>往往对应着 shuffle 操作，需要在运行过程中将同一个父 RDD 的分区传入到不同的子 RDD 分区中，中间可能涉及多个节点的数据传输
   
   - <u>窄依赖</u>的每个父 RDD 的分区只会传入到一个子 RDD 分区中，通常可以在一个节点内完成转换

5. **<font color = cornflowerblue>RDD 的 Stage (阶段) 划分</font>**
   
   ![](./屏幕截图%202022-11-20%20004219.png)
   
   Spark 此时利用了前文提到的依赖管理，调度器从 DAG 图 (有向无环图) 末端出发，逆向遍历整个依赖关系链，遇到 ShuffleDependency (宽依赖关系的一种叫法) 就断开，遇到 NarrowDenpendency 就将其加入到当前 stage

6. **<font color = cornflowerblue>RDD 操作类型</font>**
   
   - ***Creation Operation***
     
     创建操作，用于 RDD 创建工作；
     
     RDD 创建只有两种方法，一种是来自于内存集合和外部存储系统；另一种是通过转换操作生成的 RDD
   
   - ***Transformation Operation***
     
     转换操作，将 RDD 通过一定的操作转变成新的 RDD, RDD 的转换操作是惰性操作，它只是定义了一个新的 RDD，**并没有立即执行**
     
     | Transformation                 | explanation                                                                                 |
     | ------------------------------ | ------------------------------------------------------------------------------------------- |
     | map(func)                      | 对调用 map 的 RDD 数据集中的每个 eleemn 都使用 func，然后返回一个新的 RDD                                          |
     | filter(func)                   | 对调用 filter 的 RDD 数据集中的每个元素都使用 func，然后返回一个包含使 func 为 true 的元素构成的 RDD                         |
     | reduceBykey(func, [numTasks])  | 类似 groupBykey，但每个 key 对应的 value 会根据提供的 func 进行计算以得到一个新的值                                    |
     | join(otherDataset, [numTasks]) | 如果数据集是 (K， V) 关联的数据集是 (K, W)，返回 (K, (V, W)) 同时支持 leftOuterJoin, rightOutJoin, fullOuterJoin |
   
   - ***Control Operation***
     
     进行 RDD 持久化，可以让 RDD 按不同的存储策略保存在磁盘或内存中，比如 cache 接口默认将 RDD 缓存在内存中
   
   - ***Action Operation***
     
     能够触发 Spark 运行的操作；
     
     Spark 中行动操作分为两类，一类是操作输出计算结果；另一类是将 RDD 保存到外部文件系统或者数据库中

## 原理与架构

![](./屏幕截图%202022-11-20%20015345.png)

1. **<mark><font color = red>Spark Core</font></mark>**
   
   类似于 MR 的分布式内存计算框架，最大的特点是将中间计算结果直接放在内存中，提升计算性能；自带了 Standalone 模式的资源管理框架，同时也支持 YARN

2. <mark>**<font color = red>Spark SQL</font>**</mark>
   
   Spark 中用于结构化数据处理的模块；在 Spark 应用中，可以无缝的使用 SQL 语句亦或是 DataFrame API 对结构化数据进行查询

3. **<mark><font color = red>Spark Streaming</font></mark>**
   
   微批处理的**流处理引擎**，将流数据分片以后用 SparkCore 的计算引擎中进行处理；相对于 Storm，实时性稍差，优势体现在吞吐量上

## 典型案例——WordCount

![](./屏幕截图%202022-11-20%20015951.png)

# Flink 流批一体分布式实时处理引擎

## 原理及架构

1. **<font color = cornflowerblue>简介</font>**
   
   - Apache Flink 是为**分布式、高性能**的流处理应用程序打造的开源流处理框架；Flink 不仅能提供同时支持高吞吐和 exactly-once (只执行一次) 语义的实时计算，还能提供批量数据处理
   
   - 相较于市面上的其他数据处理引擎，Flink 和 Spark 都开源同时支持流处理和批处理；但是 Spark 的技术理念是基于批处理来模拟流的计算；
     
     而 Flink 则完全相反，它采用的是**基于流计算来模拟批处理**

2. **<font color = cornflowerblue>关键机制</font>**
   
   ***四个机制: 状态、时间、检查点、窗口***
   
   ![](./屏幕截图%202022-11-20%20021725.png)

3. **<font color = cornflowerblue>核心理念</font>**
   
   - Flink 与其他流计算引擎的<mark>最大区别</mark>，就是<mark>状态管理</mark>
   
   - e.g.    做数据处理，要对数据进行统计，如 Sum, Count, Min, Max，这些值是需要存储的；因为要不断更新，这些值或者变量可以理解为**一种状态**
   
   - Flink 提供了**内置的状态管理**，可以把工作式状态存储在 Flink 内部，而不需要把它存储在外部系统，这样做的好处有
     
     - 降低了计算引擎对外部系统的依赖，使得部署、运维更加简单
     
     - 对性能带来了极大的提升

4. **<font color = cornflowerblue>核心概念</font>**
   
   - <mark>DataStream</mark>
     
     它是含有重复数据的**不可修改**的集合 collection，DataStream 中元素的数量是**无限的**
     
     ![](./屏幕截图%202022-11-20%20022542.png)
   
   - <mark>DataSet</mark>
     
     Flink 系统可对数据集进行转换（例如过滤，映射，联接，分组），数据集可从读取文件或本地集合创建；
     
     结果通过接收器  Sink 返回，接收器可以将数据写入（分布式）文件或标准输出（例如命令行终端）

5. **<font color = cornflowerblue>Flink 程序</font>**
   
   程序由 Source, Transformation 和 Sink 三部分组成；其中
   
   - ***Source*** 主要负责数据的读取，支持 HDFS, kafka 和文本等；
   
   - ***Transformation*** 主要负责对数据的转换操作；
   
   - ***Sink*** 负责最终数据的输出，支持 HDFS, kafka 和文本输出等
   
   - 在各部分之间流转的数据称为 ***流  stream***

6. **<font color = cornflowerblue>数据处理</font>**
   
   Apache Flink 同时支持**批处理和流处理**，也能用来做一些基于事件的应用
   
   - 如果处理一个事件（或一条数据）的结果只跟事件本身的内容有关，称为**无状态处理**；反之结果还和之前处理过的事件有关，称为**有状态处理**；
   
   - **无状态计算**会观察每个独立的事件，并且会在最后一个时间出结果，例如一些报警和监控，一直观察每个事件，当触发警报的事件来临就会触发警告
   
   - **有状态计算**就会基于多个事件来输出结果，比如说计算过去一个小时的平均温度等等

7. **<font color = cornflowerblue>有界流与无界流</font>**
   
   - <mark>无界流</mark>
     
     有定义流的开始，但没有定义流的结束；数据源会无休止地产生数据；
     
     无界流的数据**必须持续处理**，即数据被读取后需要立刻处理，不能等到所有数据都到达再处理；
     
     因为输入是无限的，在任何时候输入都不会完成；
     
     处理无界数据通常要求以特定顺序摄取事件，例如事件发生的顺序，以便能够推断结果的完整性
   
   - <mark>有界流</mark>
     
     有定义流的开始，也有定义流的结束，有界流可以在读取所有数据后再进行计算；
     
     有界流所有数据可以被排序，所以并不需要有序摄取；
     
     有界流处理通常被称为**批处理**
   
   ![](./屏幕截图%202022-11-20%20030256.png)

8. **<font color = cornflowerblue>流与批处理机制</font>**
   
   - Flink 的两套机制分别对应各自的 API ***(DataStream API; DataSet API)***，在创建 Flink 作业时，并不能通过将两者混合在一起来同时利用 Flink 的所有功能
   
   - Flink 支持两种**关系型**的 API，Table API 和 SQL；
     
     这两个 API 都是批处理和流处理**统一**的 API，这意味着在无边界的实时数据和有边界的历史数据流上，关系型 API 会以相同的语义执行查询，并产生相同的结果
     
     > Table API / SQL 正在以流批统一的方式称为分析型用例的主要 API

## Flink 的 Time 与 Window

1. **<mark><font color = red>时间背景</font></mark>**
   
   在流处理编程中，对于时间的处理是**非常关键的**；
   
   比如计数的例子，事件流数据（例如服务器日志数据、网页点击数据和交易数据）不断产生，我们需要用 key 将事件分组，并且每隔一段时间就针对每一个 key 对应的事件计数

2. **<mark><font color = red>流处理中的事件分类</font></mark>**
   
   在实际场景中，每个事件的时间可以分为三种
   
   - event time: 时间发生时的时间
   
   - ingestion time: 事件到达流处理系统的时间
   
   - processing time: 事件被系统处理的时间
   
   e.g.
   
   ![](./屏幕截图%202022-11-20%20032506.png)

3. **<mark><font color = red>Window 概述</font></mark>**
   
   - 流式计算是一种被设计用于处理无限数据集的数据处理引擎，而 Window 是**一种切割无限数据为有限块**进行处理的手段
   
   - Window 是无限数据流处理的**核心**，它将一个无限的 stream 拆分成有限大小的 "buckets" 捅，我们可以在这些桶上做计算操作

4. <mark>**<font color = red>Window 类型</font>**</mark>
   
   根据应用类型分为两类
   
   - ***CountWindow***
     
     数据驱动，按照指定的数据条数生成一个 Window，与时间无关
   
   - ***TimeWindow***
     
     时间驱动，按照时间生成 Window

5. **<mark><font color = red>Tumnling Window (翻滚窗口)</font></mark>**
   
   - 翻滚窗口能将数据流切分成不重叠的窗口，每一个事件只能属于一个窗口
   
   - 翻滚窗具有固定的尺寸，不重叠
   
   ![](./屏幕截图%202022-11-20%20033450.png)

6. **<mark><font color = red>Sliding Window (滑动窗口)</font></mark>**
   
   - 滑动窗口和翻滚窗口类似，区别在于：**滑动窗口可以有重叠的部分**
   
   - 在滑窗中，一个元素可以对应多个窗口
   
   ![](./屏幕截图%202022-11-20%20033821.png)

7. **<mark><font color = red>Session Window (会话窗口)</font></mark>**
   
   - 会话窗口不重叠，没有固定的开始和结束时间
   
   - 当会话窗口**在一段时间内没有接收到元素时会关闭会话窗口**
   
   - 后续的元素将会被分配给新的会话窗口
   
   ![](./屏幕截图%202022-11-20%20034042.png)

## 容错

1. **<font color = cornflowerblue>Flink 容错机制</font>**
   
   为了保证程序的容错恢复以及程序启动时其状态恢复，Flink 任务都会开启 Checkpoint 或者触发 Savepoint 进行状态保存
   
   - ***Checkpoint***
     
     这种机制保证了实时程序运行时，即使突然遇到异常也能够进行**自我恢复**
   
   - ***Savepoint***
     
     是在某个时间点程序状态**全局镜像**，以后程序在进行升级，或者修改并发度等情况，还能从保存的状态位继续启动恢复
     
     > Savepoint 可以看作是 Checkpoint 在特定时期的一个状态快照

# Flume 海量日志聚合

## 简介及架构

1. **<mark><font color = red>简介</font></mark>**
   
   Flume 是**流式日志采集工具**，Flume 提供对数据进行简单处理并且写到各种数据接收方，Flume 提供从本地文件、实时日志、REST 消息、Thrift、Avro、Syslog、Kafka 等数据源上收集数据的能力
   
   > 一般采集到 Kafka

2. <mark>**<font color = red>应用场景</font>**</mark>
   
   - 提供从**固定目录**下采集日志信息到目的地 (HDFS, HBase, Kafka) 能力
   
   - 提供**实时采集日志信息**到目的地的能力
   
   - Flume 支持级联 (多个 Flume 对接起来)，合并数据的能力
   
   - Flume 支持按照用户定制采集数据的能力

3. **<mark><font color = red>Flume Agent 架构</font></mark>**
   
   - *<u>基础架构</u>*
     
     可以单节点直接采集数据，主要应用于集群内数据
     
     ![](./屏幕截图%202022-11-20%20210133.png)
   
   - *<u>多 agent 架构</u>*
     
     Flume 可以将多个节点连接起来，将最初的数据源经过收集，存储x到最终的存储系统中，主要应用于集群外的数据导入到集群内
     
     ![](./屏幕截图%202022-11-20%20210301.png)

4. **<mark><font color = red>基本概念</font></mark>**
   
   - ***Source***
     
     - 数据源，即是产生日志信息的源头，Flume 会将原始数据建模抽象成自己处理的数据对象：events 事件；
       
       并将 events 批量放到一个或多个 Channels
     
     - **Source 有驱动和轮询2种类型**
       
       1. 驱动型 source
          
          是外部主动发送数据给 Flume, 驱动 Flume 接收数据
       
       2. 轮询 source
          
          是 Flume 周期性主动去获取数据
     
     - Source **必须**至少和一个 channel 关联
   
   - ***Channel***
     
     - Channel 位于 Source 和 Sink 之间，用于**临时缓存**进来的 events，当 Sink 成功地将 events 发送到下一跳的 channel 或最终目的，events 从 Channel 移除
     
     - Channels 支持事务，可以**连接任何数量**的 Source 和 Sink
     
     - **Memory Channel**
       
       消息存放在内存中，提供高吞吐，但不提供可靠性；可能丢失数据
     
     - **File Channel**
       
       对数据持久化；但是配置较为麻烦，需要配置数据目录和 checkpoint 目录；
       
       不同的 file channel 均需要配置一个 checkpoint 目录
     
     - **JDBC Channel**
       
       内置的 derby 数据库，对 event 进行了持久化，提供高可靠性；可以取代同样具有持久特性的 file channel
   
   - ***Sink***
     
     - Sink 负责将 events 传输到下一个 flume 或最终目的，成功完成后将 events 从 channel 移除
     
     - 必须作用于一个确切的 channel
     
     | Sink 类型            | 说明                          |
     | ------------------ | --------------------------- |
     | hdfs sink          | 将数据写到 hdfs 上                |
     | avro sink          | 使用 avro 协议将数据发送给另下一跳的 Flume |
     | thift sink         | 同 avro，不过传输协议为 thrift       |
     | file roll sink     | 将数据保存在本地文件系统中               |
     | hbase sink         | 将数据写到 HBase 中               |
     | Kafka sink         | 将数据写入到 Kafka 中              |
     | MorphlineSolr sink | 将数据写入到 Solr 中               |

## 关键特性介绍

1. **<font color = cornflowerblue>支持采集日志文件</font>**
   
   Flume 支持将集群内的日志文件采集并归档到 HDFS, HBase, Kafka 上，供上层应用对数据分析、清洗数据使用
   
   ![](./屏幕截图%202022-11-20%20220612.png)

2. **<font color = cornflowerblue>支持多级级联和多路复制</font>**
   
   Flume 支持将多个 Flume 级联起来，同时级联节点内部支持数据复制
   
   ![](./屏幕截图%202022-11-20%20220640.png)

3. **<font color = cornflowerblue>级联消息压缩、加密</font>**
   
   Flume 级联节点之间的数据传输支持压缩和加密，提升数据传输效率和安全性
   
   ![](./屏幕截图%202022-11-20%20220937.png)
   
   > Source 到 Channel 到 Sink 等进程内部没有加密的必要，属于进程内部数据交换

4. **<font color = cornflowerblue>数据监控</font>**
   
   ![](./屏幕截图%202022-11-20%20221337.png)

5. **<font color = cornflowerblue>传输可靠性</font>**
   
   - Flume 在传输数据过程中，采用**事务管理方式**，保证传输过程中数据不会丢失，增强了数据传输的可靠性，同时缓存在 channel 中的数据如果采用 file channel，进程或者节点重启数据不会丢失
   
   - Source 至 channel 支持事务，channel 到 sink 支持事务；事件采集或发送失败，会重新采集或发送
   
   - 在传输过程中，若下一个的 Flume 节点故障或者数据接收异常时，可以自动切换到另外一路上继续传输
     
     ![](./屏幕截图%202022-11-20%20225909.png)

# Loader

## 简介

1. **<mark><font color = red>什么是 Loader</font></mark>**
   
   Loader 是实现 FusionInsight HD 与关系型数据库、文件系统之间交换数据和文件的数据加载工具；
   
   提供可视化向导式的作业配置管理界面；提供定时调度任务，周期性执行 Loader 作业；
   
   在界面中可指定多种不同的数据源、配置数据的清洗和转换步骤、配置集群存储系统等

2. **<mark><font color = red>应用场景</font></mark>**
   
   ![](./屏幕截图%202022-11-20%20234238.png)

## 作业管理

# Kafka

## 简介

1. **<font color = cornflowerblue>简介</font>**
   
   - Kafka 是一个分布式、分区的、多副本的、多订阅者，基于 **zookeeper** 协调的分布式日志系统
   
   - 主要应用场景有：日志收集系统和消息系统
   
   - 分布式消息传递基于可靠的消息对列，在客户端应用和消息系统之间异步传递消息；
     
     有两种主要的消息传递模式：**1）点对点传递模式、2）发布-订阅模式**；
     
     大部分的消息系统选用发布-订阅模式
     
     ***Kafka 就是一种发布-订阅模式***

2. **<font color = cornflowerblue>点对点消息传递模式</font>**
   
   在点对点消息系统中，消息持久化到一个对列中；此时，将有一个或多个消费者消费队列中的数据；但是**一条消息只能被消费一次**；当一个消费者消费了队列中的某条数据之后，**该条数据则从消息对列中删除；**
   
   该模式即使有多个消费者同时消费数据，也能保证数据处理的顺序
   
   ![](./屏幕截图%202022-11-21%20100419.png)

3. **<font color = cornflowerblue>发布-订阅消息传递模式</font>**
   
   在发布-订阅消息系统中，消息被持久化到一个 topic 中；与点对点消息系统不同的是，消费者**可以订阅一个或多个 topic** (分类)，消费者可以消费该 topic 中所有的数据，同一条数据**可以被多个消费者消费**，数据被消费后**不会立马删除；**
   
   在发布-订阅消息系统中，消息的生产者称为发布者，消费者称为订阅者;
   
   ![](./屏幕截图%202022-11-21%20101044.png)

4. **<font color = cornflowerblue>特点</font>**
   
   - **高吞吐率**
     
     即使在廉价的商用机器上也能做到单机支持每秒100000条消息的传输
   
   - 支持消息分区，及分布式消费，同时保证每个分区内消息顺序传输
   
   - 同时支持**离线数据处理**和**实时数据处理**
   
   - Scale out
     
     支持在线水平扩展
   
   ![](./屏幕截图%202022-11-21%20101925.png)

## 架构与功能

1. **<mark><font color = red>拓扑结构图</font></mark>**
   
   ![](./屏幕截图%202022-11-21%20102345.png)
   
   一个典型的 Kafka 集群中包含若干 Producer【<u>可以是 Web 前端产生的 Page View，或者是服务器日志，系统 CPU，Memory 等</u>】，若干 Broker 【<u>Kafka 支持水平扩展，一般 broker 代理数量越多，**集群吞吐率越高**</u>】，若干 Consumer，以及一个 Zookeeper 集群;
   
   Kafka 通过  Zookeeper 管理集群配置，选举 Leader，以及在 Consumer 发生变化时进行 rebalance；Priducer 使用 push 模式将消息发布到 Broker，Consumer 使用 pull 模式从 Broker 订阅并消费消息
   
   - ***Broker***
     
     Kafka 集群包含一个或多个服务实例，这些服务实例被称为 Broker
   
   - ***Topics***
     
     每条发布到 Kafka 集群的消息都有一个类别，这个类别即 Topic，也可以理解为一个存储消息的队列；
     
     e.g.    天气作为一个 Topic，每天的温度消息就可以存储在 “天气“ 这个对列里
     
     ![](./屏幕截图%202022-11-21%20104107.png)
   
   - ***Partition***
     
     Kafka 将 Topic 分成一个或多个 Partition，每个 Partition 都是有序且不可变的消息队列，且在物理上对应一个文件夹，该文件夹存储这个 Partition 的所有消息和索引文件
     
     ![](./屏幕截图%202022-11-21%20105559.png)
     
     - <u>**Partition offset**</u>
       
       每条消息在文件中的位置称为 offset (偏移量)，offset 是一个 long 型数字，它唯一标记一条消息；消费者通过 (offset, partiton, topic) 跟踪记录
       
       ![](./屏幕截图%202022-11-21%20110100.png)
       
       【*存储机制*】
       
       1. Consumer 在从 broker 读取消息后，可以选择 commit，该操作会在 Kafka 中保存
       
       2. 该 Consumer 在该 Partition 中读取的消息的 offset；
          
          该 Consumer 下一次再读该 Partition 时会从下一条开始读取
       
       3. 通过这一特性可以保证同一消费者从 Kafka 中不会重复消费数据
   
   - ***Producer***
     
     负责发布消息到 Kafka Broker
   
   - ***Consumer***
     
     消息消费者，从 Kafka Broker 读取消息的客户端
   
   - ***Consumer Group***
     
     - 每个 Consumer 属于一个特定的 Consumer Group 【<u>可为每个 Consumer 指定 group name</u>】
     
     - 每条消息只能被 Consumer Group 中的一个 Consumer 消费，但可以被多个 Consumer Group 消费；即**组间数据是共享的，组内数据是竞争的**
     
     ![](./屏幕截图%202022-11-21%20111015.png)

## 数据管理

### 数据存储可靠性

1. **<font color = cornflowerblue>Partition Replica</font>**
   
   ![](./屏幕截图%202022-11-21%20112045.png)

### 数据传输可靠性

【**消息传输保障通常有以下三种**】

1. **<mark><font color = red>最多一次  At Most Once</font></mark>**
   
   - 消息可能丢失；
   
   - 消息不回重复发送和处理；

2. **<mark><font color = red>最少一次  At Lease Once</font></mark>**
   
   - 消息不回丢失；
   
   - 消息可能会重复发送和处理；

3. **<mark><font color = red>仅有一次  Exactly Once</font></mark>**【Kafka 尚未实现】
   
   - 消息不回丢失；
   
   - 消息仅被处理一次；

【**可靠性保证**】

4. **<mark><font color = red>acks 机制</font></mark>**
   
   producer 需要 server 接收到数据之后发出的确认接受的信号，此项配置就是指 producer 需要多少个这样的缺人信号；此配置**实际上代表了数据备份的可用性**；以下设置为常用选项
   
   - ***acks = 0***
     
     设置为0表示 producer 不需要等待任何确认收到的信息
   
   - ***acks = 1***
     
     这意味着至少要等待 leader 已经成功将数据写入本地 log，但是并没有等待所有 follower 是否成功写入；这种情况下，如果 follower 没有成功备份数据，而此时 leader 又挂掉，则消息会丢失
   
   - ***acks = all***
     
     这意味着 leader 需要等待所有备份都成功写入日志，这种策略会保证只要有一个备份存活就不会丢失数据；这就是最强的保证

### 旧数据处理方式

- Kafka 把 Topic 中一个 Parition 大文件分成多个小文件段，通过多个小文件段，就容易定期清楚或删除已经消费完文件，减少磁盘占用

- 配置位置
  
  \$KAFKA_HOME/config/server.properties

# 
