# Hadoop 概述

## Hadoop

Hadoop 是一个由 Apache 基金会所开发的**分布式系统基础架构**

主要解决海量数据的**存储**和海量数据的**分析计算**问题

广义上来说，Hadoop 通常是指一个更广泛的概念——Hadoop 生态圈

## Hadoop 发展历史

1. 创始人 Doug Cutting，为了实现与 Google 类似的全文搜索功能，他在 Lucene 框架基础上进行优化升级，查询引擎和索引引擎

2. 2001年年底 Lucene 成为 Apache 基金会的一个子项目

3. 对于海量数据的场景，Lucene 框架面对与 Google 同样的困难，存储海量数据困难，检索海量速度慢

4. 学习和模仿 Google 解决这些问题的办法：微型版 Nutch

5. Doug Cutting 参考 Google 的三篇论文

   GFS --> HDFS

   Map-Reduce --> MR

   BigTable --> HBase

6. 2003-2004年，Google 公开了部分 GFS 和 MapReduce 思想的细节，以此为基础 Doug Cutting 等人用了2年的业余时间实现了 DFS, MapReduce 机制，使 Nutch 性能飙升

7. 2005年 Hadoop 作为 Lucene 的子项目 Nutch 的一部分正式引入 Apache 基金会

8. 2006年3月份，MapReduce, Nutch Distributed File System (NDFS) 分别被纳入到 Hadoop 项目中，Hadoop 就此正式诞生，标志着大数据时代来临

## Hadoop 优势

**高可靠性**

Hadoop 底层维护多个数据副本，所以即使 Hadoop 某个计算元素或存储出现故障，也不会导致数据的丢失

**高扩展性**

在集群间分配任务数据，可方便的扩展数以千计的节点，动态增加 / 删除服务器

**高效性**

在 MapReduce 的思想下，Hadoop 是并行工作的，以加快任务处理速度

**高容错性**

能够自动将失败的任务重新分配

## Hadoop 组成

1. Hadoop 1.x

   MapReduce  >>  计算 + 资源调度

   HDFS  >>  数据存储

   Common  >>  辅助工具

2. Hadoop 2.x

   MapReduce  >>  计算

   Yarn  >>  资源调度

   HDFS  >>  数据存储

   Common  >>  辅助工具

3. Hadoop 3.x

   在组成上没有区别

# HDFS 架构概述

1. **NameNode  nn**

   存储文件的**元数据**，如文件名，文件目录结构，文件属性（生成时间、副本数、文件权限），以及每个文件的块列表和块所在的 DataNode 等

2. **DataNode  dn**

   在本地文件系统存储文件块数据，以及快数据的校验和

3. **Secondary NameNode  2nn**

   每隔一段时间对 NameNode 元数据备份

# YARN 架构概述

![](./%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-03-14%20153646.png)

# MapReduce 架构概述

海量数据的计算

分为两个阶段 Map, Reduce

1. Map 阶段并行处理输入数据
2. Reduce 阶段对 Map 结果进行汇总

# 大数据技术生态体系

![](./%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-03-15%20114401.png)


