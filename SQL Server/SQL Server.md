# 数据库

## 数据库概念

**狭义**
存储数据的仓库

**广义**
可以对数据进行存储和管理的软件以及数据本身，统称为数据库

**数据库组成**
数据库由表、关系、操作组成

## 数据库存在的背景

- 几乎所有的应用软件的后台都需要数据库

- 数据库存储数据占用空间小，容易持久保存

- 存储比较安全

  创建完数据库最终得到的是两个文件，一个 mdf 数据文件，一个 LDF 日志文件

- 容易维护和升级

- 数据库移植比较容易

- 简化对数据的操作

- B/S 架构里面包含数据库

## 数据库的优势

对内存数据操作时编程语言的强项，但是对硬盘数据操作却是编程语言的弱项

对硬盘数据操作是数据库的强项，是数据库研究的核心问题

**三个方面学习数据库**

1. 数据库是如何存储数据的

   字段  记录  表  约束 (主键 外键 唯一键 非空 CHECK DEFAULT 触发器(更复杂的约束))

2. 数据库是如何操作数据的

   INSERT  UPDATE  DELETE  T-SQL  存储过程  函数  触发器

3. 数据库是如何显示数据的

   SELECT

# 图像化界面对数据库的操作

**创建数据库** 
数据库 - 右键 - 新建数据库 - 名称 - 更改路径 - 确定

**删除数据库**
选中数据库 - 右键 - 删除 - 关闭现有链接 - 确定

**分离数据库**

分离 scott 数据库

选中 scott - 右击 - 任务 - 分离 - 勾选删除连接 - 勾选更新统计信息 - 确定

**附加数据库**

附加 scott 数据库

1. 下载到 scott 数据库

2. 查看本系统用户是否对两个文件为完全控制，若不是完全控制那么在附加的时候会报错导致失败

   查看方法：文件右击 - 属性 - 安全 - Users(计算机名\Users)
   查看权限是否为完全控制

   ![](./%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-06-22%20110524.png)

   若不是，点击编辑将完全控制勾选

   ![](./%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-06-22%20110648.png)

3. 进入 SQL Server 进行附加数据库

   右击数据库 - 附加 - 添加 - 找到 scott 数据库 - 添加即可

# 新建登录账号

1. 在 Windows 身份验证下，右击登录名 - 新建

2. 常规页面选择 SQL Server 身份验证 - 输入登录名密码 - 取消勾选强制密码过期

   ![](./%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-06-22%20113045.png)

# 数据库存储数据

## 表的相关数据

1. 字段 (属性)

   模拟一个事物的某一个特征

2. 记录 (元组)

   字段的组合，表示的是一个具体的事物

3. 表

   记录的组合，表示的是同一类型事物的集合

4. 表 记录 字段的关系

   字段是事物的属性
   记录是事物本身
   表是事物的集合

## 主键 & 外键

**主键**

是数据库设计者选中用来在 DBMS 中区分一个关系中不同元组的<u>候选键</u>，主键的选择会影响某些实现问题

能够唯一标识一个事物的一个字段或者多个字段组合

- 通常是整数，不建议使用字符串
- 通常不允许修改，除非记录被删除

**外键**

若一个表中的若干字段是来自另外若干个表的主键或唯一键，则这若干个字段就是外键；
外键不一定是来自另外的表，也可能来自本表的主键

事物和事物之间的关系是通过外键来体现的

**含有外键的表叫外键表，外键字段来自的那一张表叫作主键表**

```sql
CREATE TABLE dept (
dept_id INT PRIMARY KEY,
dept_name NVARCHAR(100) NOT NULL,
dept_address NVARCHAR(100)
)

CREATE TABLE emp (
emp_id INT CONSTRAINT pk_emp_id PRIMARY KEY,  -- CONSTRAINT 后的 pk_emp_id 为主键名称; CONSTRAINT 是约束的意思
emp_name NVARCHAR(20) NOT NULL,
emp_sex NCHAR(1),
dept_id INT CONSTRAINT fk_dept_id FOREIGN KEY REFERENCES dept(dept_id)  -- 外键约束
)
-- 先删除外键表 再删除主键表
DROP TABLE emp
DROP TABLE dept
```

# 约束

对一个表中的属性操作的限制叫作约束

## 主键约束

不允许重复元素，避免了数据的冗余

## 外键约束

通过外键约束从语法上保证了本事物所关联的其他事物一定是存在的

事物和事物之间的关系是通过外键来体现的

## CHECK 约束

保证事物属性的取值在**合法范围内**

```sql
-- CHECK 约束
CREATE TABLE student (
stu_id INT PRIMARY KEY,
stu_sla INT CHECK (stu_sla >= 0 AND stu_sla <=5000)
)
```

## DEFAULT 约束

字段默认值
保证事物的属性一定会有一个值

```Sql
-- DEFAULT 约束
CREATE TABLE student_default (
stu_id INT PRIMARY KEY,
stu_sex NCHAR(1) DEFAULT '男'
)
```

## UNIQUE 约束

```SQL
-- UNIQUE 约束
CREATE TABLE student_unique (
stu_id INT PRIMARY KEY,
stu_name NVARCHAR(200) UNIQUE
)
```

保证了事物属性的取值不允许重复，但允许其中有且仅有一列为空

## UNIQUE 与主键的区别

UNIQUE 不允许重复，但允许有且仅有一列为空，而主键不允许重复也不允许为空

```SQL
-- UNIQUE 与 主键
CREATE TABLE student_unique_primarykey (
stu_id INT PRIMARY KEY,
stu_name NVARCHAR(50) UNIQUE NOT NULL,
stu_email NVARCHAR(50) UNIQUE NOT NULL,
stu_address NVARCHAR(50)
)
```

# 创建数据库

```sql
CREATE DATABASE test ON
(NAME = test,
FILENAME = 'E:\Microsoft SQL Server\MSSQL14.MSSQLSERVER\MSSQL\DATA\test.mdf',
SIZE = 10,
MAXSIZE = 50,
FILEGROWTH = 5)
LOG ON
(NAME = test_log,
FILENAME = 'E:\Microsoft SQL Server\MSSQL14.MSSQLSERVER\MSSQL\DATA\test_log.ldf',
SIZE = 5MB,
MAXSIZE = 25MB,
FILEGROWTH = 5MB)
GO
```

# 创建表并设置主外键约束

```sql
-- 先建外键表 dept
CREATE TABLE dept (
dept_id INT PRIMARY KEY,
dept_name NVARCHAR(100) NOT NULL,
dept_address NVARCHAR(100)
)

CREATE TABLE emp (
emp_id INT CONSTRAINT pk_emp_id PRIMARY KEY,  -- CONSTRAINT 后的 pk_emp_id 为主键名称; CONSTRAINT 是约束的意思
emp_name NVARCHAR(20) NOT NULL,
emp_sex NCHAR(1),
dept_id INT CONSTRAINT fk_dept_id FOREIGN KEY REFERENCES dept(dept_id)  -- 外键约束
)
```

# 添加数据

```sql
INSERT
INTO Students_H
VALUES('S01', '王建平', '男', '1995-10-12', 'D01')
```

# 关系型数据库

## 关系

表与表之间的联系
通过设置不同形式的外键来体现表与表的不同关系

**分类**

【二元联系类型的转换】

- 若实体间联系是1:1，
- 若实体间联系是1:n，
- 若实体间联系是 m:n，则将联系类型也转换成关系模式，其属性为两端实体类型的键加上联系类型的属性，而键为两端实体键的组合

1. 一对一

   既可以把表 A 的主键充当表 B 的外键
   也可以把表 B 的主键充当表 A 的外键

   可以在两个实体类型转换成的两个关系模式中任意一个关系模式的属性中加入另一个关系模式的键和联系类型的属性

2. 一对多

   ![](./%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-06-22%20191632.png)

   把 A 的主键充当 B 的外键

   在多的一方添加外键

   则在 n 端实体类型转换成的关系模式中加入1端实体类型的键和联系类型的属性

   e.g. 班级：学生，表示一个班级可有多个学生，而一个学生只能归属于一个班级

3. 多对多

   e.g. 教室：班级; 教师：学生

   将联系类型也转换成关系模式，其属性为两端实体类型的键加上联系类型的属性，而键为两端实体键的组合

   即多对多必须通过单独一张表来表示

   ![](./%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-06-22%20193425.png)

   若一个老师上不同的课程则键需要三列实体的组合

   ![](./%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-06-22%20194034.png)

## 关系的实现

```sql
-- 关系的实现
CREATE TABLE class_data (
class_id INT PRIMARY KEY,
class_num INT NOT NULL,
class_name NVARCHAR(100)
)

CREATE TABLE teacher_data (
teacher_id INT PRIMARY KEY,
teacher_name NVARCHAR(100)
)

-- 第三张表来模拟教师和班级的关系
CREATE TABLE class_teacher_mapping (
class_id INT CONSTRAINT fk_class_id FOREIGN KEY REFERENCES class_data(class_id),
teacher_id INT CONSTRAINT fk_teacher_id FOREIGN KEY REFERENCES teacher_data(teacher_id),
course	NVARCHAR(100),
CONSTRAINT pk_class_teacher_id PRIMARY KEY (class_id, teacher_id, course)
)
```

# 查询

## 计算列

```sql
-- 计算列
SELECT ename, sal * 12 YEARsal
FROM emp
-- 中文字段的话
SELECT ename, sal * 12 "年薪"
FROM emp
-- 也可以这样写  AS 可以省略
SELECT ename, sal * 12 AS YEARsal
FROM emp
```

## DISTINCT

不允许重复的，相当于去重，也可以过滤掉重复的 null

```sql
-- DISTINCT
SELECT DISTINCT deptno 
FROM emp
-- (comm, deptno) 组合过滤
SELECT DISTINCT comm, deptno
FROM emp
```

```sql
-- 以下命名会报错 逻辑有冲突
SELECT deptno, DISTINCT comm
FROM emp
```

## BETWEEN

查找范围在...之间

```sql
-- BETWEEN
-- 查找工资在1500到3000之间，包括1500，3000的所有员工信息
SELECT *
FROM emp
WHERE sal BETWEEN 1500 AND 3000
-- 等价于
SELECT *
FROM emp
WHERE sal >= 1500 AND sal <= 3000

SELECT *
FROM emp
WHERE sal < 1500 OR sal >3000
-- 等价于
SELECT *
FROM emp
WHERE sal NOT BETWEEN 1500 AND 3000
```

## IN

属于若干个孤立的值

```sql

-- IN
-- 查询工资为1500和3000的员工信息
SELECT *
FROM emp
WHERE sal IN (1500, 3000)
-- 等价于
SELECT *
FROM emp
WHERE sal = 1500 OR sal = 3000

-- 既不是1500，也不是3000，也不是5000的信息
SELECT *
FROM emp
WHERE sal NOT IN (1500, 3000, 5000)
-- 等价于
SELECT *
FROM emp
WHERE sal != 1500 AND sal != 3000 AND sal != 5000  -- 不等于符号有两种方式 1.!=; 2.<>
```

## TOP

查看数据前 n 行 / %

```sql
-- TOP
-- emp 所有信息的前两行
SELECT TOP 2 *
FROM emp
-- emp 所有信息的前15%
SELECT TOP 15 PERCENT *
FROM emp
```

## NULL

任何数据类型都允许为空

```SQL
-- NULL
-- 奖金非空员工信息
SELECT *
FROM emp
WHERE comm <> NULL  -- error 无数据

SELECT *
FROM emp
WHERE comm != NULL  -- error 无数据

SELECT *
FROM emp
WHERE comm = NULL  -- error 无数据

SELECT *
FROM emp
WHERE comm IS NULL

-- 不为空
SELECT *
FROM emp
WHERE comm IS NOT NULL
```

```sql
-- 计算每个员工包含了奖金的年薪 假设 comm 是一年的奖金
SELECT empno, ename, sal * 12 + comm "YESRsal"
FROM emp  -- 发现有很多 NULL, 结果不对
	-- NULL 不能够参与数学运算 若参与运算则结果为 NULL
```

## ISNULL

```sql
-- ISNULL (comm, 0) 表示如果 comm 为 NULL，返回0，否则返回 comm
SELECT ename, sal * 12 + ISNULL (comm, 0) "YEARsal"
FROM emp
```

## ORDER BY

```SQL
-- 将工资1500 - 3000的工资最高的前四个输出
SELECT TOP 4 *
FROM emp
WHERE sal BETWEEN 1500 AND 3000
ORDER BY sal DESC
```

```sql
SELECT *
FROM emp
ORDER BY sal  -- 默认升序

-- 先按照 deptno 排序，deptno 相同时，按照 sal 排序
SELECT *
FROM emp
ORDER BY deptno, sal

-- DESC 放在哪就对哪产生影响  这里对 sal 产生影响
SELECT *
FROM emp
ORDER BY deptno, sal DESC
```

## 模糊查询

**通配符**

`%`
表示包含某字符的任意长度数据

`_`
表示任意单个字符

`[a-f]`
a 到 f 中的任意单个字符

```sql
-- 第二个字母是 A-F 任意字符
SELECT *
FROM emp
WHERE ename LIKE '_[A-F]%'
```

`[a, f]`
a 或 f

`[^a-c]`
表示不是 a - c 任意单个字符

```sql
-- 姓“刘”且全名不多于3个汉字的学生的姓名 Sname 和出生日期 Birthday
SELECT Sname, Birthday
FROM Students_H
WHERE Sname LIKE '刘__'

-- 所有不姓刘的学生姓名 Sname 和出生日期 Birthday
SELECT Sname, Birthday
FROM Students_H
WHERE Sname NOT LIKE '刘%'

-- 课程名为“DB_设计”的课程号 Cno 和学分 Credits
SELECT Cno, Credits
FROM Courses
WHERE Cname LIKE 'DB\_设计' ESCAPE '\'
```

```sql

```



## 聚合函数

## GROUP BY

## HAVING

## 连接查询

## 嵌套查询

