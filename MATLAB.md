# MATLAB 入门

MATLAB 作为线型系统的一种分析和仿真工具，是理工科大学生应该掌握的技术工具，它作为一种编程语言和可视化工具，可解决**工程、科学计算**和**数学学科**中许多问题

MATLAB 建立在**向量、数组**和**矩阵**的基础上，使用方便，人机界面直观，输出结果可视化

**矩阵**是 MATLAB 的核心

## 变量与函数

变量命名规则

1. 必须是不含空格的单个词

2. 区分大小写

3. 最多不超过19个字符

4. 必须以字母打头，之后可以是任意字母、数字或下划线

5. 不允许使用标点符号

**特殊变量表**

| 特殊变量    | 取值                      |
| ------- | ----------------------- |
| ans     | 用于结果的缺省变量名              |
| pi      | 圆周率                     |
| eps     | 计算机的最小数，当和1相加就产生一个比1大的数 |
| flops   | 浮点运算数                   |
| inf     | 无穷大，如1/0                |
| NaN     | 不定量，如0/0                |
| i, j    | $i = j = \sqrt{-1}$     |
| realmin | 最小可用正实数                 |
| realmax | 最大可用正实数                 |

**数学运算符号**

| 运算符号 | 说明  |
| ---- | --- |
| +    | 加法  |
| -    | 减法  |
| *    | 乘法  |
| .*   | 点乘  |
| /    | 除法  |
| ./   | 点除  |
| ^    | 乘幂  |
| .^   | 点乘幂 |
| \    | 左除  |

- 乘法与点乘区别
  
  乘法  按照线性代数矩阵相乘
  
  点乘  矩阵对应元素相乘

- 除法与点除区别
  
  同上

- 点乘幂
  
  $$
  a = 
\begin{bmatrix}
1 & 2 & 3
\end{bmatrix}
；b = 
\begin{bmatrix}
2 & 4 & 6
\end{bmatrix}
\\
a .\^\ b = 
\begin{bmatrix} 
1^2 & 2^4 & 2^6
\end{bmatrix}
  $$

- 左除与右除
  
  ![](D:\HANSHAN\Mathmatical%20Modeling\Picture\屏幕截图%202022-06-05%20132603.png)

**标点符号**

- MATLAB 每条命令后，若为**逗号**或**无标点符号**，则显示命令结果；若为**分号**，则禁止显示结果

- '%' 后面所有文字为注释

- '...' 表示续行

**数学函数**

| 函数           | 名称   | 函数             | 名称        |
| ------------ | ---- | -------------- | --------- |
| $\sin$       | 正弦函数 | $asin$         | 反正弦函数     |
| $\cos$       | 余弦函数 | $acos$         | 反余弦函数     |
| $\tan$       | 正切函数 | $atan$         | 反正切函数     |
| $abs(x)$     | 绝对值  | $\max$         | 最大值       |
| $\min$       | 最小值  | $sum$          | 元素的总和     |
| $sqrt(x)$    | 开平方  | $\exp$         | 以 e 为底的指数 |
| $\log (\ln)$ | 自然对数 | $\log_{10}(x)$ | 以10为底的对数  |
| $sign(x)$    | 符号函数 | $fix(x)$       | 取整        |

## 数组

1. **创建简单的数组**
   
   ```matlab
   % 创建包含指定元素的行向量（数组）
   x = [a b c d e f]
   % 创建包含指定元素的列向量
   x = [a; b; c; d; e; f]
   % 创建从 first 开始，加1计数，到 last 结束的行向量
   x = first: last
   % 创建从 first 开始，加 increament 计数，last 结束的行向量
   x = first: increment: last
   % 创建从 first 开始，到 last 结束，有 n 个元素的行向量
   x = linspace(first, last, n)
   ```

2. **数组元素的访问**
   
   - 访问一个元素
     
     **x(i)**  访问数组 x 的第 i 个元素
   
   - 访问一块元素
     
     **x(a: b: c)**  访问数组 x 的从第 a 个元素开始，以步长为 b 到 第 c 个元素（但不超过 c ），b 可以为负数，b 缺损时为1
   
   - 直接使用元素编址序号
     
     **x([a b c d])**  提取数组 x 的第 a, b, c, d 个元素构成一个新的数组
     
     [x(a) x(b) x(c) x(d)]

3. **数组的方向**
   
   即为行列向量之分
   
   列向量：`x = [a; b; c; d; e; f]`
   
   转置：`c = b'`

4. **数组的运算**
   
   1）标量运算
   
   ![](D:\HANSHAN\Mathmatical%20Modeling\Picture\屏幕截图%202022-06-05%20141809.png)
   
   2）数组 - 数组运算
   
   两个数组有**相同**维数时，加、减、乘、除、幂运算可按**元素对元素**方式进行的，不同大小或维数的数组是不能进行运算的
   
   ![](D:\HANSHAN\Mathmatical%20Modeling\Picture\屏幕截图%202022-06-05%20143242.png)

## 矩阵

1. **矩阵的建立**
   
   **逗号**或**空格**用于分隔某一行的元素，**分号**用于区分不同的行；除了分号，在输入矩阵时，enter 键也表示开始一新行；输入矩阵时，严格要求所有行有相同的列
   
   - 特殊矩阵的建立
     
     - 空矩阵
       
       `a = []`
       
       产生一个空矩阵，当对一项操作无结果时，返回空矩阵，空矩阵的大小为零
     
     - 零矩阵
       
       `b = zeros(m, n)`
       
       产生一个 m 行、n 列的零矩阵
     
     - 1矩阵
       
       `c = ones(m, n)`
       
       产生一个 m 行、n 列的1矩阵
     
     - 单位矩阵
       
       `d = eye(m, n)`
       
       产生一个 m 行、n 列的单位矩阵

2. **矩阵中元素的操作**
   
   - 矩阵 A 的第 r 行：`A(r, :)`
   
   - 矩阵 A 的第 r 列：`A(:, r)`
   
   - 依次提取矩阵 A 的每一列，将 A 拉伸为一个列向量：
     
     `A(: )`
   
   - 提取矩阵 A 的第 i_1~i_2 行、第 j_1~j_2 列构成新矩阵：
     
     `A(i_1: i_2, j_1, j_2)`
   
   - 以逆序提取矩阵 A 的第 i_1~i_2 行，构成新矩阵：
     
     `A(i_2: -1: i_1, :)`
   
   - 以逆序提取矩阵 A 的第 j_1~j_2 列，构成新矩阵：
     
     `A(:, j_2: -1: j_1)`
   
   - 删除 A 的第 i_1~i_2 行，构成新矩阵：
     
     `A(i_1: i_2, :) = []`
   
   - 删除 A 的第 j_1~j_2列，构成新矩阵：
     
     `A(:, j_1: j_2) = []`
   
   - 将矩阵 A 与 B 拼接成新矩阵：
     
     `[A B]`; `[A; B]`

3. **矩阵的运算**
   
   - 标量 - 矩阵运算
     
     同 标量 - 数组 运算
   
   - 矩阵 - 矩阵运算
     
     - 元素对元素的运算，同数组 - 数组运算
     
     - 矩阵运算
       
       矩阵加法：A + B
       
       矩阵乘法：A * B
       
       方阵的行列式：det(A)
       
       方阵的逆：inv(A)
       
       方阵的特征值与特征向量：[V, D] = eig[A]

4. **矩阵的形态**
   
   `A = [1: 3; 4: 6]

## MATLAB 编程

命令行窗口

清空命令行  clc

清空变量名  clear

### M 脚本文件

新建脚本 -- ctrl s 保存 -- 运行 or 【命令行窗口输入 **文件名** 】

### M 函数文件

这类文件的**第一行必须是一特殊字符 function 开始**

格式为 【function 因变量=函数名（自变量名）】

***M 文件名必须与函数名一致***

> 定义函数 $f(x_1, x_2) = 100(x_2 - x_1^2)^2 + (1 - x_1)^2$

```matlab
function f = fun(x)
f = 100*(x(2)-x(1)^2)^2+(1-x(1))^2

% 调用函数
x = [1 2]
fun(x)
```

### 关系与逻辑运算

1. 关系操作符
   
   | 关系操作符 | 说明   |
   | ----- | ---- |
   | <     | 小于   |
   | <=    | 小于等于 |
   | >     | 大于   |
   | >=    | 大于等于 |
   | ==    | 等于   |
   | ~=    | 不等于  |

2. 逻辑运算符
   
   | 逻辑操作符 | 说明  |
   | ----- | --- |
   | &     | 与   |
   | \|    | 或   |
   | ~     | 非   |

### Format 命令

![](D:\HANSHAN\Mathmatical%20Modeling\Picture\屏幕截图%202022-06-11%20230828.png)

- `disp()`
  
  输出变量的的值

- `%d`
  
  输出整型；`%3d` 就是按照长度为3的整型输出，e.g. 10，就是 '_10'，‘——### 控制流 代表空格

- `%f`
  
  输出小数；`%6.2f` 就是小数点后保留2位，输出总长度为6

- `%c`
  
  输出字符

- `%s`
  
  输出字符串

MATLAB 提供多种决策或控制流结构

**for** 循环、**while** 循环、**if - else - end** 结构

1. **for 循环**
   
   允许一组命令以固定和预定的次数重复
   
   ```matlab
   for x = array
       {commands}
   end
   ```
   
   在 for 和 end 语句之间的命令串 {comands} 按数组（array）中的每一列执行一次；在每一次迭代中，x 被指定为数组的下一列，即在第 n 次循环中，x = array(:, n)
   
   > 例
   > 
   > 对 $n = 1, 2, \dots, 10$; 求下式的值
   > 
   > $$
   > x_n = \sin \frac{n \cdot \pi}{10}
   > $$
   > 
   > ```matlab
   > for n = 1: 10
   >     x(n) = sin(n*pi/10);
   > end
   > x
   > ```

2. **while 循环**
   
   与 for 循环以固定次数求一组命令相反，while 循环以不定的次数求一组语句的值
   
   ```matlab
   while expression
       {command}
   end
   ```
   
   只要在 expression 里的所有元素为真，就执行 while 和 end 语句之间的命令串 {command}
   
   > 例
   > 
   > 设银行年利率为11.25%; 将10000元存入银行，问多长时间会连本带利翻一番？
   > 
   > ```matlab
   > money = 10000;
   > years = 0;
   > while money < 20000
   >     years = years+1
   >     money = money*(1+11.25/100)
   > end
   > ```

3. **if - else - end 结构**
   
   - 一个选择的一般形式
     
     ```matlab
     if expression
         {commands}
     end
     ```
     
     若在 expression 里的所有元素为真，就执行 if 和 end 语句之间的 {commands}
     
     > 例  设
     > 
     > $$
     > f(x) = \begin{cases}
x^2 + 1 \quad x > 1 \\
2x \quad x \le 1
\end{cases}
     > $$
     > 
     > 求 $f(2), f(-1)$
     > 
     > *先新建 M 文件 fun1.m 定义函数 f(x)，再在 MATLAB 命令窗口输入 fun1(2), fun1(-1) 即可*
     > 
     > ```matlab
     > function f = fun1(x)
     > if x > 1
     >     f = x^2+1
     > end
     > 
     > if x <= 1
     >     f = 2*x
     > end
     > 
     > % 调用
     > fun1(2)
     > fun1(-1)
     > ```
   
   - 三个或更多选择的一般形式
     
     ```matlab
     if expression1
         {commands1}
     elseif expression2
         {commands2}
     elseif expression3
         {commands3}
     elseif ...
     ...
     else
         {commands}
     end
     end
     end
     ...
     end
     ```
     
     > 例  设
     > 
     > $$
     > f(x) = \begin{cases}
x^2+1 \quad x > 1 \\
2x \quad 0 < x \le 1 \\
x^3 \quad x \le 0
\end{cases}
     > $$
     > 
     > 求 $f(2), f(0.5), f(-1)$
     > 
     > ```matlab
     > function f = fun2(x)
     > if x > 1
     >     f = x^2+1
     > elseif x <= 0
     >     f = x^3
     >    else
     >     f = 2*x
     >    end
     > end
     > 
     > % 调用
     > fun2(2)
     > fun2(0.5)
     > fun2(-1)
     > ```

4. **switch 结构**
   
   ```matlab
   switch expression
   case value1
       statement1
   case value2
       statement2
   ...
   otherwise
       statement
   end
   ```
   
   > 例
   > 
   > ```matlab
   > input_num = 1;
   > switch input_num
   > case -1
   >     disp('negative 1');
   > case 0
   >     disp('zero');
   > case 1
   >     disp('positive 1');
   > otherwise
   >     disp('other value');
   > end
   > ```

# 绘图

MATLAB 作图是通过描点、连线来实现的，故在画一个图形之前，必须先取得该图形上的一系列的点的坐标，然后将该点集的坐标传给 MATLAB 函数画图

## 二维图形

1. 曲线图
   
   `plot(x, y, s)`
   
   - x, y: 向量，分别表示点集的横纵坐标
   
   - s: 线型
     
     ![](D:\HANSHAN\Mathmatical%20Modeling\Picture\屏幕截图%202022-06-08%20212129.png)
   
   `plot(x, y)` ---- 画实线
   
   `plot(x, y1, s1, x, y2, s2, ..., x, yn, sn)` ---- 将多条线画在一起
   
   > 例
   > 
   > 在 [0, 2*pi] 用红线画 sin(x), 用绿圈画 cos(x)
   > 
   > ```matlab
   > x = linspace(0, 2*pi, 30);
   > y = sin(x);
   > z = cos(x);
   > plot(x, y, 'r', x, z, 'bo')
   > ```
   > 
   > Out:
   > 
   > ![](D:\HANSHAN\Mathmatical%20Modeling\Picture\屏幕截图%202022-06-08%20212842.png)

2. 符号函数（显函数、隐函数和参数方程）画图
   
   - ezplot
     
     - 在 a < x < b 绘制显函数 f = f(x) 的函数图
       
       `ezplot('f(x)', [a, b])`
     
     - 在区间 xmin < x < xmax 和 ymin < y < ymax 绘制隐函数 f(x, y) = 0 的函数图
       
       `ezplot('f(x, y)', [xmin, xmax, ymin, ymax])`
     
     - 在区间 tmin < t < tmax 绘制参数方程 x = x(t), y = y(t) 的函数图
       
       `ezplot('x(t)', 'y(t)', [tmin, tmax])`
     
     > 例
     > 
     > 在 [0, pi] 上画 y = cos(x) 的图形
     > 
     > `ezplot('cos(x)', [0, pi])`
     > 
     > 在 [0, 2*pi] 上画 x = cos^3 t, y = sin^3 t 星形图
     > 
     > `ezplot('cos(t)^3', 'sin(t)^3', [0, 2*pi])`
     > 
     > 在 [-2, 0.5], [0, 2] 上画隐函数 e^x + sin(xy) = 0 的图形
     > 
     > `ezplot('exp(x) + sin(x*y)', [-2, 0.5, 0, 2])`
   
   - fplot
     
     绘制字符串 fun 指定的函数在 lims = [xmin, xmax] 的图形
     
     `fplot('fun', lims)`
     
     > 注意：
     > 
     > - fun 必须是 M 文件的函数名或是独立变量为 x 的字符串
     > 
     > - fplot 函数不能画参数方程和隐函数图形，但在一个图上可以画多个图形
     
     ————
     
     > 例
     > 
     > 在 [-1, 2] 上画 y = e^{2x} + sin(3x^2) 的图形
     > 
     > ```matlab
     > % 老版本用法
     > function Y = myfun1(x)
     > Y = exp(2*x) + sin(3*x^2)
     > 
     > fplot('myfun1', [-1, 2])
     > 
     > % 新版本用法
     > % fplot(@(x) fun)
     > fplot(@(x) exp(2*x) + sin(3*x^2), [-1, 2])
     > ```
     > 
     > 1
     > 
     > 在 [-2, 2] 范围内绘制函数 tanh 的图形
     > 
     > `fplot('tanh', [-2, 2])`
     > 
     > x, y 的取值范围都在 [-2*pi, 2\*pi], 画函数 tanh(x), sin(x), cos(x) 的图形
     > 
     > ```matlab
     > % method_1
     > fplot('[tanh(x), sin(x), cos(x)]', 2*pi\*[-1 1 -1 1 -1 1])
     > 
     > % method_2
     > fplot(@(x) tanh(x), 2*pi*[-1 1], 'r')
     > hold on
     > fplot(@(x) sin(x), 2*pi*[-1 1], 'g')
     > hold on
     > fplot(@(x) cos(x), 2*pi*[-1 1], 'b')
     > hold off
     > ```

3. 极坐标图
   
   用角度 theta (弧度表示) 和极半径 rho 作极坐标图，用 s表示指定线型
   
   `polar(theta, rho, s)`
   
   > 例
   > 
   > 画出 r = sin 2\theta cos 2\theta 的极坐标图
   > 
   > ```matlab
   > theta = linspace(0, 2*pi);
   > rho = sin(2*theta).*cos(2*theta);
   > polar(theta, tho, 'g');
   > title('Polar plot of sin(2*theta).*cos(2*theta)')'
   > ```
   > 
   > Out:
   > 
   > ![](D:\HANSHAN\Mathmatical%20Modeling\Picture\屏幕截图%202022-06-08%20221324.png)

4. 双轴图
   
   `plotyy()`
   
   ```matlab
   x = 0: 0.01: 20;
   y1 = 200*exp(-0.05*x) .* sin(x);
   y2 = 0.8*exp(-0.5*x) .* sin(10*x);
   [AX, H1, H2] = plotyy(x, y1, x, y2);
   set(get(AX(1), 'Ylabel'), 'String', 'Left Y-axis')
   set(get(AX(2), 'Ylabel'), 'String', 'Right Y-axis')
   title('Labeling plotyy')
   set(H1, 'LineStyle', '--'); set(H2, 'LineStyle', ':');
   ```
   
   Out:
   
   ![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-17%20112919.png)
   
   直方图
   
   `hist()`
   
   ```matlab
   y = randn(1, 1000);
   subplot(2, 1, 1);
   hist(y, 10);
   title('Bins = 10');
   subplot(2, 1, 2);
   hist(y, 50);
   title('Bins = 50');
   ```
   
   Out:
   
   ![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-17%20113006.png)
   
   ![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-17%20113035.png)

5. 柱状图
   
   `bar()`
   
   ```matlab
   x = [1 2 5 4 8]; y = [x; 1: 5];
   subplot(1, 3, 1); bar(x); title('A bargraph of vector x');
   subplot(1, 3, 2); bar(y); title('A bargraph of vector y');
   subplot(1, 3, 3); bar3(y); title('A 3D bargraph');
   ```
   
   Out:
   
   ![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-17%20113428.png)
   
   ```matlab
   x = [1 2 5 4 8]; y = [x; 1: 5];
   subplot(1, 2, 1);
   bar(y, 'stacked');
   title('Stacked');
   
   subplot(1, 2, 2);
   barh(y);
   title('Horizontal');
   ```
   
   Out:
   
   ![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-17%20113923.png)

6. 饼图
   
   `pie()`
   
   ```matlab
   a = [10 5 20 30];
   subplot(1, 3, 1); pie(a);
   subplot(1, 3, 2); pie(a, [0, 0, 0, 1]);
   subplot(1, 3, 3); pie3(a, [0, 0, 0, 1]);
   ```
   
   Out:
   
   ![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-17%20114159.png)

## 三维图形

1. 空间曲线
   
   - 一条曲线
     
     `plot3(x, y, z, s)`
     
     - x, y, z: n 维向量，分别表示曲线上点集的横纵坐标、函数值
     
     - s: 线型，指定颜色等
     
     > 在区间 [0, 10*pi] 画出参数曲线 x = sin(t), y = cos(t), z = t
     > 
     > ```matlab
     > t = 0: pi/50: 10*pi;
     > plot3(sin(t), cos(t), t)
     > rotate3d  % 旋转
     > ```
   
   - 多条曲线
     
     `plot3(x, y, z)`
     
     - x, y, z: 都是 m*n 矩阵，其对应的每一列表示一条曲线
     
     > 例
     > 
     > 画多条曲线观察函数 Z ;= (X + Y) .^ 2
     > 
     > ```matlab
     > x = -3: 0.1 : 3; y = 1: 0.1: 5;
     > [X, Y] = meshgrid(x, y);
     > % meshgrid(x, y) 是产生一个以向量 x 为行、向量 y 为列的矩阵
     > Z = (X + Y) .^ 2;
     > plot3(X, Y, Z)
     > ```

2. 空间曲面
   
   - 空间曲面
     
     `surf(x, y, z)`
     
     > 例
     > 
     > 画多条曲线观察函数 Z = (X + Y) .^ 2
     > 
     > ```matlab
     > x = -3: 0.1 : 3; y = 1: 0.1: 5;
     > [X, Y] = meshgrid(x, y);
     > % meshgrid(x, y) 是产生一个以向量 x 为行、向量 y 为列的矩阵
     > Z = (X + Y) .^ 2;
     > surf(X, Y, Z)
     > shading flat  % 将当前图形变得更加平滑
     > ```
   
   - 网格曲面
     
     `mesh(x, y, z)`
   
   - 在网格周围画一个 curtain 图（如，参考平面）
     
     `meshz(x, y, z)`
     
     > 例
     > 
     > 绘 peaks 的网格图
     > 
     > ```matlab
     > [X, Y] = meshgrid(-3: .125: 3);
     > z = peaks(X, Y);
     > meshz(X, Y, Z)
     > ```
     > 
     > Out:
     > 
     > ![](D:\HANSHAN\Mathmatical%20Modeling\Picture\屏幕截图%202022-06-08%20231452.png)

## 图形处理

1. 加格栅、图例和标注
   
   - `grid on`
     
     加格栅在当前图上
     
     `grid off`
     
     删除格栅
   
   - 图例
     
     在当前图形的 x 轴上加图例 string
     
     `hh = xlabel(string)`
     
     在当前图形的 y 轴上加 图例 string
     
     `hh = ylabel(string)`
     
     在当前图形的 y 轴上加图例 string
     
     `hh = zlabel(string)`
     
     在当前图形的顶端上加图例 string
     
     `hh = title(string)`
   
   - 标注
     
     `hh = gtext(string)`
     
     用鼠标放置标注在现有的图上，运行命令，屏幕上出现当前图形，在图形上出现一个交叉的十字，该十字随鼠标的移动移动，当按下鼠标左键时，该标注 string 放在当前交叉的位置

2. 定制坐标
   
   `axis([xmin xmax ymin ymax zmin zmax])`
   
   - x, y, z 的最大、最小值
   
   `axis auto` ：将坐标轴返回到自动缺省值
   
   > 例
   > 
   > 在区间 [0.005, 0.01] 显示 sin(1/x) 的图形
   > 
   > ```matlab
   > x = linspace(0.0001, 0.01, 1000);
   > y = sin(1 ./ x);
   > plot(x, y)
   > axis([0.005 0.01 -1 1])
   > ```

3. 图形保持
   
   - `hold on` ：保持当前图形，以便继续画图到当前图上
     
     `hold off` ：释放当前图形窗口
   
   - 不覆盖上一个图形窗口
     
     `figure(h)` ：新建 h 窗口，激活图形使其可见，并把它置于其它图形之上
     
     > 例
     > 
     > 区间 [0, 2*pi] 新建两个窗口分别画出 y = sin(x), z = cos(x)
     > 
     > ```matlab
     > x = linspace(0, 2*pi, 100);
     > y = sin(x); z = cos(x);
     > plot(x, y);
     > title('sin(x)');
     > pause
     > figure(2);
     > plot(x, z);
     > title('cos(x)');
     > ```

4. 分割窗口
   
   `h = subplot(mrows, ncols, thisplot)`
   
   划分整个作图区域为 mrows*ncols 块（逐行对块访问）并激活第 thisplot 块，其后的作图语句将图形画在该块上
   
   > 例
   > 
   > 将屏幕分割为四块，并分别画出 y = sin(x), z = cos(x), a = sin(x)*cos(x), b = sin(x)/cos(x)
   > 
   > ```matlab
   > x = linspace(0, 2*pi, 100);
   > y = sin(x); z = cos(x);
   > a = sin(x) .* cos(x);
   > b = sin(x) ./ (cos(x) + eps); % 为了保证分母不为零 加一个计算机最小的一个数 eps
   > subplot(2, 2, 1); plot(x, y), title('sin(x)')
   > subplot(2, 2, 2); plot(x, z), title('cos(x)')
   > subplot(2, 2, 3); plot(x, a), title('sin(x)cos(x)')
   > subplot(2, 2, 4); plot(x, b), title('sin(x)/cos(x)')
   > ```

5. 缩放图形
   
   `zoom on` ：为当前图形打开缩放模式
   
   `zoom off` ：关闭缩放模式
   
   单击鼠标左键，则在当前图形窗口中，以鼠标点中的点为中心的图形放大2倍；单击鼠标右键，则缩小2倍
   
   > 例
   > 
   > 缩放 y = sin(x) 的图形
   > 
   > ```matlab
   > x = linspace(0, 2*pi, 30);
   > y = sin(x);
   > plot(x, y)
   > zoom on
   > ```

6. 改变视角
   
   - `view(a, b)`
     
     改变视角到 (a, b)，a 是方位角，b 为仰角；缺省视角为 (-37.5, 30)
   
   - `view([x, y, z])`
     
     空间向量表示，三个量只关心它们的比例，与数值的大小无关，x 轴的 view([1, 0, 0]), y 轴的 view([0, 1, 0]), z 轴的 view([0, 0, 1])
     
     > 例
     > 
     > 画出曲面 Z = (X + Y) .^ 2 在不同视角的网格图
     > 
     > ```matlab
     > x = -3: 0.1: 3; y = 1: 0.1: 5;
     > [X, Y] = meshgrid(x, y);
     > Z = (X + Y) .^ 2;
     > subplot(2, 2, 1), mesh(X, Y, Z)
     > subplot(2, 2, 2), mesh(X, Y, Z), view(50, -34)
     > subplot(2, 2, 3), mesh(X, Y, Z), view(-60, 70)
     > subplot(2, 2, 4), mesh(X, Y, Z), view(0, 1)
     > ```
     > 
     > Out:
     > 
     > ![](D:\HANSHAN\Mathmatical%20Modeling\Picture\屏幕截图%202022-06-09%20145552.png)

7. 动画
   
   函数 moviein() 产生一个帧矩阵来存放动画中的帧
   
   函数 getframe() 对当前的图像进行快照
   
   函数 movie() 按顺序回放各帧
   
   > 例
   > 
   > 将曲面 peaks 做成动画
   > 
   > ```matlab
   > [x, y, z] = peaks(30);
   > surf(x, y, z)
   > axis([-3 3 -3 3 -10 10])
   > m-moviein(15);
   > for i = 1: 15
   >     view(-37.5 + 24*(i-1), 30)
   >     m(:, i) = getframe;
   > end
   > movie(m)
   > ```

8. 其他处理
   
   `hold on`: 在一个图上画多条曲线
   
   `hold off`: 结束在一个图上画多条曲线
   
   `legend('L1', 'L2', 'L3', ...)`
   
   `title()`
   
   `xlabel()`
   
   `ylabel()`
   
   `zlabel()`
   
   `Font`
   
   `Font size`
   
   `Line width`
   
   `Axis limit`
   
   `Tick position`
   
   `Tict label`
   
   `get(h)`
   
   `set(h, 'LineStyle', '-.', 'LineWidth', 7.0, 'Color', 'g')`

## 特殊二、三维图形

### 特殊二维图形函数

1. 极坐标图
   
   `polar(theta, rho, s)`

2. 散点图
   
   `scatter(X, Y, S, C)`
   
   ```matlab
   load seamount
   scatter(x, y, 5, z)
   ```

3. 平面等值 (高) 线图
   
   `contour(x, y, z, n)`
   
   n 个等值线的二维等值线图
   
   > 例
   > 
   > 在范围 -2 < x < 2, -2 < y < 3 内绘 z = xe^{-x^2-y^2} 的等值线图
   > 
   > ```matlab
   > [X, Y] = meshgrid(-2: .2: 2, -2: .2: 3);
   > Z = X .* exp(-X .^ 2 - Y .^ 2);
   > [C, h] = contour(X, Y, Z);
   > clabel(C, h)
   > colormap cool
   > ```
   > 
   > Out:
   > 
   > ![](D:\HANSHAN\Mathmatical%20Modeling\Picture\屏幕截图%202022-06-09%20152527.png)

### 特殊三维图形函数

1. 空间等值线图
   
   `contour3(x, y, z, n)`
   
   > 例
   > 
   > 山峰的三维和二维等值线图
   > 
   > ```matlab
   > [x, y, z] = peaks;
   > subplot(1, 2, 1)
   > contour3(x, y, z, 16, 's')
   > grid, xlabel('x - axis'), ylabel('y - axis')
   > zlabel('z-axis')
   > title('contour3 of peaks');
   > subplot(1, 2, 2)
   > contour(x, y, z, 16, 's')
   > grid, xlabel('x - axis'), ylabel('y - axis')
   > title('contour of peaks')
   > ```
   > 
   > Out:
   > 
   > ![](D:\HANSHAN\Mathmatical%20Modeling\Picture\屏幕截图%202022-06-09%20153521.png)

## 图形保存

编辑 -- 复制图窗 -- 粘贴

## 作图代码

### 二维图像

#### 曲线

color

r: 红；g: 绿；b: 蓝；c: 蓝绿；m: 紫红；y: 黄；k: 黑；w: 白

```matlab
clear; clc; close all;
x = linspace(1 ,200 , 100);  % 均匀生成数字1 ~ 2 0, 共计100个
y1 = log(x) + 1;  % 生成函数 y = log(x) + 1
y2 = log(x) + 2;  % 生成函数 y = log(x) + 2
figure;
plot(x, y1);  % 作图 y = log(x) + 1
hold on  % 多图共存在一个窗口上
plot(x , y2, 'LineWidth', 2);  % 作图 y = log(x) + 2，LineWidth 指线型的宽度，粗细尺寸2
hold off  % 关闭多图共存在一个窗口上
legend('y1', 'y2');  % 生成图例 y1 和 y2
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-08-23%20160020.png)

#### 散点图

常用来比较理论数据和实验数据的趋势关系

```matlab
figure;
y3 = y1 + rand(1, 100) - 0.5;
plot(x, y1, 'LineWidth', 2, 'Color', [0.21, 0.21, 0.67]);
hold on
% 设置数据点的形状，数据点的填充颜色，数据点的轮廓颜色
plot(x, y3, 'o', 'LoneWidth', 2, 'Color', [0.46, 0.63, 0.90], 'MarkerFaceColor', [0.35, 0.90, 0.89], 'MarkerEdgeColor', [0.18, 0.62, 0.17]);
hold off
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-08-23%20160836.png)

#### 渐变图

用不同的颜色，数据点大小表征不同数值，更加直观

```matlab
x = linspace(0, 3*pi, 200);
y = cos(x) + rand(1, 200);  % 随机生成1*200，位于 [0. 1] 的数字
sz = 25;  % 尺寸为25
c = linspace(1, 10, length(x));
scatter(x, y, sz, c, 'filled')
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-08-23%20161149.png)

#### 条形图

```matlab
A = [60.689; 87.714; 143.1; 267.9515];
C = [127.5; 160.4; 231.9; 400.2];
B = C - A;
D = [A, B, C];
bar1 = bar([2: 5: 17], A, 'BarWidth', 0.2, 'FaceColor', 'k');
hold on;
bar2 = bar([3: 5: 18], B, 'BarWidth', 0.2, 'FaceColor', [0.5, 0.5, 0.5]);
hold on;
bar3 = bar([4: 5: 19], C, 'BarWidth', 0.2, 'FaceColor', 'w');
ylabel('耗时/s'); xlabel('GMM阶数');
legend('训练耗时', '测试耗时', '总耗时');
labelID = {'8阶', '16阶', '32阶', '64阶'};
set(gca, 'XTick', 3: 5: 20);
set(gca, 'XTickLabel', labelID);
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-08-23%20162052.png)

#### 填充图

```matlab
x = 0.4: 0.1: 2*pi;
y1 = sin(2*x);
y2 = sin(x);
% 确定 y1, y2 的上下边界
maxY = max([y1: y2);
minY = min([y1: y2]);
% 确定填充多边形，按照顺时针方向来确定点
% figure 实现左右翻转
xFill = [x, fliplr(x)];
yFill = [maxY, fliplr(minY)];
figure
fill(xFill, yFill, [0.21, 0.21, .0.67]);
hold on;
% 绘制轮廓线
plot(x, y1, 'k', 'LineWidth', 2)
plot(x, y2, 'k', 'LineWidth', 2)
hold off
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-08-23%20162812.png)

#### 双轴图

```matlab
figure;
load('accidents.mat', 'hwydata')
ind = 1: 51;
drivers = hwydata(:, 5);
yyaxis left
scatter(ind, drivers, 'LineWidth', 2);
title('Highway Data');
xlabel('States');
ylabel('Licensed Drivers (thousands)');
pop = hwydata(:, 7);
yyasis right
scatter(ind, pop, 'LineWidth', 2);
ylabel('Vehicle Miles Traveled (millions)';
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-08-23%20163428.png)

#### 场图

```matlab
% 直接把 streamline 函数的帮助文档 demo 拷贝
[x, y] = meshgrid(0: 0.1: 1, 0: 0.1: 1);
u = x;
v = -y;
startx = 0.1: 0.1: 0.9;
starty = ones(size(startx));
% 需要获取所有流线的属性
figure;
% quiver 函数使用箭头来直观的显示矢量场
quiver(x, y, u, v);
streamline(x, y, u, v, startx, starty);
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-08-23%20163928.png)

#### 等高线图

```matlab
figure;
[X, Y, Z] = peaks;
subplot(2, 2, 1);
contour(X, Y, Z, 20, 'LineWidth', 2);  % 20组等高线
subplot(2, 2, 2);
contour(X, Y, Z, '--', 'LineWidth', 2);
subplot(2, 2, 3);
v = [1, 1];  % Z 为1的等高线提取出来
contour(X, Y, Z, v, 'LineWidth', 2);
x = -2: 0.2: 2;
y = -2: 0.2: 3;
[X, Y] = meshgrid(x, y);
Z = X .* exp(-X .^ 2 - Y .^ 2);
subplot(2, 2, 4);
contour(X, Y, Z, 'ShowText', 'on', 'LineWidth', 2);
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-08-23%20171226.png)

#### 等高线填充图

```matlab
figure;
subplot(2, 2, 1);
[X, Y, Z] = peaks(50);
contourf(X, Y, Z);
subplot(2, 2, 2);
contourf(X, Y, Z, '--');
% 限制范围
subplot(2, 2, 3);
contourf(X, Y, Z, [2, 3], 'ShowText', 'on');
subplot(2, 2, 4);
contourf(X, Y, Z, [2, 2]);
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-08-23%20172240.png)

### 三维图像

#### 曲线

```matlab
figure;
t = 0: pi/20: 10*pi;
xt = sin(t);
yt = cos(t);
plot3(xt, yt, t, '-o', 'Color', 'b', 'MarkerSize', 10);
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-08-23%20164327.png)

```matlab
figure;
x = -20: 10: 20;
y = 0: 100;
% 随便生成5组数据，也就是目标图上的五条曲线数据
z = zeros(5, 101);
z(1, 1: 10: end) = linspace(1, 10, 11);
z(2, 1: 10: end) = linspace(1, 20, 11);
z(3, 1: 10: end) = linspace(1, 5, 11);
z(4, 5: 10: end) = linspace(1, 10, 10);
z(5, 80: 2: end) = linspace(1, 5, 11);
for i = 1: 5
    % x 方向每条曲线都是一个值，重复 y 的长度这么多次
    xx = x(i)*ones(1, 101);
    % z 方向的值，每次取一条
    zz = z(i, :);
    % plot3 在 xyz 空间绘制曲线，保证 xyz 长度一致即可
    plot3(xx, y, zz, 'LineWidth', 2)
    hold on
end
hold off
legend('line1', 'line2', 'line3', 'line4', 'line5');
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-08-23%20165004.png)

#### 散点图

```matlab
figure;
[X, Y, Z] = sphere(16);
x = [0.5*X(:); 0.75*X(:); X(:)];
y = [0.5*Y(:); 0.75*Y(:); Y(:)];
Z = [0.5*Z(:); 0.75*Z(:); Z(:)];
S = repmat([70, 50, 20], numel(X), 1);
C = repmat([1, 2, 3], numel(X), 1);
s = S(:);
c = C(:);
h = scatter3(x, y, z, s, c);
h.MarkerFaceColor = [0, 0.5, 0.5];
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-08-23%20165403.png)

```matlab
x = linspace(1, 200, 100);
y1 = log(x) + 1;
y2 = log(x) + 2;
y3 = y1 + rand(1, 100) - 0.5;
figure;
scatter3(x, y2, y3, x, x, 'filled');
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-08-23%20165724.png)

#### 伪彩图

```matlab
[x, y, z] = peaks(30);
figure;
plot1 = subplot(1, 2, 1);
surf(x, y, z);
% 获取第一幅图的 colormap，默认为 parula
plot2 = subplot(1, 2, 2);
surf(x, y, z);
% 下面设置的是第二幅图的颜色
colormap(hot);
% 设置第一幅图颜色显示为 parula
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-08-23%20170155.png)

#### 裁剪伪彩图

```matlab
figure;
n = 300;
[x, y, z] = peaks(n);
subplot(2, 2, [1, 3])
surf(x, y, z);
shading interp  % 渲染
view(0, 90)
for i = 1: n
    for j = 1: n
        if x(i, j)^2 + 2*y(i, j)^2 > 6 && 2*x(i, j)^2 + y(i, j)^2 < 6
            z(i, j) = NaN;
        end
    end
end
subplot(2, 2, 2)
surf(x, y, z)
shading interp
view(0, 90)
subplot(2, 2, 4)
surf(x, y, z);
shadig interp
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-08-23%20170600.png)

#### 等高线

```matlab
figure('Position', [0, 0, 900, 400]);
subplot(1, 3, 1);
[X, Y, Z] = sphere(50);
contour3(X, Y, Z, 'LineWidth', 2);
[X, Y] = meshgrid(-2: 0.25: 2);
Z = X .* exp(-X .^ 2 - Y .^ 2);
subplot(1, 3, 2);
contour3(X, Y, Z, [-0.2, -0.1, 0.1, 0.2], 'ShowText', 'on', 'LineWidth', 2)
[X, Y, Z] = peaks;
subplot(1, 3, 3);
contour3(X, Y, Z, [2, 2], 'LineWidth', 2);
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-08-23%20171918.png)

#### 矢量场图

```matlab
figure;
[X, Y, Z] = peaks(30);
% 矢量场，曲面法线
[U, V, W] = surfnorm(X, Y, Z);
% 箭头长度，颜色
quiver3(X, Y, Z, U, V, W, 0.5, 'r);
hold on
surf(X, Y, Z);
xlim([-3, 3]);
ylim([-3, 3.2]);
shading interp
hold off
view(0, 90);
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-08-23%20172707.png)

#### 分子模型图

```matlab
% 球面的坐标信息，为了看起来平滑一点，给到100
[x, y, z] = sphere(100);
% C 大小
C = 10;
% H 大小
H = 5;
figure;
% 大球
surf(C*x, C*y, C*z, 'FaceColor', 'red', 'EdgeColor', 'none');
hold on
% 四个小球，都偏离一点位置，准确的位置需要计算，这里演示一个大概位置
```

### 组合图

#### 为彩图 + 投影图

```matlab
x = linspace(-3, 3, 30);
y = linspace(-4, 4, 40);
[X, Y] = meshgrid(x, y);
Z = peaks(X, Y);
z1 = max(Z);
z2 = max(Z, [], 2);
figure;
subplot(3, 3, [1, 2]);
plot(x, z1, 'LineWidth', 2);
subplot(3, 3, [6, 9]);
plot(z2, y, 'LineWidth', 2);
subplot(3, 3, [4, 5, 7, 8]);
surf(x, y, Z);
xlim([-3, 3]);
ylim([-4, 4]);
view(0, 90);
shading interp
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-08-23%20173224.png)

# 结构体 (Structure)

一种存储异构数据的方法，structures 包含称为字段的数组

```matlab
% e.g.
student.name = 'John Doe';
student.id = 'jdo2@sfu.ca';
student.number = 301073268;
student.grade = [100, 75, 73; ...
95, 91, 85.5; ...
100, 98, 72];
student
```

Out:

<img title="" src="file:///D:/HANSHAN/Mathmatical Modeling/Picture_MATLAB/屏幕截图%202022-07-16%20173447.png" alt="" width="251">

```matlab
% 第二个同学
student(2).name = 'Ann Lane';
student(2).id = 'aln4@sfu.ca';
student(2).number = 301078853;
student(2).grade [95 100 90; 95 82 97; 100 85 100];
student
```

Out:

<img title="" src="file:///D:/HANSHAN/Mathmatical Modeling/Picture_MATLAB/屏幕截图%202022-07-16%20173848.png" alt="" width="294">

**Structure Function**

| Function    | Description     |
| ----------- | --------------- |
| cell2struct | 将单元阵列转换为结构阵列    |
| fieldnames  | 结构的字段名或对象的公共字段  |
| getfield    | 结构数组字段          |
| isfield     | 确认输入是否为结构数组字段   |
| isstruct    | 确认输入是否为结构数组     |
| orderfields | 结构数组的序域         |
| rmfield     | 从结构中删除字段        |
| setfield    | 为结构数组字段赋值       |
| struct      | 创建结构数组          |
| struct2cell | 将结构转换为单元阵列      |
| structfun   | 将函数应用于标量结构的每个字段 |

## 嵌套结构体 (Nesting Structures)

```matlab
A = struct('data', [3 4 7; 8 0 1], 'nest', ...
struct('testnum', 'Test 1', ...
'xdata', [4 2 8], 'ydata', [7 1 6]));
A(2).data = 9 3 2; 7 6 5];
A(2).nest.testnum = 'Test 2';
A(2).nest.xdata = [3 4 2];
A(2).nest.ydata = [5 0 9];
A.nest
```

# 单元格阵列 (Cell Array)

另一种存储异构数据的方法，与矩阵类似，但每个条目包含不同类型的数据

```matlab
A(1, 1) = {[1 4 3; 0 5 8 ; 7 2 9]};
A(1, 2) = {'Anne Smith'};
A(2, 1) = {3 + 7i};
A(2, 2) = {-pi: pi: pi};
A
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-16%20180827.png)

```matlab
A{(1, 1)} = [1 4 3; 0 5 8 ; 7 2 9];
A{(1, 2)} = 'Anne Smith';
A{(2, 1)} = 3 + 7i;
A{(2, 2)} = -pi: pi: pi;
A
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-16%20180827.png)

**Cell Array Function**

| Function    | Description         |
| ----------- | ------------------- |
| cell        | 创建单元格阵列             |
| cell2mat    | 将单元格数组转换为数字数组       |
| cell2struct | 将单元格阵列转换为结构阵列       |
| celldisp    | 单元格数组内容             |
| cellfun     | 将函数应用于单元阵列中的每个单元    |
| cellplot    | 单元阵列的图形显示结构         |
| cellstr     | 从字符数组创建字符串的单元格数组    |
| iscell      | 确认输入是否为单元格数组        |
| mat2cell    | 将阵列转换为具有不同大小单元的单元阵列 |
| num2cell    | 将阵列转换为大小一致的单元阵列     |
| struct2cell | 将结构转换为单元阵列          |

## 读取单元格阵列数据

{} 用于访问单元格数组的内容

## 改变形态

```matlab
A = {'James Bond', [1 2; 3 4 ; 5 6]; pi, magic(5)};
C = reshape(A, 1, 4)
```

# 档案存取

1. `save()` and `load()`
   
   ```matlab
   clear; a = magic(4);
   % 普通文字软件不能查看
   save mydata1.mat
   % 普通文字文件可以查看
   save mydata2.mat -ascii
   ```
   
   ```matlab
   load('mydata1.mat')
   load('mydata2.mat', '-ascii')
   ```

2. `xlsread()` and `xlswrite()`
   
   ![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-17%20000138.png)
   
   ```matlab
   score = xlsread('04score.slsx')
   score = xlsread('04score.xlsx', 'B2:D4')
   ```
   
   ```matlab
   M = mean(score')';
   xlswrite('04score.xlsx', M, 1, 'E2:E4');
   xlswrite('04score.xlsx', {'Mean'}, 1, 'E1');
   ```
   
   ```matlab
   % 读取文字部分
   [score header] = xlsread('04score.xlsx')
   ```

| Function | Description       |
| -------- | ----------------- |
| fopen    | 打开文件，或获取有关打开文件的信息 |
| fclose   | 关闭一个或全部打开         |
| fscanf   | 从文本文件读取数据         |
| fprintf  | 将数据写入文本文件         |
| feof     | 文件结束测试            |

# 数值微积分

## 多项式微分与积分

### 微分

对函数 $f(x)$ 做微分：

$$
f'(x) \ \ or  \ \ \frac{df(x)}{dx}
$$

某个点的切线向量的变化量 

一个多项式

$$
f(x) = a_n x^n + a_{n-1}x^{n-1} + ... + a_1x + a_0
$$

微分即对 $f(x)$ 求导

**MATLAB**

> 例
> 
> $$
> f(x) = x^3 -2x -5
> $$
> 
> `p = [1 0 -2 -5];` 

> 例
> 
> $$
> f(x) = 9x^3 - 5x^2 + 3x +7 \quad -2 \le x \le 5
> $$
> 
> ```matlab
> a = [9, -5, 3, 7]; x = -2: 0.01: 5;
> f = polyval(a, x);
> plot(x, f, 'LineWidth', 2);
> xlabel('x'); ylabel('f(x)');
> set(gca, 'FontSize', 14)
> ```
> 
> Out:
> 
> ![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-17%20121150.png)
> 
> **计算微分**
> 
> ```matlab
> p = [9, -5, 3, 7];
> polyder(p)
> ```
> 
> Out:
> 
> ![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-17%20122140.png)

> Exercise
> 
> $$
> f(x) = (20x^3 - 7x^2 + 5x + 10)(4x^2 + 12x - 3) \quad -2 \le x \le  1
> $$
> 
> `conv()`
> 
> 合并
> 
> $$
> f(x) = 80x^5 + 212x^4 - 124x^3 + 121x^2 + 105x - 30 \quad -2 \le x \le 1
> $$
> 
> ```matlab
> a = [80 212 -124 121 105 -30]; x = -2: 0.01: 1;
> f = polyval(a, x);
> l1 = plot(x, f, 'LineWidth', 4, 'LineStyle', '-', 'Color', 'b')
> xlabel('x')
> ylabel('y')
> set(gca, 'FontSize', 14)
> f_ = polyder(a)
> hold on
> l2 = plot(x, f_, 'LineWidth', 4, 'LineStyle', '--', 'Color', 'r')
> hold off
> legend('f(x)', "f'(x)")
> ```
> 
> Out:
> 
> ![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-17%20151331.png)

### 积分

一个多项式

$$
f(x) = a_n x^n + a_{n-1}x^{n-1} + ... + a_1x + a_0
$$

积分即对 $f(x)$ 求原函数

**MATLAB**

> 例
> 
> $$
> f(x) = 9x^3 - 5x^2 + 3x +7 \quad -2 \le x \le 5
> $$
> 
> ```matlab
> p = [9 -5 3 7];
> polyint(p, 3)  % 积分
> % value
> polyval(polyint(p, 3), 7)
> ```
> 
> Out:
> 
> ![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-17%20151441.png)
> 
> ![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-17%20151614.png)

## 数值微分与积分

### 微分

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-17%20151754.png)

`diff()`: 计算两个数值之间的差

```matlab
x = [1 2 5 2 1];
diff(x)
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-17%20152219.png)

> 例
> 
> 点 (2, 7)，点 (1, 5) 之间的斜率
> 
> ```matlab
> x = [1 2]; y = [5 7];
> slope = diff(y) ./ diff(x)
> ```

> 例
> 
> $f(x) = \sin(x)$，找出 $x_0 = \pi /2$ 的微分，且 $h = 0.1$
> 
> ```matlab
> x0 = pi/2; h = 0.1;
> x = [x0 x0 + h];
> y = [sin(x0) sin(x0+h)];
> m = diff(y) ./ diff(x)
> ```

**计算一个区间的微分**

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-17%20153400.png)

- 创建一个区间数组 [0, 2Π]

- 步长为 h

- 计算这些点的微分

```matlab
h = 0.5;
x = 0: h: 2*pi;
y = sin(x);
m = diff(y) ./ diff(x);
```

```matlab
clc, clear, close all
h = 0.01;
x = -2: h: 4;
y = sin(x);
m = diff(y) ./ diff(x);
plot(x, y, 'k-', x(1: end - 1), m, 'bo')
```

```matlab
g = colormap(lines); hold on;
for i = 1: 4
    x = 0: power(10, -i): pi;
    y = sin(x); m = diff(y) ./ diff(x);
    plot(x(1: end - 1), m 'Color', g(i, :);
end
hold off;
set(gca, 'XLim', [0, pi/2]); set(gca, 'YLim', [0, 1.2]);
set(gca, 'FontSize', 18);  set(gca, 'FontName', 'symbol');
set(gca, 'XTick', 0: pi/4: pi/2); set(gca, 'XTickLabel', {'0', 'pi/4', 'pi/2');
h = legend('h=0.1', 'h=0.01', 'h=0.001', h=0.0001');
set(h, 'FontName', 'Times New Roman'); box on;
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-18%20021700.png)

```matlab
% -y' = y + 1; 初始条件: y(0) = 1
% 构建
syms y(t)
s = dsolve('-Dy = y + 1', 'y(0) = 1');
% 作图
t = 0: 0.1: 5;
y = eval(subs(s));
% eval 可以将字符串当作命令来执行
plot(t, y)
```

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-18%20035746.png)

`[t, Xt] = ode45(odefun, tspan, x0)`

- odefun: 函数句柄

- tspan: [t0 tfinal] 或者一系列散点 [t0, t1, ..., tf]

- x0: 初始值向量

- t: 返回列向量的时间点

- Xt: 返回对应 T 的求解列向量

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-18%20035543.png)

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-18%20035453.png)

```matlab
fun = inline('-2*y + 2*x^2 + 2*x', 'x', 'y');
[x, y] = ode23(fun, [0, 0.5], 1);
plot(x, y, 'o-')
```

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-18%20105549.png)

```matlab
% 编写函数
function fy = vdp(t, x)
fy = [x(2); 7*(1 - x(1)^2)*x(2) - x(1)];

y0 = [1; 0];
[t, x] = ode45('vdp', [0, 40, y0);
y = x(:, 1);
dy = x(:, 2);
plot(t, y, 'k', t, dy, 'b')
```

$$
x_1' = x_1 - 0.1x_1x_2 + 0.01t \\
x_2' = -x_2 + 0.02x_1x_2 + 0.044 \\
x_1(0) = 30; x_2(0) = 20
$$

```matlab
% 创建函数
function xprim = xprim3(t, x)
xprim = [x(1) - 0.1*x(1)*x(2) + 0.01*t; ...
-x(2) + 0.02*x(1)*x(2) + 0.044];

% 调用
[t, x] = ode45('xprim3', [0, 20], [30;20]);
plot(t, x);
xlabel('time t0 = 0, tt = 20')
ylabel('x values x1(0) = 30, x2(0) = 20');
```

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-18%20112559.png)

```matlab
function dx = f2(t, y)
% 初始化 dx 为两行一列的矩阵
dx = zeros(2, 1)
dx(1) = 0.04*(1-y(1)) - (1-y(2))*y(1) + 0.0001*(1 - y(2))^2;
dx(2) = -10000*y(1) + 3000*(1-y(2))*2;

[t, x] = ode45('f2', [0 100], [1 1]);
plot(t, x(:, 1), '+', t, x(:, 2), '*');
```

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-18%20114227.png)

```matlab
function dx = odefun(t, x)
dx = zeros(2, 1);
dx(1) = x(2);
dx(2) = -t*x(1) + exp(t)*y(2) + 3*sin(2*t);

tspan = [3.9 4];
y0 = [8 2];
[t, x] = ode45('odefun', tspan, y0);

plot(t, x(:, 1), '-o'), t, x(:, 2), '-*')
legend('y', 'y''')
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-18%20115141.png)

### 积分

# 求解方程式的根

```matlab
syms x
solve('x*sin(x)-x', x)

syms x
y = x*sin(x) - x;
solve(y, x)
```

$$
\begin{cases}
x - 2y = 5 \\
x + y = 6
\end{cases}
$$

```matlab
syms x y
eq1 = x - 2*y - 5;
eq2 = x + y - 6;
A = solve(eq1, eq2, x, y)
```

```matlab
% y = 4x^5
syms x
y = 4*x^5;
yprime = diff(y)
```

```matlab
% f(x) = 1.2x + 0.3 + xsin(x)
f2 = @(x)(1.2*x + 0.3 + x*sin(x));
fsolve(f2, 0)
```

```matlab
% ax^2 - b = 0
syms x a b
eq = a*x^2-b
solve(eq)
% b 与 a, x 的关
solve(eq, b)
```

**求解微分**

$y = 4x^5$

```matlab
syms x
y = 4*x^5;
yprime = diff(y)
```

**求解积分**

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-18%20011416.png)

```matlab
syms x; y = x^2*exp(x);
z = int(y);
z = z - subs(z, x, 0)
```

`fsolve()`

$f(x) = 1.2x + 0.3 + x\sin (x)$

```matlab
f = @(x) 1.2*x + 0.3 + x*sin(x);
fsolve(f, 0)
```

`fzero()`

$f(x) = x^2$

```matlab
f = @(x) x.^2
fzero(f, 0.1)
若图像未穿过 x 轴，此函数解不出来
```

`roots()`

求解多项式

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-18%20024116.png)

`roots([1 -3.5 2.75 2.125 -3.875 1.25])`

**递归函数**

$$
n! = 1 \times 2 \times 3 \times \cdots \times n
$$

```matlab
function output = fact(n)
% fact recursively finds n!
if n == 1
    output = 1;
else
    output = n*fact(n-1);
end
end
```

# 统计

- mean: 数组的平均值

- median: 数组中值

- mode: 数组中最常见的值，众数

- prctile: 数据集的百分比

- var: 方差

- std: 标准差

- skewness: 偏度；分布偏度的度量；左偏：skewness < 0; 右偏：skewness > 0
  
  ```matlab
  X = rand([10 3])*3;
  X(X(:, 1) < 0, 1) = 0; X(X(:, 3) > 0, 3) = 0;
  boxplot(X, {'Right-skewed', 'Symmetric', 'Left-skewed'});
  y = skewness(X)
  ```
  
  Out:
  
  ![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-18%20152530.png)

## 假设检验

- 确认假设检验的概率，例如0.95

- 找到 $H_0$ 的 $95\%$ 置信区间

- 检查你的分数是否在区间内

**假设检验术语**

- 置信区间 $1 - \alpha$

- 显著性水平 $\alpha$

- p 值

```matlab
load stockreturns;
x1 = stocks(:, 3);
x2 = stocks(:, 10);
boxplot([x1, x2], {'3', '10'});
[h, p] = ttest2(x1, x2)
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-18%20153932.png)

# 回归与内插

1. sum of squared erroes (SSE): 均方误差
   
   $$
   SSE = \sum_i (y_i - \hat{y})^2 \\
\quad \\
\hat{y} = \beta_0 + \beta _1 x_i \\
SSE = \sum_i (y_i - \beta_0 + \beta _1 x_i )^2
   $$
   
   需要使均方误差最小，则
   
   ![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-18%20161045.png)
   
   ![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-18%20161228.png)
   
   ```matlab
   % solve
   x = [-1.2 -0.5 0.3 0.9 1.8 2.6 3.0 3.5];
   y = [-15.6 -8.5 2.2 4.5 6.6 8.2 8.9 10.0];
   fit = polyfit(x, y, 1);
   % 查看拟合后每个点
   xfit = [x(1): 0.1: x(end)];
   yfit = fit(1)*xfit + fit(2);
   plot(x, y, 'ro', xfit, yfit);
   set(gca, 'FontSize', 14);
   legend(2, 'data points', 'best-fit');
   ```

2. 判断是否线性相关
   
   通过使用
   
   - `scatter()`
   
   - `corrcoef()` $-1 \le r \le 1$
   
   ```matlab
   x = [-1.2 -0.5 0.3 0.9 1.8 2.6 3.0 3.5];
   y = [-15.6 -8.5 2.2 4.5 6.6 8.2 8.9 10.0];
   scatter(x, y);
   box on;
   axis square;
   corrcoef(x, y)
   ```
   
   $r$ 为相关系数，越接近1为正相关；越接近-1为负相关；
   
   Out:
   
   ![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-18%20162935.png)
   
   ![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-18%20162955.png)
   
   ```matlab
   % 高阶多项式
   x = [-1.2 -0.5 0.3 0.9 1.8 2.6 3.0 3.5];
   y = [-15.6 -8.5 2.2 4.5 6.6 8.2 8.9 10.0];
   figure('Position', [50 50 1500 400]);
   for i = 1: 3
       subplot(1, 3, i); p = polyfit(x, y, i);
       xfit = x(1): 0.1: x(end); yfit = polyval(p, xfit);
       plot(x, y, 'ro', xfit, yfit); set(gca, 'FontSize', 14);
       ylim([-17, 11]); legend(4, 'Data points', 'Fitted curve');
   end
   ```
   
   Out:
   
   ![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-07-18%20163433.png)

3. 回归的变量有多个
   
   $$
   y = \beta_0 + \beta_1x_1 + \beta_2x_2
   $$
   
   使用 `regress()`
   
   ```matlab
   % 内置数据集
   load carsmall;
   y = MPG;
   x1 = Weight; x2 = Horsepower;
   X = [ones(length(x1), 1) x1 x2];
   b = regress(y, X);
   x1fit = min(x1): 100: max(x1);
   x2fit = min(x2): 10: max(x2);
   [X1FIT, X2FIT] = meshgrid(x1fit, x2fit);
   YEIT = b(1) + b(2)*X1FIT + b(3)*X2FIT;
   scatter3(x1, x2, y, 'filled');
   hold on;
   mesh(X1FIT, X2FIT, YFIT);
   hold off;
   xlabel('Weight');
   ylabel('Horsepower');
   zlabel('MPG');
   view(50, 10);
   ```

# 图像处理

## 读取图像

```matlab
file_path = '.\附件1\';  % 图像文件夹路径  
img_path_list = dir(strcat(file_path,'*.bmp'));  % 获取该文件夹中所有png格式的图像  
img_num = length(img_path_list)  % 获取图像总数量 
I=cell(1,img_num);
if img_num > 0  % 有满足条件的图像  
    for j = 1:img_num  % 逐一读取图像  
        image_name = img_path_list(j).name;  % 图像名  
        image = imread(strcat(file_path,image_name));
        I{j}=image;
        fprintf('%d %d %s\n',i,j,strcat(file_path,image_name));  % 显示正在处理的图像名  
        % 图像处理过程 省略  
        % 这里直接可以访问细胞元数据的方式访问数据
        % 二值化图像
        Pic = im2bw(I{j});
        % edge 算法的 log 算子对二值化图像进行边缘提取
        PicEdge = edge(Pic, 'log');
        subplot(4, 5, j)
        imshow(PicEdge)
        title(image_name)
    end
end
```

## 图像边缘提取

对图片进行边缘提取，其中一种就是 edge 算法，这个 edge 算法中有好几个算子，每一个算子分别对应着一种边缘提取的原理

```matlab
% 读取一张图片，并显示
original_picture=imread('D:\SoftWare\matlab2016a\Project\Picture\cat.jpg');
Pic2 = im2bw(original_picture, thresh);
figure(1)
subplot(2, 2, 1);
imshow(original_picture);
title('原始RGB图像')
subplot(222)
imshow(Pic2)
title('二值化图像')

% 用 edge 算法对二值化图像进行边缘提取
PicEdge1 = edge(Pic2,'log');
subplot(223);
imshow(PicEdge1);
title('log算子')

PicEdge2 = edge(Pic2, 'canny');
subplot(224);
imshow(PicEdge2);
title('canny算子');

PicEdge3 = edge(Pic2, 'sobel');
figure(2)
subplot(221)
imshow(PicEdge3);
title('sobel算子')

PicEdge4 = edge(Pic2, 'prewitt');
subplot(222)
imshow(PicEdge4);
title('sprewitt算子')

PicEdge5 = edge(Pic2, 'zerocross');
subplot(223)
imshow(PicEdge5);
title('zerocross算子')

PicEdge6 = edge(Pic2, 'roberts');
subplot(224)
imshow(PicEdge6);
title('roberts算子')
```

Out:

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-08-10%20025633.png)

![](D:\HANSHAN\Mathmatical%20Modeling\Picture_MATLAB\屏幕截图%202022-08-10%20025650.png)
