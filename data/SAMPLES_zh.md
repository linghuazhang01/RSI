# 数据样例（中文译本；训练集每个 10 道，评测集每个 3 道）

由 `data/scripts/show_samples.py` 生成（随机种子 20261118）。训练集从抽样池（`pool`，GPQA Main 与 IFEval 从验收集 `gate`）里抽，评测集从留出池（`test`）里抽。长文本已截断。GPQA 按数据作者要求不展示题目。

> 本文件是 `data/SAMPLES.md` 的中文译本，仅供阅读。训练和评测用的仍是英文原文：指令遵循的约束（如全小写、关键词次数）和工具调用的参数值都按英文判分。代码、公式、JSON、函数名和参数值保持原样。

## 数学

### 训练集：gsm8k_train（pool，7221 条）

**gsm8k/train/6012**

题目：
````
一位母亲在儿子一岁生日那天种下了一棵 16 英寸高的树。种树那天，男孩身高 24 英寸。如果这棵树的生长速度是男孩的两倍，那么当男孩长到 36 英寸高时，这棵树有多高？
````
答案 / 判分依据：
````
40
````

**gsm8k/train/7338**

题目：
````
乔为澳大利亚之行存了 $6,000。如果他在机票上花了 $1,200，在酒店上花了 $800，在餐饮上花了 $3,000，他还剩下多少美元？
````
答案 / 判分依据：
````
1000
````

**gsm8k/train/2560**

题目：
````
史密斯先生因为未能按时支付价值 $150 的账单，被收取了 2% 的滞纳金。如果他打算今天结清账单，史密斯先生需要支付多少钱？
````
答案 / 判分依据：
````
153
````

**gsm8k/train/3620**

题目：
````
水箱上的水位计显示水箱有 1/3 的水。要把水箱加满，需要再加入 16 加仑水。水箱装满时能容纳多少加仑水？
````
答案 / 判分依据：
````
24
````

**gsm8k/train/1566**

题目：
````
两位公寓室友每月平均分摊房租、水电费和食品杂货开支。整套公寓的房租是 $1100，水电费是 $114。如果其中一位室友一共支付了 $757，那么整套公寓的食品杂货开支是多少美元？
````
答案 / 判分依据：
````
300
````

**gsm8k/train/1345**

题目：
````
Mimi 决定重新开始去健身房。周末她花了 $8,000 购买运动鞋和服装。她在 Nike 运动鞋上花的钱是 Adidas 的三倍。她在 Adidas 上花的钱是 Skechers 价格的 1/5。如果 Mimi 购买 Adidas 运动鞋花了 $600，她在服装上花了多少钱？
````
答案 / 判分依据：
````
2600
````

**gsm8k/train/6744**

题目：
````
Genevieve 从超市货架上挑了一些樱桃，价格为每千克 $8。当 Genevieve 走到收银台时，她发现自己还差 $400 才够付总价，于是她的朋友 Clarice 凑了一部分钱。如果 Genevieve 身上有 $1600，她买了多少千克樱桃？
````
答案 / 判分依据：
````
250
````

**gsm8k/train/1135**

题目：
````
Reggie 的父亲给了他 $48。Reggie 买了 5 本书，每本 $2。Reggie 还剩多少钱？
````
答案 / 判分依据：
````
38
````

**gsm8k/train/2727**

题目：
````
Oshea 买了 200 颗罗勒种子。他想把它们种到花盆里。他有 4 个大花盆，每个可以种 20 颗种子。他还有一些小花盆，每个可以种 4 颗种子。他需要多少个小花盆才能种下所有的罗勒种子？
````
答案 / 判分依据：
````
30
````

**gsm8k/train/3912**

题目：
````
Quincy 刚刚用一笔 5 年期无息贷款买了一辆车。他支付了 $5,000.00 的首付，使得每月还款额为 $250.00。这辆车的价格是多少？
````
答案 / 判分依据：
````
20000
````

### 训练集：math_train（pool，7116 条）

**math/train/geometry/289**

题目：
````
一根铁丝被剪成两段，一段长 $a$，另一段长 $b$。长为 $a$ 的一段被弯成一个正三角形，长为 $b$ 的一段被弯成一个正六边形。该三角形与该六边形的面积相等。$\frac{a}{b}$ 是多少？
````
答案 / 判分依据：
````
\frac{\sqrt6}2
````

**math/train/intermediate_algebra/631**

题目：
````
设 $f(x)$ 是一个实系数三次多项式，满足 \[|f(1)|=|f(2)|=|f(3)|=|f(5)|=|f(6)|=|f(7)|=12.\]求 $|f(0)|$。
````
答案 / 判分依据：
````
72
````

**math/train/prealgebra/379**

题目：
````
求 8 和 15 的最小公倍数。
````
答案 / 判分依据：
````
120
````

**math/train/intermediate_algebra/1238**

题目：
````
计算 $\left|\frac12 - \frac38i\right|$。
````
答案 / 判分依据：
````
\frac58
````

**math/train/prealgebra/367**

题目：
````
Auston 身高 60 英寸。使用换算关系 1 英寸 = 2.54 厘米，Auston 的身高是多少厘米？答案用小数表示，精确到十分位。
````
答案 / 判分依据：
````
152.4
````

**math/train/algebra/707**

题目：
````
四年前你以 $10\%$ 的利率投资了一笔钱。现在账户里有 $\$439.23$。如果利息按年复利计算，你四年前投资了多少钱？
````
答案 / 判分依据：
````
300
````

**math/train/counting_and_probability/211**

题目：
````
在杨辉三角（帕斯卡三角形）中，以 1 开头、接着是 12 的那一行，第四个数是多少？
````
答案 / 判分依据：
````
220
````

**math/train/prealgebra/18**

题目：
````
求前七个个位数字为 7 的素数之和。
````
答案 / 判分依据：
````
379
````

**math/train/algebra/1022**

题目：
````
解关于 $m$ 的方程：$(m-4)^3 = \left(\frac 18\right)^{-1}$。
````
答案 / 判分依据：
````
6
````

**math/train/counting_and_probability/306**

题目：
````
有无限多个由彩色纸做成的全等正三角形。每个三角形是纯色的，纸的两面颜色相同。如图所示，用四个这样的纸三角形拼成一个大正三角形。如果无法通过平移、旋转和/或翻转把一个大三角形放到另一个上，使得它们对应的小三角形颜色相同，则称这两个大三角形是可区分的。已知可供选择的三角形颜色共有六种，可以构造出多少个可区分的大正三角形？

[asy]
draw((0,0)--(10,0)--(5,8.7)--cycle);
draw((5,0)--(2.5,4.3)--(7.5,4.3 …（共 717 字符，已截断）
````
答案 / 判分依据：
````
336
````

### 训练集：dapo_math（pool，15929 条）

**dapo/8fd50a64-31c9-465c-870e-342dfd6bc2e8**

题目：
````
下列哪一项等于乘积
$\frac{8}{4}\cdot\frac{12}{8}\cdot\frac{16}{12}\cdot\cdots\cdot\frac{4n+4}{4n}\cdot\cdots\cdot\frac{2008}{2004}$？
$(A)\ 251\qquad(B)\ 502\qquad(C)\ 1004\qquad(D)\ 2008\qquad(E)\ 4016$
````
答案 / 判分依据：
````
502
````

**dapo/0e8faa6c-bfc5-401d-95aa-8b2fc661c7b3**

题目：
````
方程 \df{1}{x}+\df{1}{y}=\df{1}{2021} 的正整数解的组数为__________.
````
答案 / 判分依据：
````
9
````

**dapo/61fabc4c-b456-413b-87c7-a8e43bef9728**

题目：
````
从集合 $\{1,2,3,4,5,6,7,8,9,10\}$ 中随机选取两个不同的整数 $x$ 和 $y$。$xy-x-y$ 为偶数的概率是多少？答案的形式为 \frac{m}{n}，其中 gcd(m, n) = 1。请给出 m + n 的值。
````
答案 / 判分依据：
````
11
````

**dapo/0209e8b8-c475-4375-8a38-afd1d52e9818**

题目：
````
正方形 $ABCD$ 的面积为 $36$，且 $\overline{AB}$ 平行于 $x$ 轴。顶点 $A$、$B$ 和 $C$ 分别在 $y = \log_{a}x$、$y = 2\log_{a}x$ 和 $y = 3\log_{a}x$ 的图像上。原答案的形式为 \(\sqrt[k]{3}\)。请确定 $k$ 的值。
````
答案 / 判分依据：
````
6
````

**dapo/36ab56f8-8ebb-4794-b767-b4f674d17bc5**

题目：
````
在一个 $4 \times 4$ 的棋盘上，每个方格里各写一个 $1$ 到 $16$ 的数。Andres 和 Pablo 各选四个数。Andres 选出每一行中最大的数，Pablo 选出每一列中最大的数。同一个数可以被两人同时选中。然后，所有被选中的数都从棋盘上移除。棋盘上剩下的数之和最大可能是多少？
````
答案 / 判分依据：
````
78
````

**dapo/48c2b92d-5074-49ad-8352-47ceca2a8781**

题目：
````
求下图中三角形 $JKL$ 的面积。


[asy]
unitsize(1inch);
pair P,Q,R;
P = (0,0);
Q= (sqrt(3),0);
R = (0,1);
draw (P--Q--R--P,linewidth(0.9));
draw(rightanglemark(Q,P,R,3));
label("$J$",P,S);
label("$K$",Q,S);
label("$L$",R,N);
label("$20$",(Q+R)/2,NE);
label("$60^\circ$",(0,0.75),E);
[/asy]答案的形式为 k\sqrt{m}+n,。请给出 k + m + n 的值。
````
答案 / 判分依据：
````
53
````

**dapo/5e59caa8-1717-4e0a-b590-77dde0212db0**

题目：
````
圆内接四边形 $ABCD$ 的边长为 $AB = 3, BC = AD = 5$，$CD = 8$。它的外接圆半径可以写成 $a\sqrt{b}/c$ 的形式，其中 $a, b, c$ 为正整数，$a, c$ 互素，且 $b$ 不能被任何素数的平方整除。求 $a + b + c$。
````
答案 / 判分依据：
````
16
````

**dapo/c3b8c638-5fc0-44fe-aead-339da9af7b20**

题目：
````
集合 \( \{1, 2, \dots, 15\} \) 有多少个非空子集，使得该子集中各元素的平方和是 \( 5 \) 的倍数？
````
答案 / 判分依据：
````
7583
````

**dapo/19a2f9db-5090-4259-a463-1523bd00d90b**

题目：
````
如果从集合 $\{ 2, 4, 10, 12, 15, 20, 50 \}$ 中随机选取两个不同的元素并相乘，乘积是 100 的倍数的概率是多少？请用普通分数表示答案。答案的形式为 \frac{m}{n}，其中 gcd(m, n) = 1。请给出 m + n 的值。
````
答案 / 判分依据：
````
4
````

**dapo/968c8a71-dcdc-4d49-9fac-c58f9092af44**

题目：
````
你有两块黑板 $A$ 和 $B$。你需要在它们上面写一些整数，每个整数都大于或等于 $2$ 且小于或等于 $20$。条件是：黑板 $A$ 上的每个数都必须与黑板 $B$ 上的每个数互素。求写在 $A$ 上的整数个数与写在 $B$ 上的整数个数之积的最大可能值。
````
答案 / 判分依据：
````
65
````

### 评测集：MATH-500（test，500 条）

**math/test/precalculus/470**

题目：
````
对于常数 $c,$ 在柱坐标 $(r,\theta,z)$ 中，求方程所描述的形状
\[z = c.\](A) 直线
(B) 圆
(C) 平面
(D) 球面
(E) 圆柱面
(F) 圆锥面

请填写正确选项的字母。
````
答案 / 判分依据：
````
\text{(C)}
````

**math/test/precalculus/396**

题目：
````
求过点 $(0,-1,-1),$ $(-4,4,4),$ 和 $(4,5,1)$ 的平面方程。请将答案写成如下形式
\[Ax + By + Cz + D = 0,\]其中 $A,$ $B,$ $C,$ $D$ 为整数，满足 $A > 0$ 且 $\gcd(|A|,|B|,|C|,|D|) = 1.$
````
答案 / 判分依据：
````
5x - 7y + 11z + 4 = 0
````

**math/test/intermediate_algebra/659**

题目：
````
求 $(x - 3)^3 + (x -7)^3 = (2x - 10)^3$ 的根。
````
答案 / 判分依据：
````
3, 5, 7
````

### 评测集：gsm8k（test，1319 条）

**gsm8k/test/195**

题目：
````
一支钢笔的价格等于一支铅笔和一块橡皮的价格之和。一支铅笔 $1.20，一块橡皮 $0.30。8 支钢笔多少钱？
````
答案 / 判分依据：
````
12
````

**gsm8k/test/530**

题目：
````
在记录了 8 周的锻炼情况后，Josh 发现他每周锻炼 4 小时，只有个别情况例外。有一次，他连续两周每周锻炼 5 小时。另有一次，他在一周内锻炼了 6 小时。他这 8 周一共锻炼了多少小时？
````
答案 / 判分依据：
````
36
````

**gsm8k/test/1263**

题目：
````
Conor 住在海滩附近，喜欢每天去那里玩。某一周的第一天，他在海滩上看到 50 个人。第二天，海滩上的人比第一天多 20 人，第三天，总人数增加到第二天人数的两倍。如果这一周剩下的每天 Conor 平均看到 60 个人，请计算 Conor 这一周在海滩上看到的总人数。
````
答案 / 判分依据：
````
500
````

### 评测集：aime24（test，30 条）

**aime24/66**

题目：
````
设 $\mathcal{B}$ 为表面积为 $54$、体积为 $23$ 的长方体的集合。设 $r$ 为能够容纳 $\mathcal{B}$ 中每个长方体的最小球的半径。$r^2$ 的值可以写成 $\frac{p}{q}$，其中 $p$ 和 $q$ 是互素的正整数。求 $p+q$。
````
答案 / 判分依据：
````
721
````

**aime24/78**

题目：
````
设 \(O=(0,0)\)，\(A=\left(\tfrac{1}{2},0\right)\)，\(B=\left(0,\tfrac{\sqrt{3}}{2}\right)\) 为坐标平面内的点。设 \(\mathcal{F}\) 为位于第一象限内、长度为 1 的线段 \(\overline{PQ}\) 构成的族，其中 \(P\) 在 $x$ 轴上，\(Q\) 在 $y$ 轴上。在 \(\overline{AB}\) 上存在唯一一点 \(C\)，异于 \(A\) 和 \(B\)，它不属于 \(\mathcal{F}\) 中除 \(\overline{AB}\) 以外的任何线段。于是 \(OC^2=\tfrac{p}{q}\)，其中 \(p\) 和 \(q\) 是互素的正整数。求 \(p+q\)。
````
答案 / 判分依据：
````
023
````

**aime24/67**

题目：
````
存在大于 1 的实数 $x$ 和 $y$，使得 $\log_x\left(y^x\right)=\log_y\left(x^{4y}\right)=10$。求 $xy$。
````
答案 / 判分依据：
````
025
````

### 评测集：aime25（test，30 条）

**aime25/12**

题目：
````
三维坐标空间中位于平面 $x+y+z=75$ 上且坐标满足不等式
$$x-yz<y-zx<z-xy$$
的点的集合构成三个互不相交的凸区域。其中恰有一个区域的面积是有限的。该有限区域的面积可以表示为 $a\sqrt{b}$ 的形式，其中 $a$ 和 $b$ 是正整数，且 $b$ 不能被任何素数的平方整除。求 $a+b$。
````
答案 / 判分依据：
````
510
````

**aime25/23**

题目：
````
Silas 有无限多枚 $1$ 美分、$10$ 美分和 $25$ 美分的硬币，他想凑出总价值为 $N$ 美分的一组硬币，其中 $N$ 是正整数。他使用所谓的\textit{贪心算法}：依次选择面值最大且不会使硬币总值超过 $N$ 的硬币。例如，要凑 $42$ 美分，Silas 会先选一枚 $25$ 美分硬币，再选一枚 $10$ 美分硬币，然后选 $7$ 枚 $1$ 美分硬币。然而，这组共 $9$ 枚硬币比凑出 $42$ 美分所需的硬币数更多；事实上，选 $4$ 枚 $10$ 美分硬币和 $2$ 枚 $1$ 美分硬币就能以仅 $6$ 枚硬币达到同样的总值。

一般地，如果对于给定的 $N$，没有任何其他由 $1$ 美分、$10$ 美分和 $25$ 美分硬币组成、总值为 $N$ 美分的组合所用硬币数严格少于贪心算法给出的组合，则称贪心算法对该 $N$ \emph{成功}。…（共 1025 字符，已截断）
````
答案 / 判分依据：
````
610
````

**aime25/21**

题目：
````
半径为 $6$、圆心为 $A$ 的圆 $\omega_1$ 在点 $B$ 处与半径为 $15$ 的圆 $\omega_2$ 内切。点 $C$ 和 $D$ 在 $\omega_2$ 上，使得 $\overline{BC}$ 是 $\omega_2$ 的直径，且 $\overline{BC} \perp \overline{AD}$。矩形 $EFGH$ 内接于 $\omega_1$，使得 $\overline{EF} \perp \overline{BC}$，$C$ 离 $\overline{GH}$ 比离 $\overline{EF}$ 更近，$D$ 离 $\overline{FG}$ 比离 $\overline{EH}$ 更近，如图所示。三角形 $\triangle DGF$ 和 $\triangle CHG$ 的面积相等。矩形 $EFGH$ 的面积为 $\frac{m}{n}$，其中 $m$ 和 $n$ 是互素的正整数。求 $m + n$。


\begin{tikzpicture}[scale=0.15]

    \coordinate (A) at (0,0);
    \coordinate (B) at (6,0);
    \coordinate (C) at (-24,0);
    \coordinate (D) at (0,12);

    \coordinate (O) at (-9, 0);

    \def\rone{6} 
    \def\rtwo{15} …（共 1895 字符，已截断）
````
答案 / 判分依据：
````
293
````

## 代码

### 训练集：taco_verified（pool，9806 条）

**taco/18347**

题目：
````
给定一个由小写英文字母组成的单词，编写程序去除其中的重复字母。输出中的字符必须保持与其在原单词中首次出现时相同的顺序。

输入格式

输入包含若干组测试数据。
输入文件的第一行包含一个正整数 T，表示测试数据的组数。
接下来是 T 行，每行包含一个单词 W（不含空格，只含小写英文字母）。

输出格式

输出必须恰好包含 T 行，每行包含一个单词，即所求的答案。

约束条件

1 ≤ T ≤ 100
1 ≤ |W| ≤ 31

SAMPLE INPUT
2
hello
world
mississippi

SAMPLE OUTPUT
helo
world
misp
````
答案 / 判分依据：
````python
（hackerearth，难度 MEDIUM，1 个测试，标准输入输出）
参考解：
t = int(input())
words = list()

while t > 0:
	word = input()
	index = [0] * 26
	new_word = ""
	for char in word:
		asc_val = ord(char) - 97
		if(index[asc_val] == 1):
			pass
		else:
			new_word += char
			index[asc_val] = 1
	print(new_word)
	t -= 1
````

**taco/11891**

题目：
````
高桥决定送给他的母亲一个字符串。

字符串 T 的价值定义为 T 与 T' 的最长公共子序列的长度，其中 T' 是将 T 反转后得到的字符串。也就是说，价值是下面两个相等的字符串的最大长度：T 的一个子序列（可以不连续），以及 T' 的一个子序列（可以不连续）。

高桥有一个字符串 S。他想送给母亲一个价值尽可能高的字符串，因此他希望最多把 S 中的 K 个字符改成任意其他字符，以得到价值尽可能高的字符串。请求出能够达到的最高价值。

约束条件

* 1 \leq |S| \leq 300
* 0 \leq K \leq |S …（共 978 字符，已截断）
````
答案 / 判分依据：
````python
（atcoder，难度 UNKNOWN_DIFFICULTY，102 个测试，标准输入输出）
参考解：
import sys
readline = sys.stdin.readline
S = list(map(ord, readline().strip()))
N = len(S)
S = [None] + S + [None]
K = int(readline())
if N == 1:
	ans = 1
elif N == 2:
	ans = 1
	if S[1] == S[2]:
		ans = 2
	if K:
		ans = 2
else:
	dp1 = [[0] * (K + 1) for _ in range(N + 2)]
	dp2 = [[0] * (K + 1) for _ in range(N + 2)]
	for i in range(1, N + 1):
		dp1[i][0] = 1
	for i in range(1, N):
		dp2[i][0] = 1
		if S[i] == S[i + 1]:
			dp2[i][0] = 2
		elif K:
			dp2[i][1] = 2
	for s in range(3, N + 1):
		dp3 = [[0] * (K + 1) for _ in range(N + 2)]
		for i …（共 1051 字符，已截断）
````

**taco/5096**

题目：
````
厨师有一个大型乐团在他的厨房里，他正在为下周的音乐会做准备。他还有一份乐谱，用长度为 $N$ 的字符串 $S$ 表示。乐谱中的每个音符用一个小写或大写英文字母表示；小写字母 'a' 到 'z' 的值为 $1$ 到 $26$，大写字母 'A' 到 'Z' 的值为 $27$ 到 $52$。

厨师认为有 $M$ 段旋律是*优美旋律*（编号 $1$ 到 $M$）。对每个合法的 $i$，第 $i$ 段优美旋律用字符串 $T_{i}$ 表示，其*优美值*为 $C_{i}$。

此外还有 $A$ 段*好旋律*。对每个合法的 $i$，第 $i$ 段好旋律由 …（共 7975 字符，已截断）
````
答案 / 判分依据：
````python
（codechef，难度 VERY_HARD，1 个测试，标准输入输出）
参考解：
(n, m, a, x, r) = map(int, input().split())
s = input()
t = []
c = []
p = []
q = []
for i in range(m):
	(y, z) = input().split()
	t.append(y)
	c.append(z)
for i in range(a):
	(y, z) = input().split()
	p.append(y)
	q.append(z)
if s == 'AFBBet':
	print('4\n1 2 A\n3 1 3\n2 7 8 2\n1 9 p')
else:
	print(0)
````

**taco/16686**

题目：
````
在你的密码分析师新工作的第三天，你遇到了迄今为止最棘手的任务。你的工作是实现一个简单的关键词密码。关键词密码是一种单表替换密码，需要提供两个参数（字符串, 关键词）。加密方法是：取关键词，去掉其中重复出现的字母；然后把字母表中未使用的其余字母依次追加到关键词末尾。

例如，如果你的字符串是 "hello"，关键词是 "wednesday"，那么你的加密密钥就是 'wednsaybcfghijklmopqrtuvxz'。

要加密 'hello'，你需要按如下方式替换，
```
              abcdefghijklmnopqrstuvwxyz
  hello ==>   | …（共 908 字符，已截断）
````
答案 / 判分依据：
````python
（codewars，难度 EASY，10 个测试，函数式 keyword_cipher）
参考解：
abc = 'abcdefghijklmnopqrstuvwxyz'

def keyword_cipher(s, keyword, key=''):
	for c in keyword + abc:
		if c not in key:
			key += c
	return s.lower().translate(str.maketrans(abc, key))
````

**taco/8581**

题目：
````
会津若松市政府决定铺设一条覆盖全市区域的热水管道，为住宅供暖。管道从若干温泉出发，连接市内的每一个地区。管道可以在温泉或地区处分叉，但不允许出现环。市政府希望使管道的总长度最小，以便以最低的费用建成。

请编写程序，计算管道的最小长度。程序读入的输入由以下三部分组成：

提示

第一行对应第一部分：有三个温泉和五个地区。接下来的三行是第二部分：一个 h …（共 2403 字符，已截断）
````
答案 / 判分依据：
````python
（aizu，难度 UNKNOWN_DIFFICULTY，101 个测试，标准输入输出）
参考解：
while 1:
	(s, d) = [int(x) for x in input().split()]
	A = [999 for x in range(d)]
	if s == 0:
		break
	for j in range(s):
		l = [int(x) for x in input().split()]
		for i in range(d):
			if l[i] != 0:
				A[i] = min(A[i], l[i])
	B = [[999 for x in range(d)] for y in range(d)]
	for j in range(d - 1):
		l = [int(x) for x in input().split()]
		for i in range(d - 1 - j):
			if l[i] != 0:
				B[j][i + j + 1] = l[i]
				B[i + j + 1][j] = l[i]
	notevaled = [1 for x in range(d)]
	ans = 0
	for i in range(d):
		mini = A.index(min(A))
		ans += A[mini]
		not …（共 718 字符，已截断）
````

**taco/19872**

题目：
````
阅读 [俄语] 的题目描述

厨师有一个序列 $A_{1}, A_{2}, \ldots, A_{N}$。在一次操作中，厨师可以选择一个下标 $i$（$1 ≤ i ≤ N$），并把 $A_{i}$ 变为 $A_{i}^{p}$，其中 $p = \max(0, {\lceil \frac{A_{i}}{2} \rceil} - 1)$。

请帮助厨师求出使序列之和为偶数所需的最少操作次数，或者判定这是不可能的。

------  输入 ------
输入的第一行包含一个整数 $T$，表示测试数据的组数。接下来是 $T$ 组测试数据的描述。
每组测试数据的第一行包含一个整数 $N$。
第二行包含 $N$ 个以空格分隔的整数 $A_{1}, A_{2}, \ldots, A_{N}$。

------  输出 --- …（共 1623 字符，已截断）
````
答案 / 判分依据：
````python
（codechef，难度 MEDIUM，1 个测试，标准输入输出）
参考解：
t = int(input())
for _ in range(t):
	n = int(input())
	a = list(map(int, input().split()))
	if sum(a) % 2 == 0:
		print(0)
	else:
		c = 0
		for i in range(n):
			if a[i] == 2:
				print(1)
				c = 1
				break
		if c == 0:
			print(-1)
````

**taco/15713**

题目：
````
伽利略最近的项目是确定天空中某些区域的恒星密度。为此，他开始在网上寻找数据集，并在牛顿的博客上发现了一个数据集。牛顿把夜空分解为一个沃罗诺伊（Voronoi）剖分，其生成点排列成网格。他把每个沃罗诺伊单元中的恒星数量存放在矩阵中与该生成点在网格中的位置相对应的位置上。
这个数据集并不能直接帮到伽利略，因为他需要能够查询天空中某个矩形区域内的恒星数量。伽利略尝试自己写了一个程序来完成这件事，但结果太慢了。你能帮帮他吗？

----- …（共 1443 字符，已截断）
````
答案 / 判分依据：
````python
（codechef，难度 UNKNOWN_DIFFICULTY，1 个测试，标准输入输出）
参考解：
import sys

def main():
	s = sys.stdin.readline
	(n, m) = list(map(int, s().split()))
	nums = {}
	for i in range(1, n + 1):
		nums[i] = list(map(int, s().split()))
	cases = int(s())
	for case in range(cases):
		(px, py, qx, qy) = list(map(int, s().split()))
		ans = []
		for i in range(px, qx + 1):
			for j in range(py - 1, qy):
				ans.append(nums[i][j])
		print(sum(ans))

def __starting_point():
	main()
__starting_point()
````

**taco/2695**

题目：
````
集合是不含重复元素的无序元素集合。

当集合被打印、迭代或转换为序列时，其元素将以任意顺序出现。

示例

>>> print set()
set([])

>>> print set('HackerRank')
set(['a', 'c', 'e', 'H', 'k', 'n', 'r', 'R'])

>>> print set([1,2,1,2,3,4,5,6,0,9,12,22,3])
set([0, 1, 2, 3, 4, 5, 6, 9, 12, 22])

>>> print set((1,2,3,4,5,5))
set([1, 2, 3, 4, 5])

>>> print set(set(['H','a','c','k','e','r','r','a','n','k']))
set(['a', 'c', 'r', 'e', 'H', 'k', 'n'])

>>> print set({'Hacker' : 'DOSHI', 'Rank' : 616 })
set(['Hacker', 'Rank'])

>>> print set(enumerate(['H','a','c','k','e','r','r','a','n','k']))
set([(6, 'r'), (7, 'a'), (3, 'k' …（共 2150 字符，已截断）
````
答案 / 判分依据：
````python
（hackerrank，难度 EASY，1 个测试，标准输入输出）
参考解：
n = int(input())
val = list(map(int, input().split()))
val = set(val)
print(sum(val) / len(val))
````

**taco/24557**

题目：
````
厨师有一个由 N 个元素组成的数组 A。他想求出满足下列条件的不相交区间对 [a, b] 和 [c, d]（1 ≤ a ≤ b < c ≤ d ≤ N）的数量：不存在任何一个数同时出现在子数组 {Aa, Aa+1, ... , Ab} 和 {Ac, Ac+1, ... , Ad} 中。
请帮助厨师求出这个数量。

-----输入-----
- 输入的第一行包含一个整数 T，表示测试数据的组数。接下来是 T 组测试数据的描述。
- 每组测试数据的第一行包含一个整数 N，表示数组中元素的个数。
- 第二行包含 N 个以空格分隔的整数 A1, A2, ..., AN。

-----输出-----
- 对每组测试数据，输出一行，包含一个 …（共 1282 字符，已截断）
````
答案 / 判分依据：
````python
（codechef，难度 UNKNOWN_DIFFICULTY，1 个测试，标准输入输出）
参考解：
def repeats(A, x):
	for i in range(N):
		if x == A[i]:
			return i
	return -1

def assign_indices(A, newA):
	for i in range(N):
		x = repeats(A, A[i])
		if x == -1:
			newA[i] = i
		else:
			newA[i] = x
T = int(input())
for t in range(T):
	N = int(input())
	A = input().split(' ')
	indices = [0 for x in range(1005)]
	for i in range(N):
		A[i] = int(A[i])
	assign_indices(A, indices)
	total = 0
	taken = []
	for i in range(N - 1):
		taken = [0 for x in range(1005)]
		for j in range(i, N):
			before = 0
			taken[indices[j]] += 1
			for k in range(j …（共 722 字符，已截断）
````

**taco/23909**

题目：
````
AtCoder 幼儿园的两个学生在争抢糖果包。
一共有三包糖果，分别含有 a、b、c 颗糖。
Evi 老师想把这些糖果包分给两个学生，使每个学生得到的糖果数相同。请判断是否可行。
注意，Evi 不能把糖果从包里拿出来，每一包的全部内容必须整包给其中一个学生。

-----约束条件-----
 - 1 ≦ a, b, c ≦ 100

-----输入-----
输入以如下格式从标准输入给出：
a b c

-----输出-----
如果可以分配这些糖果包使每个学生得到的糖果数相同，输出 Yes。否则 …（共 913 字符，已截断）
````
答案 / 判分依据：
````python
（atcoder，难度 EASY，106 个测试，标准输入输出）
参考解：
(a, b, c) = map(int, input().split())
if a == b + c or b == a + c or c == a + b:
	print('Yes')
else:
	print('No')
````

### 评测集：humanevalplus（test，164 条）

**HumanEval/110**

题目：
````
def exchange(lst1, lst2):
    """在这个问题中，你需要实现一个函数，它接收两个数字列表，
    并判断是否可以在它们之间交换元素，使 lst1 成为一个只包含偶数的列表。
    lst1 与 lst2 之间交换元素的个数没有限制。
    如果可以在 lst1 与 lst2 之间交换元素，使 lst1 的所有元素都变为偶数，
    则返回 "YES"。
    否则返回 "NO"。
    例如：
    exchange([1, 2, 3, 4], [1, 2, 3, 4]) => "YES"
    exchange([1, 2, 3, 4], [1, 5, 3, 4]) => "NO"
    假定输入的列表都是非空的。
    """
````
答案 / 判分依据：
````python
cnt_odd = len(list(filter(lambda x: x % 2 == 1, lst1)))
    cnt_even = len(list(filter(lambda x: x % 2 == 0, lst2)))
    return "YES" if cnt_odd <= cnt_even else "NO"
````

**HumanEval/112**

题目：
````
def reverse_delete(s,c):
    """任务
    给定两个字符串 s 和 c，你需要删除 s 中所有与 c 中任意字符相等的字符，
    然后检查结果字符串是否为回文串。
    如果一个字符串正着读和倒着读都一样，则称其为回文串。
    你应当返回一个元组，包含结果字符串以及检查结果 True/False。
    示例
    对于 s = "abcde", c = "ae"，结果应为 ('bcd',False)
    对于 s = "abcdef", c = "b"  结果应为 ('acdef',False)
    对于 s = "abcdedcba", c = "ab"，结果应为 ('cdedc',True)
    """
````
答案 / 判分依据：
````python
ss = "".join(filter(lambda ch: ch not in c, s))
    return ss, ss == ss[::-1]
````

**HumanEval/19**

题目：
````
from typing import List


def sort_numbers(numbers: str) -> str:
    """ 输入是一个以空格分隔的字符串，由 'zero' 到 'nine' 的数字单词组成。
    合法的取值有 'zero'、'one'、'two'、'three'、'four'、'five'、'six'、'seven'、'eight' 和 'nine'。
    返回将这些数字从小到大排序后的字符串
    >>> sort_numbers('three one five')
    'one three five'
    """
````
答案 / 判分依据：
````python
to_int = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}

    if numbers == "": return ""
    return " ".join(sorted(numbers.split(" "), key=lambda n: to_int[n]))
````

### 评测集：mbppplus（test，378 条）

**Mbpp/462**

题目：
````
编写一个函数，找出给定列表中元素的所有可能组合。
````
答案 / 判分依据：
````python
def combinations_list(list1):
    if len(list1) == 0:
        return [[]]
    result = []
    for el in combinations_list(list1[1:]):
        result += [el, el+[list1[0]]]
    return result
````

**Mbpp/75**

题目：
````
编写一个函数，从给定的元组列表中找出所有元素都能被 k 整除的元组。
````
答案 / 判分依据：
````python
def find_tuples(test_list, K):
  res = [sub for sub in test_list if all(ele % K == 0 for ele in sub)]
  return res
````

**Mbpp/798**

题目：
````
编写一个 Python 函数，求数组的和。
````
答案 / 判分依据：
````python
def _sum(arr):  
    return sum(arr)
````

### 评测集：livecodebench（test，131 条）

**lcb/abc398_f**

题目：
````
找出一个以 S 为前缀的最短回文串。

输入

输入以如下格式从标准输入给出：
S

输出

输出答案。
如果存在多个解，输出其中任意一个均可被接受。

约束条件


- S 是长度在 1 到 500000 之间（含端点）、由大写英文字母组成的字符串。

样例输入 1

ABC

样例输出 1

ABCBA

ABCBA 是以 S= ABC 为前缀的一个最短回文串。

样例输入 2

Z

样例输出 2

Z

Z 是以 S= Z 为前缀的一个最短回文串。

样例输入 3

TREE

样例输出 3

TREERT

TREERT 是以 S= TREE 为前缀的一个最短回文串。
````
答案 / 判分依据：
````
无参考解；公开测试示例 [{"input": "ABC", "output": "ABCBA", "testtype": "stdin"}]
````

**lcb/3793**

题目：
````
给你两个字符串 s 和 t。
你可以从 s 中选取一个子串（可以为空），再从 t 中选取一个子串（可以为空），然后按顺序将它们拼接，得到一个新字符串。
返回以这种方式能够构成的最长回文串的长度。
 
示例 1：

输入：s = "a", t = "a"
输出：2
解释：
将 s 中的 "a" 与 t 中的 "a" 拼接，得到 "aa"，它是长度为 2 的回文串。

示例 2：

输入：s = "abc", t = "def"
输出：1
解释：
由于所有字符都不相同，最长的回文串就是任意单个字符，所以答案为 1。

示例 3：

输入：s = "b", t = "aaaa"
输出：4
解释：
选取 t 中的 "aaaa" 即为最长回文串，长度为 4。

示例 4：

输入：s = "abcde", t = "ecdba"
输出：5
解释：
将 s 中的 "abc" 与 t 中的 "ba" 拼接，得到 "abcba"，它是长度为 5 的回文串。

 
约束条件：

1 …（共 974 字符，已截断）
````
答案 / 判分依据：
````
无参考解；公开测试示例 [{"input": "\"a\"\n\"a\"", "output": "2", "testtype": "functional"}]
````

**lcb/abc399_d**

题目：
````
N 对情侣坐成一排。
请统计满足下列条件的情侣对的数量：两对情侣原本都没有相邻而坐，并且通过在这四个人之间交换座位，两对情侣最终都能相邻而坐。

给定一个长度为 2N 的序列 A = (A_1, A_2, \dots, A_{2N})。整数 1, 2, \dots, N 在 A 中各恰好出现两次。
求满足 1 \leq a < b \leq N 且满足以下所有条件的整数对 (a, b) 的个数：

- A 中 a 的两次出现位置不相邻。
- A 中 b 的两次出现位置不相邻。
- 按任意顺序执行下述操作一次或多次，可以达到 A 中 a 的两次出现位置相邻、且 b 的两次出现位置也相邻的状态。
- 选择一个整数对 (i, j)（1 \leq i \leq 2N, 1 \leq j \leq 2N），使得 A_i = a 且 A_j = b，并交换 A_i 与 A_j …（共 2113 字符，已截断）
````
答案 / 判分依据：
````
无参考解；公开测试示例 [{"input": "3\n3\n1 2 3 3 1 2\n4\n1 1 2 2 3 3 4 4\n5\n1 2 3 4 5 1 2 3 4 5", "output": "1\n0\n4", "testtype": "stdin"}]
````

## 通用推理 / 知识

### 训练集：triviaqa_train（pool，73225 条）

**triviaqa/qg_1424**

题目：
````
1799 年由法国人发现、对破译埃及象形文字起到关键作用、后来其名称又被用作一款计算机语言学习软件品牌的古埃及文物是什么？
````
答案 / 判分依据：
````
["罗塞塔石碑（Rosetta Stone）", "Stone of Rosetta", "Pierre de Rosette", "Rossetta Stone", "Rosetta inscription", "Rashid Stone", "The Rosetta Stone", "Roseta stone", "Rosetta stone", "Rosseta Stone", "Decree of Memphis (Ptolemy V)"]
````

**triviaqa/qw_5707**

题目：
````
被称为圣马力诺的国家，其正式名称是什么？
````
答案 / 判分依据：
````
["最尊贵的圣马力诺共和国（The Most Serene Republic of San Marino）", "Capitani reggenti", "Land of San Marino", "San Marino", "San Marino/Transnational issues", "Liberation from the Alberoni Occupation", "Capitano reggente", "Most Serene Republic of San Marino", "Republic of S.Marino", "Public holidays in San Marino", "Climate of San Marino", "Sammarinese", "Anniversary of the Arengo and the Feast of the Militants", "Culture of San Marino", "San Marino Republic", "ISO 3166-1:SM", "Liberation Day (San Marino)", "Republic of San Marino", "Feast of San Marino and the Republic", "Saint-Marin", "San marino", "Europe/San Mar …（共 648 字符，已截断）
````

**triviaqa/jp_2832**

题目：
````
普拉达（Prada）总部位于哪座欧洲城市？
````
答案 / 判分依据：
````
["米兰（Milan）", "The weather in Milan", "The weather in Milano", "UN/LOCODE:ITMIL", "Milan, Italy", "Comune di Milano", "Miláno", "Milanesi", "Mediolani", "Lordship of Milan", "History of Milan", "Milanese duchy", "Milano", "Milan (Italy)", "Milan, Spain", "Ambrosiani", "Meneghini", "Milao", "Milano, Italy", "Milão", "Milán", "Millaine"]
````

**triviaqa/qb_1003**

题目：
````
克里米亚的塞瓦斯托波尔位于哪片水域的沿岸？
````
答案 / 判分依据：
````
["黑海（Black Sea）", "Marea Neagră", "Black Sea shelf", "Black Sea littoral", "Черноморский", "Black sea", "შავი ზღვის", "შავი ზღვა", "Euxine", "Чорноморський", "The Black Sea", "Black seas", "Schwarzes Meer", "Pontus Euxinus", "Pontus Euxinos", "Euxine Sea", "Kara Deniz", "Черно море"]
````

**triviaqa/qg_2688**

题目：
````
哪家公司使用“出门别忘带上它”（don’t leave home without it）这句广告语？
````
答案 / 判分依据：
````
["美国运通（American Express）", "Starwood Preferred Guest Credit Card from American Express", "Aexp.co", "American Express Co", "American Express Bank Ltd", "AMEX Bank of Canada", "Platinum Card", "American Express Bank", "American Express Bank Ltd.", "AmericanExpress.com", "Amex Bank of Canada", "AMERICAN EXPRESS COMPANY", "American Express Card", "Ed Gilligan", "American Express Publishing", "American Express Bank Limited", "Blue From American Express", "Americanexpress.com", "American Express Bank International", "The American Express Company", "American express travel", "American Express Company", "C …（共 802 字符，已截断）
````

**triviaqa/sfq_19887**

题目：
````
西班牙哪个地区的名称在英语中意为“非常坚硬”（very hard）？
````
答案 / 判分依据：
````
["埃斯特雷马杜拉（Extremadura）", "Extramadura", "Estremaua", "Extremadura, Spain", "Estremaúra", "Extremadura (Spain)", "Santa Marta de los Barros", "History of Extremadura"]
````

**triviaqa/sfq_2772**

题目：
````
除太阳外，距离地球最近的恒星是哪颗？
````
答案 / 判分依据：
````
["比邻星（Proxima Centauri）", "Gl 551", "GCTP 3278.00", "V645 Centauri", "Proxima centauri", "Proxima Centauri C", "Nearest star to our planet", "LHS 49", "Alpha Centauri C", "Closest star", "Rigil Kent C", "Proxima Centari", "Proxima Centaurus", "Alpha Proxima", "Proxima Centuri", "Gliese 551", "Alpha proxima"]
````

**triviaqa/odql_14381**

题目：
````
哪位作家写了一系列以私家侦探“V.I. 沃肖斯基”（V.I. Warshawsky）为主角的书？
````
答案 / 判分依据：
````
["萨拉·帕雷茨基（SARAH PARETSKY）", "Sarah Paretsky", "Sara Paretsky", "Anna Piretsky"]
````

**triviaqa/odql_15022**

题目：
````
"""Say what you want"" (1997) 和 ""In our lifetime"" (1998) 是哪个乐队的热门歌曲？"
````
答案 / 判分依据：
````
["德州乐队（Texas）", "Texas (U.S. State)", "US-TX", "Texos", "Lone Star State", "Texas, USA", "Religion in Texas", "Texas (state)", "Lake Ozark, Texas", "Christianity in Texas", "The State of Texas", "Twenty-eighth State", "Texas-sized", "Texas (State)", "28th State", "The Lone Star State", "Twenty-Eighth State", "Texas, United States", "Texas (U.S. state)", "Texass", "State of Texas", "Everything is bigger in Texas", "28th state"]
````

**triviaqa/bt_595**

题目：
````
哪首歌在 1972 年是迈克尔·杰克逊（Michael Jackson）的前十名金曲，在 1985 年又是玛蒂·韦布（Marti Webb）的前十名金曲？
````
答案 / 判分依据：
````
["《Ben》（Ben）"]
````

### 训练集：arc（pool，2870 条）

**arc/ARC-Easy/Mercury_SC_LBS10351**

题目：
````
星座的外观在夜间如何变化？

选项：
A. 它的形状看起来变得更长。
B. 它的位置相对于地平线看起来发生了移动。
C. 它最亮的恒星开始呈现橙色。
D. 它与北极星的距离减小。
````
答案 / 判分依据：
````
B
````

**arc/ARC-Easy/ACTAAP_2012_7_1**

题目：
````
人体的哪个器官系统与环境进行气体交换？

选项：
A. 内分泌系统
B. 循环系统
C. 呼吸系统
D. 皮肤系统
````
答案 / 判分依据：
````
C
````

**arc/ARC-Easy/Mercury_7015943**

题目：
````
气态二氧化硫是一种化合物，在大气中与水结合形成酸雨。二氧化硫的主要来源是什么？

选项：
A. 火山喷发物
B. 化石燃料的燃烧
C. 热带森林的破坏
D. 采矿和矿物开采
````
答案 / 判分依据：
````
B
````

**arc/ARC-Easy/Mercury_7181318**

题目：
````
当水从地球表面蒸发进入大气时，其分子进入气态。这种气体发生什么变化会使分子凝结成形成云的小水滴？

选项：
A. 质量减小。
B. 体积增大。
C. 温度降低。
D. 压强增大。
````
答案 / 判分依据：
````
C
````

**arc/ARC-Easy/Mercury_7075198**

题目：
````
当多细胞生物受到损伤时，它会通过以下方式进行自我修复：

选项：
A. 它发生突变。
B. 细胞协同工作。
C. 细胞移动。
D. 它自己制造食物。
````
答案 / 判分依据：
````
B
````

**arc/ARC-Easy/Mercury_SC_400165**

题目：
````
对于玩具车在木地板上比在厚地毯上滚得更远，最合理的解释是什么？

选项：
A. 车在地毯上更重。
B. 车在地板上更重。
C. 地毯的阻力更大。
D. 地板的牵引力更大。
````
答案 / 判分依据：
````
C
````

**arc/ARC-Challenge/MCAS_2005_9_19**

题目：
````
一名学生站在静止不动的滑板上。学生和滑板的总质量为 50 千克。学生以 5 m/s 的速度向前扔出一个质量为 2 千克的球。假设滑板轮子无摩擦，学生和滑板将如何运动？

选项：
A. 以 0.4 m/s 向前
B. 以 5 m/s 向前
C. 以 0.2 m/s 向后
D. 以 5 m/s 向后
````
答案 / 判分依据：
````
D
````

**arc/ARC-Easy/MCAS_2003_8_20**

题目：
````
哪个界包含多细胞、没有叶绿素、并从腐烂组织中吸收营养的生物？

选项：
A. 真菌界
B. 植物界
C. 原生生物界
D. 动物界
````
答案 / 判分依据：
````
A
````

**arc/ARC-Easy/Mercury_7041073**

题目：
````
什么最能调节过热身体的温度？

选项：
A. 出汗减少
B. 呼吸频率降低
C. 皮肤血管扩张
D. 心率加快
````
答案 / 判分依据：
````
C
````

**arc/ARC-Challenge/Mercury_7041633**

题目：
````
下列人类活动中，哪一项不会导致物种灭绝？

选项：
A. 狩猎
B. 栖息地破坏
C. 恢复生态学
D. 引入外来物种
````
答案 / 判分依据：
````
C
````

### 训练集：gpqa_main（gate，250 条）

（GPQA：受访问条款限制，不展示题目。）

### 评测集：gpqa_diamond（test，198 条）

（GPQA：受访问条款限制，不展示题目。）

### 评测集：mmlu_pro（test，2000 条）

**mmlu_pro/4586**

题目：
````
对于电池 Pt/Ag(s)/Ag Br(s)/Br^-(a = .34)/Br_2(1atm)/Pt，求 E° 的值并判断该反应的自发性。已知半电池电极电势：AgBr/Ag 为 0.071，B_2/Br^- 为 1.065。

选项：
A. 0.071，自发
B. -0.994，自发
C. -0.994，非自发
D. 1.136，自发
E. 1.065，非自发
F. 0.994，非自发
G. -1.136，非自发
H. -0.071，自发
I. 0.071，非自发
J. 0.994，自发
````
答案 / 判分依据：
````
J
````

**mmlu_pro/12086**

题目：
````
计算半径为 R、带电量为 q 的薄球形导体壳的总静电势能。

选项：
A. {2q^2 / (4\pi\epsilon_0R)} 焦耳
B. {q^2 / (2\pi\epsilon_0R)} 焦耳
C. {q^2R / (4\pi\epsilon_0)} 焦耳
D. {q^2 / (4\pi\epsilon_0)} 焦耳
E. {q^2 / (16\pi\epsilon_0R)} 焦耳
F. {q^2 / (4\pi\epsilon_0R^2)} 焦耳
G. {q^3 / (8\pi\epsilon_0R^2)} 焦耳
H. {q^2 / (8\pi\epsilon_0R)} 焦耳
I. {q^2 / (π\epsilon_0R)} 焦耳
J. {q^2 / (4\pi\epsilon_0R)} 焦耳
````
答案 / 判分依据：
````
H
````

**mmlu_pro/8779**

题目：
````
下列关于简单随机样本的说法中，哪一项是错误的？

选项：
A. 总体的大小会影响样本的随机性。
B. 总体中的每个元素被抽中的机会均等。
C. 样本必须足够大才能被正确地视为简单随机样本。
D. 简单随机样本的特征可能与总体的特征相差很大。
E. 仅检查样本无法判断它是否为简单随机样本。
````
答案 / 判分依据：
````
C
````

### 评测集：triviaqa（test，2000 条）

**triviaqa/bb_7406**

题目：
````
2007 年谁成为了法国新任总统？
````
答案 / 判分依据：
````
["尼古拉·萨科齐（Nicolas Sarkozy）", "Hyperpresident", "Nicolas Paul Stéphane Sarkozy de Nagy-Bocsa", "Sarko the American", "Nick sarkozy", "Sarkosy", "Sarzoky", "Nicolas sarkosy", "Nicolas Sarközy", "Sarko", "Nick sarkozee", "Giulia Sarkozy", "Sarkozy", "Nicolas Sarkoezy", "Nicholas Sarkozy", "Nicolas Paul Stephane Sarkozy de Nagy-Bocsa", "Nikolas sarkozy", "Nicholas Sarzoky", "Nicolas Paul Stéphane Sarközy de Nagy-Bocsa", "President Sarkozy", "Sarcozy", "Sarko l'Americain", "Nicolas Paul Stephane Sarkoezy de N …（共 639 字符，已截断）
````

**triviaqa/qb_1757**

题目：
````
新西兰国旗上有几颗星？
````
答案 / 判分依据：
````
["四（Four）", "four", "4"]
````

**triviaqa/bb_5924**

题目：
````
哪个表示爬行动物的古拉丁词，与现代蛇类的科学亚目分类名称非常相似？
````
答案 / 判分依据：
````
["蛇亚目（Serpentes）", "Snkae", "Snakes", "Serpentigenous", "Snake anatomy", "🐍", "Snakey", "Snake diet", "Snake locomotion", "Serpentiform", "Snake", "Snake food", "Snake (zoology)", "Coronellidae", "Snakiest"]
````

## 指令遵循

说明：约束（如全小写、关键词次数）是针对英文原文判分的，中文译文仅供阅读。

### 训练集：ifeval_train（pool，13954 条）

**rlvr_ifeval/11666**

题目：
````
撰写一个恰当完成该请求的回答。

### Instruction:
编写一个 JSON 文档，用于存储游戏中玩家的姓名和得分。

### Response: 回答最多使用 19 句话。
````
答案 / 判分依据：
````
{"func_name": "verify_sentence_constraint", "N": 19, "quantifier": "at most"}
````

**rlvr_ifeval/4004**

题目：
````
写一首不超过 10 行、表达感激之情的诗。

 在你的回答中包含关键词 deracialize, fermery。
````
答案 / 判分依据：
````
{"func_name": "verify_keywords", "keyword_list": ["deracialize", "fermery"]}
````

**rlvr_ifeval/14184**

题目：
````
针对下面这篇文章写一个问题：冲突造成的死亡人数十分巨大。克伦威尔派的威廉·配第（William Petty）在 1650 年代主持了爱尔兰首次科学的土地与人口调查，他得出结论：1641 年至 1653 年间，爱尔兰至少有 40 万人、最多可能有 62 万人死亡。真实数字也许更低，但最低的估计也约为 20 万。当时爱尔兰人口仅约 150 万。据估计，约三分之二的死者是平民。爱尔兰的失败导致天主教徒所有的土地被大规模没收，并使英格兰新教徒在爱尔兰的统治持续了两个多世纪。这些战争，尤其是 t …（共 1195 字符，已截断）
````
答案 / 判分依据：
````
{"func_name": "validate_no_commas"}
````

**rlvr_ifeval/6604**

题目：
````
问题：信息：  - 莱昂·博莱（Léon Bollée，1870 年 4 月 1 日－1913 年 12 月 16 日）是法国汽车制造商和发明家。  - 鲁塞尔（Roussel）是 1908 年至 1914 年间生产的一款法国汽车。该公司在沙勒维尔-梅齐耶尔的一家工厂生产轻型汽车、小型汽车（voiturette）和出租车；提供四缸 10 马力和 12 马力发动机。  - 沙勒维尔-梅齐耶尔是法国北部的一个市镇，是大东部大区阿登省的首府。沙勒维尔-梅齐耶尔位于默兹河畔。  - voiturette 是一种微型汽车。历史。"Voiturette" 一词由莱昂·博莱于 1895 年首次注册，用来命名他的新型机动三轮车。该词变得如此流行，以至于 e …（共 1278 字符，已截断）
````
答案 / 判分依据：
````
{"func_name": "verify_letter_frequency", "N": 23, "letter": "c"}
````

**rlvr_ifeval/5110**

题目：
````
两个表亲相隔一万英里，但常常谈起要一起去夏威夷群岛游玩。莉莉住在缅因州，而希瑟住在夏威夷。  哪位表亲离卡霍奥拉维岛更远？
----
Answer: Lilly


Q: 基思是市医院的一名医生。今天他要接诊六位患者：患者 A、患者 B、患者 C、患者 D、患者 E 和患者 F。患者 A 吸烟，但患者 B 不吸烟。患者 C 使用防晒霜，但患者 D 不使用任何防晒霜。患者 E 使用日光浴灯晒黑，而患者 F 在阳光下晒黑。  患者 E 和患者 C 相比，哪位更容易罹患皮肤癌？
A: patient E


Question: 迈克正在阅读一篇关于 an …（共 2388 字符，已截断）
````
答案 / 判分依据：
````
{"func_name": "verify_keyword_frequency", "N": 16, "word": "coadjutorship"}
````

**rlvr_ifeval/276**

题目：
````
在你的回答中，字母 f 应出现 14 次。给定一个科学问题，生成一个正确的答案。

Example Input: Cody's velocity is zero so therefore he doesn't have what?
Example Output: momentum

Example Input: What are the contacts between neurons called?
Example Output: synapses

Example Input: What will happen to sea levels due to increased global temperatures?
Example Output:
````
（注：上面的示例输入输出为英文原文，译文：科迪的速度为零，所以他没有什么？→ momentum（动量）；神经元之间的接触点叫什么？→ synapses（突触）；全球气温升高会使海平面发生什么变化？）

答案 / 判分依据：
````
{"func_name": "verify_letter_frequency", "N": 14, "letter": "f"}
````

**rlvr_ifeval/2603**

题目：
````
回答中不要包含关键词 feminin, unripeness, untechnically。写一张解雇马特的便条，但要写得像阿弗罗曼（Afroman）在唱 "because I got high"（因为我嗨了）。
````
答案 / 判分依据：
````
{"func_name": "validate_forbidden_words", "forbidden_words": ["feminin", "unripeness", "untechnically"]}
````

**rlvr_ifeval/8212**

题目：
````
test: Exprimând sprijinul ţării sale faţă de găsirea rapidă a unei soluţii la statutul provinciei Kosovo, Subsecretarul american de Stat pentru Afaceri Politice, Nicholas Burns, a declarat miercuri că menţinerea acestei situaţii este inacceptabilă.
English?（译：测试：[上述罗马尼亚语句子]。英文？）

translation: Voicing his country's support for a rapid solution to the Kosovo status issue, US Undersecretary of State for Political Affairs Nicholas Burns said Wednesday the continuing uncertainty is no longer acceptable.（译：表达其国家对迅速解决科索沃地位问题的支持，美国负责政治事务的副国务卿尼古拉斯·伯恩斯周三表示，持续的不确定状态已不可接受。）
De asemenea, trebuie să comparăm în ce fel de mediu operăm în Europa.（罗马尼亚语：我们还必须比较我们在欧洲身处何种环境。）

In English?（用英文？）
xxxxx
We also have to compare in what kind of environment in Europe we are operating.
How is "The late dictator favoured a strongly cen …（共 1628 字符，已截断）
````
答案 / 判分依据：
````
{"func_name": "validate_json_format"}
````

**rlvr_ifeval/7551**

题目：
````
用双引号将你的整个回答括起来。阅读下面的文章并选出最佳答案。文章：你想成为一名艺术家吗（  ）你想成为一名艺术家吗？来我们的俱乐部吧，你可以在快乐中找到自己。我们开设吉他、鼓、小提琴和钢琴课程，每门只需 Y=20。你还可以学唱歌、学跳舞，每门 Y=25。如果你喜欢美术，你也会感到满意——每门只需 Y=30！问题：如果你想学吉他和舞蹈，你应该支付   _  。- Y=45 - Y=20 - Y=30 - Y=25
Answer:
````
答案 / 判分依据：
````
{"func_name": "validate_quotation"}
````

**rlvr_ifeval/8310**

题目：
````
应当有 3 个段落。段落之间且仅段落之间以两个换行符分隔。第 1 个段落必须以单词 embed 开头。test: Testo d'Entrata Bequem mit öffentlichen Verkehrsmitteln zu erreichen, in strategisch günstiger Lage zur Messe, dem Mazdapalace und dem Gebiet San Siro gelegen, erlaubt es Ihnen das Hotel Bonola, sich einfach und in aller Freiheit durch Mailand zu bewegen.
English?（译：测试：[上述意大利语/德语文本]。英文？）

translation: AT HOTEL BONOLA TICKET FOR MACHT OF MILAN AND INTER AT SAN SIRO STADIUM.（译：在 BONOLA 酒店可购买圣西罗球场 AC 米兰与国际米兰比赛的门票。）

test: Die Maschinen der Serie AT S sind mit einer nicht angetriebenen Rollenbahn ausgestattet (auf Anfrage sind die Rollen PVC beschichtet), sie sind einfach und flexibel in der Handh …（共 1469 字符，已截断）
````
答案 / 判分依据：
````
{"func_name": "validate_paragraphs", "N": 3, "i": 1, "first_word": "embed"}
````

### 训练集：ifeval（gate，135 条）

**ifeval/2071**

题目：
````
你能给我一个适合儿童的《指环王》简短概述吗？（Could you give me a short summary of The Lord of the Rings that is child-friendly?）
首先，逐字不改地重复 "Could you give me a short summary of The Lord of the Rings that is child-friendly?"（你能给我一个适合儿童的《指环王》简短概述吗？），然后再给出你的回答。不要先说任何别的话，只在最开头重复该请求。
````
答案 / 判分依据：
````
{"instruction_id_list": ["combination:repeat_prompt"], "kwargs": [{"prompt_to_repeat": "Could you give me a short summary of The Lord of the Rings that is child-friendly?"}]}
````

**ifeval/1372**

题目：
````
撰写一篇关于里普利水族馆（Ripley's Aquarium）的博客文章。确保你的博客文章至少包含 7 个用方括号表示的占位符，例如 [location]。
````
答案 / 判分依据：
````
{"instruction_id_list": ["detectable_content:number_placeholders"], "kwargs": [{"num_placeholders": 7}]}
````

**ifeval/3414**

题目：
````
你能详细阐述一下 "I froze when I was jogging"（我慢跑时冻僵了）吗？包含一些全大写的单词。具体来说，这类大写单词应有 5 到 10 个。
````
答案 / 判分依据：
````
{"instruction_id_list": ["change_case:capital_word_frequency", "change_case:capital_word_frequency"], "kwargs": [{"capital_relation": "at least", "capital_frequency": 5}, {"capital_relation": "less than", "capital_frequency": 11}]}
````

### 评测集：ifeval（test，271 条）

**ifeval/1939**

题目：
````
我是一名新手养狗人，想请教如何训练我的小狗。你能帮我吗？给我几个选项。特别要求你用 "Which one you choose?"（你选哪一个？）来结束你的回答。
````
答案 / 判分依据：
````
{"instruction_id_list": ["startend:end_checker"], "kwargs": [{"end_phrase": "Which one you choose?"}]}
````

**ifeval/1108**

题目：
````
汉堡是三明治吗？请只使用卡纳达语回答，不允许使用其他任何语言。
````
答案 / 判分依据：
````
{"instruction_id_list": ["language:response_language"], "kwargs": [{"language": "kn"}]}
````

**ifeval/1219**

题目：
````
运动鞋哪个品牌更好：Prada 还是 Nike？你的整个回答应使用英文，并且全部大写。在回答末尾，请明确添加一条以 P.S. 开头的附言。单词 sneaker 应在你的回答中出现 10 次或更多次。
````
答案 / 判分依据：
````
{"instruction_id_list": ["change_case:english_capital", "detectable_content:postscript", "keywords:frequency"], "kwargs": [{}, {"postscript_marker": "P.S."}, {"relation": "at least", "keyword": "sneaker", "frequency": 10}]}
````

### 评测集：ifbench（test，300 条）

**ifbench/5**

题目：
````
一个质量为 m 的粒子只能沿着一条无摩擦的、竖直的、半径为 a 的圆形轨道运动。(a) 忽略重力，将该质点的经典能量写成角动量的函数，并假设量子力学哈密顿量由代入量子角动量算符得到。哈密顿算符是什么？(b) 写出位置表象下的薛定谔方程。角度 φ 应当是其中的核心变量。(c) 求位置表象下的能量本征值和能量本征函数。在回答中至少使用 5 个不同的并列连词。
````
答案 / 判分依据：
````
{"instruction_id_list": ["count:conjunctions"], "kwargs": [{"small_n": 5.0}]}
````

**ifbench/123**

题目：
````
包含引号嵌套引号再嵌套引号的内容，至少 3 层深，双引号与单引号交替使用。方程 $x_1 + x_2 + x_3 + x_4 = 20$ 有多少个解，其中对所有 i，$x_i$ 为非负整数，如果：(a) 没有任何限制？(b) $x_1 > 1$，$x_2 > 2$，$x_3 > 3$，$x_4 > 4$？(c) $x_1 > 4$ 且 $x_3 < 5$？
````
答案 / 判分依据：
````
{"instruction_id_list": ["format:quotes"], "kwargs": [{}]}
````

**ifbench/172**

题目：
````
python 从文件路径中提取文件名。确保停用词占你回答总词数的比例不超过 23%。
````
答案 / 判分依据：
````
{"instruction_id_list": ["ratio:stop_words"], "kwargs": [{"percentage": 23.0}]}
````

## 工具调用

说明：参数值按英文原文判分，译文仅供阅读。

### 训练集：sealtool_train（pool，11019 条）

**sealtool/train-difficult-4154**

题目：
````
请根据 task_instruction 选择所需的 API，并返回 api_calling 列表。
输出格式：[{"api": "", "parameters": {"": ""}, "responses": ["API_call_0","API_call_1"]},{"api": "", "parameters": {"": ""}, "responses": ["API_call_2"]}]
responses 可以用作参数值。responses 的数量取决于 api_list 中的信息。

Input:
api_list = [{'api_name': 'calculateNoiseFigure', 'api_description': '计算噪声系数', 'field': 'Electrical Engineering/RF and Microwave Engineering', 'parameters': {'frequency': {'type': 'float', 'description': '需要计算噪声系数的频率（例如 1 GHz、10 GHz）'}, 'temperature': {'type': 'float', 'de …（共 3570 字符，已截断）
````
答案 / 判分依据：
````
[{"api": "getBirdDiet", "parameters": {"species": "parrot"}, "responses": ["API_call_0"]}, {"api": "calculateNoiseFigure", "parameters": {"frequency": 1000000000.0, "temperature": 300}, "responses": ["API_call_1"]}, {"api": "detectFoodContamination", "parameters": {"sample": "banana"}, "responses": ["API_call_2", "API_call_3"]}]
````

**sealtool/train-difficult-4749**

题目：
````
请根据 task_instruction 选择所需的 API，并返回 api_calling 列表。
输出格式：[{"api": "", "parameters": {"": ""}, "responses": ["API_call_0","API_call_1"]},{"api": "", "parameters": {"": ""}, "responses": ["API_call_2"]}]
responses 可以用作参数值。responses 的数量取决于 api_list 中的信息。

Input:
api_list = [{'api_name': 'getInsulinDosage', 'api_description': '获取患者的胰岛素剂量建议', 'field': 'Healthcare/Endocrinology', 'parameters': {'patient_id': {'type': 'str', 'description': '患者的唯一 ID'}, 'meal_period': {'type': 'str', 'description': 'Specify a meal period for insulin dosag …（共 4323 字符，已截断）
````
答案 / 判分依据：
````
[{"api": "getHematologyResults", "parameters": {"patient_id": "UId7i9E4Hzni", "species": "dog"}, "responses": ["API_call_0", "API_call_1", "API_call_2", "API_call_3"]}, {"api": "getAthleteInformation", "parameters": {"athlete_name": "Michael Phelps"}, "responses": ["API_call_4", "API_call_5", "API_call_6", "API_call_7"]}, {"api": "getInsulinDosage", "parameters": {"patient_id": "LH4fnkIB5F", "meal_period": "breakfast", "blood_glucose_level": 29.1}, "responses": ["API_call_8"]}]
````

**sealtool/train-difficult-9616**

题目：
````
请根据 task_instruction 选择所需的 API，并返回 api_calling 列表。
输出格式：[{"api": "", "parameters": {"": ""}, "responses": ["API_call_0","API_call_1"]},{"api": "", "parameters": {"": ""}, "responses": ["API_call_2"]}]
responses 可以用作参数值。responses 的数量取决于 api_list 中的信息。

Input:
api_list = [{'api_name': 'getResearchParticipants', 'api_description': '获取研究参与者的信息', 'field': 'Research/Qualitative Research', 'parameters': {'study_id': {'type': 'int', 'description': '研究项目的 ID'}, 'participant_id': {'type': 'int', 'description': '参与者的 ID'}}, …（共 3503 字符，已截断）
````
答案 / 判分依据：
````
[{"api": "performQualityControl", "parameters": {"data": "temperature"}, "responses": ["API_call_0", "API_call_1"]}, {"api": "getResearchParticipants", "parameters": {"study_id": 93, "participant_id": 5}, "responses": ["API_call_2", "API_call_3"]}, {"api": "analyzeCallQuality", "parameters": {"phone_number": "+1234567890", "duration": 120}, "responses": ["API_call_4"]}]
````

**sealtool/train-easy-1800**

题目：
````
请根据 task_instruction 选择所需的 API，并返回 api_calling 列表。
输出格式：[{"api": "", "parameters": {"": ""}, "responses": ["API_call_0","API_call_1"]},{"api": "", "parameters": {"": ""}, "responses": ["API_call_2"]}]
responses 可以用作参数值。responses 的数量取决于 api_list 中的信息。

Input:
api_list = [{'api_name': 'getFlightData', 'api_description': '从航空电子系统获取航班数据', 'field': 'Aviation/Avionics', 'parameters': {'flight_number': {'type': 'str', 'description': '分配给特定航班的编号（例如 AA123、DL456）'}, 'aircraft': {'type': 'str', 'description': 'The aircraft model or code …（共 4385 字符，已截断）
````
答案 / 判分依据：
````
[{"api": "getFlightData", "parameters": {"flight_number": "AA123", "aircraft": "A320"}, "responses": ["API_call_0", "API_call_1", "API_call_2"]}]
````

**sealtool/train-difficult-4153**

题目：
````
请根据 task_instruction 选择所需的 API，并返回 api_calling 列表。
输出格式：[{"api": "", "parameters": {"": ""}, "responses": ["API_call_0","API_call_1"]},{"api": "", "parameters": {"": ""}, "responses": ["API_call_2"]}]
responses 可以用作参数值。responses 的数量取决于 api_list 中的信息。

Input:
api_list = [{'api_name': 'getPsychiatricNursingPatient', 'api_description': '获取精神科护理患者的信息', 'field': 'Nursing/Psychiatric Nursing', 'parameters': {'patient_id': {'type': 'str', 'description': '患者的 ID（例如 John123、Mary456）'}, 'admission_date': {'type': 'str', 'description': …（共 4629 字符，已截断）
````
答案 / 判分依据：
````
[{"api": "getBrainActivity", "parameters": {"subject": "human", "region": "prefrontal cortex"}, "responses": ["API_call_0", "API_call_1"]}, {"api": "analyzeTargetAudience", "parameters": {"campaign_type": "raising awareness about a disease", "target_population": "patients with diabetes"}, "responses": ["API_call_2"]}, {"api": "getPsychiatricNursingPatient", "parameters": {"patient_id": "John123", "admission_date": "2022-05-10", "care_team": "Team A"}, "responses": ["API_call_3", "API_call_4", "API_call_5", "API_call_6"]}]
````

**sealtool/train-difficult-7467**

题目：
````
请根据 task_instruction 选择所需的 API，并返回 api_calling 列表。
输出格式：[{"api": "", "parameters": {"": ""}, "responses": ["API_call_0","API_call_1"]},{"api": "", "parameters": {"": ""}, "responses": ["API_call_2"]}]
responses 可以用作参数值。responses 的数量取决于 api_list 中的信息。

Input:
api_list = [{'api_name': 'getBehaviorAdvice', 'api_description': '获取针对特定兽医问题的行为建议', 'field': 'Veterinary Science/Veterinary Behavior', 'parameters': {'issue': {'type': 'str', 'description': 'The specific veterinary issue for which you need behavior advice (e.g., separation anxiety, aggression, …（共 3917 字符，已截断）
````
答案 / 判分依据：
````
[{"api": "managePain", "parameters": {"patient_id": "ID123", "location": "head", "intensity": 8.5}, "responses": ["API_call_0", "API_call_1"]}, {"api": "getBehaviorAdvice", "parameters": {"issue": "separation anxiety"}, "responses": ["API_call_2"]}, {"api": "getPetBloodSugarLevel", "parameters": {"pet_name": "Rex"}, "responses": ["API_call_3"]}]
````

**sealtool/train-difficult-8774**

题目：
````
请根据 task_instruction 选择所需的 API，并返回 api_calling 列表。
输出格式：[{"api": "", "parameters": {"": ""}, "responses": ["API_call_0","API_call_1"]},{"api": "", "parameters": {"": ""}, "responses": ["API_call_2"]}]
responses 可以用作参数值。responses 的数量取决于 api_list 中的信息。

Input:
api_list = [{'api_name': 'getFacilityLocation', 'api_description': '获取医疗机构的位置信息', 'field': 'Architecture/Healthcare architecture', 'parameters': {'facility_id': {'type': 'str', 'description': '医疗机构的唯一标识符'}}, 'required': ['facility_id'], 'responses': { …（共 3877 字符，已截断）
````
答案 / 判分依据：
````
[{"api": "getProjectSchedule", "parameters": {"project_id": "abC123"}, "responses": ["API_call_0"]}, {"api": "getBuildingIndoorAirQuality", "parameters": {"building_name": "Sydney Opera House", "time_of_day": "morning"}, "responses": ["API_call_1"]}, {"api": "getFacilityLocation", "parameters": {"facility_id": "xyz789"}, "responses": ["API_call_2", "API_call_3", "API_call_4"]}]
````

**sealtool/train-difficult-6944**

题目：
````
请根据 task_instruction 选择所需的 API，并返回 api_calling 列表。
输出格式：[{"api": "", "parameters": {"": ""}, "responses": ["API_call_0","API_call_1"]},{"api": "", "parameters": {"": ""}, "responses": ["API_call_2"]}]
responses 可以用作参数值。responses 的数量取决于 api_list 中的信息。

Input:
api_list = [{'api_name': 'modifyDatabase', 'api_description': '修改现有数据库的配置', 'field': 'Cloud Computing/Database as a Service (DBaaS)', 'parameters': {'database_name': {'type': 'str', 'description': '待修改数据库的名称'}, 'new_storage_size': {'type': 'int', 'description': 'The n …（共 5038 字符，已截断）
````
答案 / 判分依据：
````
[{"api": "getProductInfo", "parameters": {"product_id": "ABC123", "location": "aisle E, shelf F"}, "responses": ["API_call_0", "API_call_1", "API_call_2", "API_call_3", "API_call_4"]}, {"api": "modifyDatabase", "parameters": {"database_name": "customers_db", "new_storage_size": 1000}, "responses": ["API_call_5"]}, {"api": "createAdvertisement", "parameters": {"campaign_name": "Summer Sale", "target_audience": "young adults", "budget": 10.0, "platform": "Google Ads"}, "responses": ["API_call_6", "API_call_7"]}]
````

**sealtool/train-difficult-6318**

题目：
````
请根据 task_instruction 选择所需的 API，并返回 api_calling 列表。
输出格式：[{"api": "", "parameters": {"": ""}, "responses": ["API_call_0","API_call_1"]},{"api": "", "parameters": {"": ""}, "responses": ["API_call_2"]}]
responses 可以用作参数值。responses 的数量取决于 api_list 中的信息。

Input:
api_list = [{'api_name': 'getShipLocation', 'api_description': '获取船舶当前所在位置', 'field': 'Transportation/Water Transportation', 'parameters': {'ship_id': {'type': 'str', 'description': '要获取位置的船舶 ID'}}, 'required': ['ship_id'], 'responses': {'latitude': {'type': 'f …（共 4373 字符，已截断）
````
答案 / 判分依据：
````
[{"api": "getMicroNanoRobotStatus", "parameters": {"robot_name": "NanoBot 2000"}, "responses": ["API_call_0"]}, {"api": "getShipLocation", "parameters": {"ship_id": "qoy4wiOhS"}, "responses": ["API_call_1", "API_call_2"]}, {"api": "getMicrobialGrowth", "parameters": {"product_name": "Antibiotic", "batch_number": "B321", "sampling_time": "post-production", "test_type": "Microbial limit testing"}, "responses": ["API_call_3", "API_call_4"]}]
````

**sealtool/train-difficult-4823**

题目：
````
请根据 task_instruction 选择所需的 API，并返回 api_calling 列表。
输出格式：[{"api": "", "parameters": {"": ""}, "responses": ["API_call_0","API_call_1"]},{"api": "", "parameters": {"": ""}, "responses": ["API_call_2"]}]
responses 可以用作参数值。responses 的数量取决于 api_list 中的信息。

Input:
api_list = [{'api_name': 'getLibraryTechnologyPlan', 'api_description': '获取图书馆技术规划信息', 'field': 'Library Science/Library Technology Planning', 'parameters': {'library_name': {'type': 'str', 'description': '要获取技术规划的图书馆名称'}, 'budget': {'type …（共 4262 字符，已截断）
````
答案 / 判分依据：
````
[{"api": "getDigitalTransformation", "parameters": {"country": "Germany"}, "responses": ["API_call_0"]}, {"api": "updateApplicationVersion", "parameters": {"application_name": "2004-04-29", "new_version": "The new version of the application introduces new features"}, "responses": ["API_call_1"]}, {"api": "getLibraryTechnologyPlan", "parameters": {"library_name": "Matplotlib", "budget": 36.1, "timeline": "3 years"}, "responses": ["API_call_2", "API_call_3"]}]
````

### 评测集：sealtool（test，1354 条）

**sealtool/test_out_domain-difficult-115**

题目：
````
请根据 task_instruction 选择所需的 API，并返回 api_calling 列表。
输出格式：[{"api": "", "parameters": {"": ""}, "responses": ["API_call_0","API_call_1"]},{"api": "", "parameters": {"": ""}, "responses": ["API_call_2"]}]
responses 可以用作参数值。responses 的数量取决于 api_list 中的信息。

Input:
api_list = [{'api_name': 'getFilmInfo', 'api_description': '获取电影的信息', 'field': 'Entertainment/Film', 'parameters': {'film_name': {'type': 'str', 'description': '要查询信息的电影名称'}}, 'required': ['film_name'], 'responses': {'title': {'type': 'str', 'description': '电影的标题'}, 'release_year': {'type': 'int', 'description': '电影的上映年份'}, 'director': {'type': 'str', 'description': "The name of the film's director"}, 'actors': {'type': 'str', 'de …（共 3902 字符，已截断）
````
答案 / 判分依据：
````
[{"api": "getBrandInfo", "parameters": {"brand_name": "Gucci"}, "responses": ["API_call_0", "API_call_1", "API_call_2", "API_call_3"]}, {"api": "getFilmInfo", "parameters": {"film_name": "Pulp Fiction"}, "responses": ["API_call_4", "API_call_5", "API_call_6", "API_call_7", "API_call_8"]}, {"api": "getVisualCultureInfo", "parameters": {"topic": "modern art"}, "responses": ["API_call_9"]}]
````

**sealtool/test_out_domain-difficult-260**

题目：
````
请根据 task_instruction 选择所需的 API，并返回 api_calling 列表。
输出格式：[{"api": "", "parameters": {"": ""}, "responses": ["API_call_0","API_call_1"]},{"api": "", "parameters": {"": ""}, "responses": ["API_call_2"]}]
responses 可以用作参数值。responses 的数量取决于 api_list 中的信息。

Input:
api_list = [{'api_name': 'getRobotPosition', 'api_description': '获取机器人的当前位置', 'field': 'Science/Robotics', 'parameters': {'robot_id': {'type': 'int', 'description': '要获取位置的机器人 ID'}}, 'required': ['robot_id'], 'responses': {'position': {'type': 'str', 'description': '指定机器人的当前位置'}}}, {'api_name': 'createEdgeDevice', 'api_description': '在网络中创建新的边缘设备', 'field': 'Internet of Things/Edge Computing', 'paramet …（共 3476 字符，已截断）
````
答案 / 判分依据：
````
[{"api": "getRobotPosition", "parameters": {"robot_id": 26}, "responses": ["API_call_0"]}, {"api": "createEdgeDevice", "parameters": {"network_id": "Xevb1GzHAP", "device_name": "iEdge"}, "responses": ["API_call_1"]}, {"api": "validateInput", "parameters": {"input": "1Tbgt6LYv"}, "responses": ["API_call_2", "API_call_3"]}]
````

**sealtool/test_out_domain-difficult-584**

题目：
````
请根据 task_instruction 选择所需的 API，并返回 api_calling 列表。
输出格式：[{"api": "", "parameters": {"": ""}, "responses": ["API_call_0","API_call_1"]},{"api": "", "parameters": {"": ""}, "responses": ["API_call_2"]}]
responses 可以用作参数值。responses 的数量取决于 api_list 中的信息。

Input:
api_list = [{'api_name': 'restoreArtwork', 'api_description': '修复艺术品的受损部分', 'field': 'Art and Design/Art Conservation', 'parameters': {'artwork_id': {'type': 'str', 'description': '待修复艺术品的 ID'}, 'damaged_parts': {'type': 'str', 'description': '艺术品受损部分的描述'}, 'restoration_method': {'type': 'str', 'description': '修复所用的方法（例如补色、补绘）'}, 'time_required': {'type': 'str', 'description': 'The approximate time …（共 4540 字符，已截断）
````
答案 / 判分依据：
````
[{"api": "getArtwork", "parameters": {"artwork_id": "ksj29d8qhW"}, "responses": ["API_call_0", "API_call_1", "API_call_2", "API_call_3"]}, {"api": "restoreArtwork", "parameters": {"artwork_id": "API_call_0", "damaged_parts": "cracked paint", "restoration_method": "retouching"}, "responses": ["API_call_4"]}]
````

### 评测集：apibank（test，534 条）

**apibank/level-1-api/121**

题目：
````
请根据之前的对话上下文，按 [ApiName(key1='value1', key2='value2', ...)] 的格式生成一个 API 请求。
当前年份是 2023 年。
Input: 
User: User's utterence
AI: AI's response

期望输出：
API-Request: [ApiName(key1='value1', key2='value2', ...)]

API descriptions:
{"name": "QueryStock", "description": "该 API 查询给定股票代码和日期的股票价格。", "input_parameters": {"stock_code": {"type": "str", "description": "给定股票的股票代码。"}, "date": {"type": "str", "description": "股票价格的日期。格式：%Y-%m-%d"}}, "output_parameters": {"stock_price": {"type": "float", "description": "给定股票的股票价格。"}}}
{"name": "GetUserToken", "description": "通过用户名和密码获取用户令牌。", "input_parameters": {"username": {"type": "str", "description": "用户的用户名。"}, "password" …（共 1172 字符，已截断）
````
答案 / 判分依据：
````
API-Request: [QueryStock(stock_code='NFLX', date='2022-02-21')]
````

**apibank/level-1-api/50**

题目：
````
请根据之前的对话上下文，按 [ApiName(key1='value1', key2='value2', ...)] 的格式生成一个 API 请求。
当前年份是 2023 年。
Input: 
User: User's utterence
AI: AI's response

期望输出：
API-Request: [ApiName(key1='value1', key2='value2', ...)]

API descriptions:
{"name": "GetUserToken", "description": "通过用户名和密码获取用户令牌。", "input_parameters": {"username": {"type": "str", "description": "用户的用户名。"}, "password": {"type": "str", "description": "用户的密码。"}}, "output_parameters": {"token": {"type": "str", "description": "用户的令牌。"}}}
{"name": "AddReminder", "description": "添加提醒事项的 API，包含内容和时间。", "input_parameters": {"token": {"type": "str", "description": "用户的令牌。"}, "content": {"type": "str", "description": "会议的内容。"} …（共 1586 字符，已截断）
````
答案 / 判分依据：
````
API-Request: [AddReminder(token='n9m8k7j6h5g4f3d2s1a0', content='Call mom', time='2022-05-10 10:00:00')]
````

**apibank/level-2-api/47**

题目：
````
请根据之前的对话上下文，按 [ApiName(key1='value1', key2='value2', ...)] 的格式生成一个 API 请求。
当前年份是 2023 年。
Input: 
User: User's utterence
AI: AI's response

期望输出：
API-Request: [ApiName(key1='value1', key2='value2', ...)]

API descriptions:
{"name": "ToolSearcher", "description": "根据关键词在工具库中搜索相关工具。", "input_parameters": {"keywords": {"type": "str", "description": "要搜索的关键词。"}}, "output_parameters": {"best_matchs": {"type": "Union[List[dict], dict]", "description": "最匹配的工具。"}}}
User: 你能帮我为 2023 年 1 月 5 日下午 3 点主题为“销售报告”的会议添加一个提醒吗？
AI: 好的，我可以帮您。首先，让我搜索相关工具。
API-Request: [ToolSearcher(keywords='add reminder')]->[{'name': 'GetUserToken', 'description': 'G …（共 2147 字符，已截断）
````
答案 / 判分依据：
````
API-Request: [GetUserToken(username='newuser', password='newpass')]
````

### 评测集：bfcl（test，3641 条）

**live_irrelevance_575-179-4**

题目：
````
[user] 行程
````
答案 / 判分依据：
````
类别 live_irrelevance；可用函数 ['EventQuery', 'reschedule_event']；标准答案 null
````

**parallel_multiple_100**

题目：
````
[user] “你能告诉我使用 ARIMA 模型预测的 Apple Inc. 未来 30 天的股价吗？然后再用 LSTM 模型给出 Microsoft Corporation 未来 45 天的股票预测。之后，请提供纽约市未来 7 天的天气预报，再给出洛杉矶未来 14 天的天气预报。”
````
答案 / 判分依据：
````
类别 parallel_multiple；可用函数 ['weather_forecast', 'stock_forecast']；标准答案 [{"stock_forecast": {"company": ["Apple Inc.", "AAPL"], "days": [30], "model": ["ARIMA", ""]}}, {"stock_forecast": {"company": ["Microsoft Corporation", "MSFT"], "days": [45], "model": ["LSTM"]}}, {"weather_forecast": {"location": ["New York City", "NYC", "New York", "NY"], "days": [7]}}, {"weather_forecast": {"location": ["Los Angeles", "LA", "Los Angeles, California", "CA"], "days": [14]}}]
````

**live_multiple_326-132-16**

题目：
````
[user] 我想找一部电影来看。有没有由 Vincent Andriano 主演、Gurinder Chadha 执导的电影？
````
答案 / 判分依据：
````
类别 live_multiple；可用函数 ['Movies_3_FindMovies', 'Restaurants_2_ReserveRestaurant', 'Restaurants_2_FindRestaurants']；标准答案 [{"Movies_3_FindMovies": {"directed_by": ["Gurinder Chadha"], "genre": ["", "dontcare"], "cast": ["Vincent Andriano"]}}]
````
