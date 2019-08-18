# <center>SVM 支持向量机（一）</center>

## 1、感知机

### 1.1、感知机模型

&#8195;&#8195;假设现在要判断是否给某个客户办理信用卡，已有的是用户的性别、年龄、学历、工作年限、负债情况等信息，用户个人金融信息统计如下表所示：

| 用户 \  特征 | 性别 | 年龄 | 学历 | 工作年限 | 负债情况（元） |
| :----------: | :--: | :--: | :--: | :------: | :------------: |
|    用户1     |  男  |  23  | 本科 |    1     |      5000      |
|    用户2     |  女  |  25  | 高中 |    6     |      6000      |
|    用户3     |  女  |  27  | 硕士 |    1     |      1200      |
|    用户4     |  男  |  26  | 硕士 |    1     |      1000      |
|      …       |  …   |  …   |  …   |    …     |       …        |

&#8195;&#8195;可以将每个用户看成一个向量 $x_i \ , i=12,...$，向量的维度由用户的性别、年龄、学历、工作年限、负债情况等信息组成，即 $x_i=( x^{(1)}_i\,\, x^{(2)}_i \ ,...\ , x^{(n}_i )$，那么一种简单的判别方法就是对用户各个维度求一个加权和，并且为每一个维度赋予一个权重$w_j \ , j=1,2,...,n$，当这个加权和超过某一个门限值，就判定可以给该用户办理信用卡，低于门限值则拒绝，如：

- 如果$\sum\limits^n_{j=1}w_jx_i^{(j)} \ ≥ threshold$，则可以给用$x_i$办理信用卡
- 如果$\sum\limits^n_{j=1}w_jx_i^{(j)} \ \le threshold$，则拒绝给用$x_i$办理信用卡

&#8195;&#8195;可以将是否给办理的结果用 “+1” 和 “-1” 来表示，这样，上面的判决式可以进行一定变形，即不等式左右分别减去阈值“$threshold$”，从而可以得到一个符号函数，即：
$$
h(x_i)=sign[(\sum\limits^n_{j=1}w_jx_i^{(j)}) -\ threshold \ ]
$$
&#8195;&#8195;这样，公式(1)中的 $h(x)$ 就被称为<font color=red>**感知机函数**</font>，可以再进一步变形得到两个向量内积的形式，即：
$$
\begin{aligned}
h(x_i) &= sign[(\sum\limits^n_{j=1}w_jx_i^{(j)}) -\ threshold \ ] \\
&= sign[(\sum\limits^n_{j=1}w_jx_i^{(j)}) +\ \mathbf{b} \ ] \\
&= sign(\mathbf{w\cdot x + b})
\end{aligned}
$$
&#8195;&#8195;其中，$\mathbf{w}=(w_1, w_2 ,..., w_n)$是各个特征权重组成的向量；$\mathbf {x = (x_1, x_2 ,..., x_n)}$是数据的特征向量；$\mathbf b = - threshold$s 是阈值（取负，也叫偏置常数）。

&#8195;&#8195;所以，由以上可得感知机模型的函数如下：
$$
\begin{aligned}
h(x) &= \ sign(\mathbf{w\cdot x + b})  \\
\\
&=
\begin{cases}
+1  ,\quad \mathbf{w\cdot x + b}>0  \\
-1 , \quad \mathbf{w\cdot x + b}<0  \\
\end{cases}
\end{aligned}
$$
&#8195;&#8195;当 $\mathbf x$ 的维度为2时，可以得到二维平面的一些点，而感知机的分界线对应的是一条直线，即感知机的分界函数为：$\mathbf b +w_1x^{1}+w_2x^{2}=0$此时 $x^{(1)} \ 和\ x^{(2)}$分别对应 x 轴和 y 轴，常数项 b 对应的是截距，如下图：

<center><img src="https://img-blog.csdnimg.cn/2019071311243895.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" width=50%>



&#8195;&#8195;感知机对应于输入空间将实例划分为正负两类的分割超平面$\mathbf{w\cdot x + b} = 0$，属于判别模型。

</br>

### 1.2、感知机学习方法

&#8195;&#8195;**感知机模型学习的目的是为了确定参数** $\mathbf {w,b}$，所以假设训练集为$T=\{ (\mathbf x_1,y_1) ,(\mathbf x_2,y_2) ,..., (\mathbf x_2,y_2) \}$。

&#8195;&#8195;感知机和支持向量机的重要前提均是 <font color=#FF0000>训练数据集**线性可分**</font>，则感知机学习的目标就是求得一个能够将训练数据集中正负样本完全区分的超平面。

<center><img src="https://img-blog.csdnimg.cn/20190713112552326.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" width=50%>

&#8195;&#8195;对二维来说，即希望能到的超平面（分割直线）离红色最近的点与蓝色最近的点的距离尽可能的远，这样能够很明显的将两类进行区分。感知器求解采用的是损失函数最小的方法，即定义一个损失函数，通过将损失函数最小化来求解 $\mathbf {w,b}$ 。

​	感知机模型选择的损失函数是 <font color=dgreen>**误分类点**</font> 到分类超平面 $S$ 的总距离。样本点中任意一点 $x_i$ 到分类超平面 $S$ 的距离为：
$$
d = \frac{| \mathbf {w\cdot x_i + b} |}{ ||\mathbf w||_2}
$$
​	其中，$||\mathbf w||_2$ 为向量 $\mathbf w$ 的模，也叫 L2 范数。

​	应用感知机模型函数，对于样本点 $x_i$ 有：
$$
\begin{aligned}
h(x_i) &= \ sign(\mathbf{w\cdot x_i + b})  \\
\\
&=
\begin{cases}
+1  ,\quad \mathbf{w\cdot x_i + b}>0  \\
-1 , \quad \mathbf{w\cdot x_i + b}<0  \\
\end{cases}
\end{aligned}
$$
​	对于上面公式，可得：

- 对于分类**正确**的样本点，有 $y_i ( \mathbf{w\cdot x + b} ) ≥1 > 0$ 恒成立，即：

  - $y_i = +1$时，有$\mathbf{w\cdot x + b} >0$ ;
  - $y_i = -1$时，有$\mathbf{w\cdot x + b}<0$ 。

- 对于分类**错误**的样本点，有 $y_i ( \mathbf{w\cdot x + b} ) ≤ -1 < 0$ 恒成立，即:

  - $y_i = +1$时，有$\mathbf{w\cdot x + b}<0$  ;
  - $y_i = -1$时，有$\mathbf{w\cdot x + b}>0$ 。

  所以，对于误分类点 $(x_i, y_i)$ 到超平面 $S$ 的距离可等价为如下表达式：
  $$
  \begin{aligned}
  d^{\ '} &= \frac{| \mathbf {w\cdot x_i + b} |}{ ||\mathbf w||_2}  \\
  \\
  &= \frac{|y_i| \cdot| \mathbf {w\cdot x_i + b} |}{ ||\mathbf w||_2}  \\
  \\
  &= \frac{- y_i \cdot( \mathbf {w\cdot x_i + b} )}{ ||\mathbf w||_2}
  \end{aligned}
  $$
  

  &#8195;最终，将所有错误分类点归为一个集合M，则它们到超平面的距离的总和定义为<font color=#ff000>**损失函数**</font>：
  $$
  L({\mathbf {w,b}}) = -\frac{1}{||\mathbf w||_2 \cdot } \cdot \sum\limits_{x_i \in M} y_i\cdot ( \mathbf {w\cdot x + b} )
  $$
  

  &#8195;所以，当所有错误分类点到超平面的距离最小的时候，损失函数是最小的，这是得到的模型是最优的，而又因为所有样本点到超平面的距离公式中分母$||\mathbf w||_2$是相同的，所以公式（7）的**损失函数**最终可变形为：
  $$
  L({\mathbf {w,b}}) = - \sum\limits_{\mathbf x_i \in M} y_i\cdot ( \mathbf {w\cdot x + b} )
  $$



## 2、感知机模型实现

### 2.1、感知机优化算法



&#8195;&#8195;对于求解感知机最优模型的问题转换为最优解，即求损失函数的最小值时对应的参数即为模型参数，采用梯度下降法进行求解，分别对参数求偏导：
$$
\begin{cases}
\frac{\partial}{\partial \mathbf w} L(\mathbf {w, b}) = \frac{\partial}{\partial \mathbf w} [-\sum\limits_{x_i \in M}y_i\cdot(w \cdot \mathbf x_i + \mathbf b) ] = -\sum\limits_{x_i \in M}y_i\mathbf x_i  \\

\\
\frac{\partial}{\partial \mathbf w} L(\mathbf {w, b}) = \frac{\partial}{\partial \mathbf b} [-\sum\limits_{x_i \in M}y_i\cdot(w \cdot \mathbf x_i + \mathbf b) ] = -\sum\limits_{x_i \in M}y_i  \\
\end{cases}
\\
$$
&#8195;其中，$\mathbf w = (w_1, w_2 ,..., w_n)$  是各个特征对应的权重向量 ；$\mathbf x_i = ( x^{(1)}_i\,\, x^{(2)}_i \ ,...\ , x^{(m}_i )$ 。 

&#8195;求解过程即每次随机选取一个误分类样本点$（x_i , y_i）$，对$\mathbf {w, b}$ 进行一次更新，迭代计算公式如下：
$$
\begin{aligned}
\mathbf w^{‘} &\leftarrow  \mathbf w + \eta y_i\mathbf x_i   \\
\mathbf b^{‘} &\leftarrow \mathbf b + \eta y_i
\end{aligned}
$$
&#8195;&#8195;其中，$0 ≤ \eta ≤ 1$是算法的学习率，即步长。通过梯度下降法不停迭代使得损失函数$L({\mathbf {w,b}})$快速地不断减小，直到满足要求。



### 2.2、感知机模型求解

输入：线性可分训练集 $T=\{ (\mathbf x_1 ,y_1) ,(\mathbf x_2 ,y_2), ..., (\mathbf x_n ,y_n) \}$，且 $y_i \in \{-1,1\} $，学习率 $\eta$。

输出：感知机模型函数 $h(\mathbf x) =sign(\mathbf {w\cdot x + b})$。

求解步骤：

1. 选取初始值向量$\mathbf w$ 和偏置常数$\mathbf b$ .

2. 在训练集中选取数据$(\mathbf x_i , y_i)$ .

3. 求解计算$y_i ( \mathbf{w\cdot x + b} ) <0$ （判定为误分类点），则进行更新.
   $$
   \begin{aligned}
   \mathbf w^{‘} &\leftarrow  \mathbf w + \eta y_i\mathbf x_i   \\
   \mathbf b^{‘} &\leftarrow \mathbf b + \eta y_i
   \end{aligned}
   $$

4. 重复第2步和第3步，直至训练集中没有误分类点或满足迭代终止条件，得到感知机模型  $h(\mathbf x) =sign(\mathbf {w \cdot x  + b})$.

</br>



## 3、感知机模型总结

​		感知机模型的基本思想是：西安随机选择一个超平面，对样本点进行划分，然后当一个实例点被误分类，即位于分类超平面错误一侧时，调整初始值向量$\mathbf w$ 和偏置常数$\mathbf b$ ，使分类超平面向该误分类点的一侧移动，直至超平面越过该误分类点为止。所以，如果给的初始值不同，则最后得到的分割超平面 $\mathbf{w\cdot x + b} = 0$ 也可能不同，即感知机模型存在多个分割超平面。

> **注意**：感知机模型有两个问题：
>
> 1. 使用前提是数据集必须为线性可分，当数据集线性不可分的时候，感知记得学习算法不收敛，迭代过程会发生震荡；
> 2. 感知机模型仅适用于二分类问题，在实际应用中存在一定限制。

</br>





---
> 文中实例及参考：
> + 刘宇波老师《Python入门机器学习》
> + 《机器学习基础》
> + 贪心学院，[https://www.greedyai.com](https://www.greedyai.com)

---