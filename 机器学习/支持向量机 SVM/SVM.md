

# 支持向量机 SVM

## 1、感知机

### 1.1、感知机模型

假设现在要判断是否给某个客户办理信用卡，已有的是用户的性别、年龄、学历、工作年限、负债情况等信息，用户个人金融信息统计如下表所示：

| 用户 \  特征 | 性别 | 年龄 | 学历 | 工作年限 | 负债情况（元） |
| :----------: | :--: | :--: | :--: | :------: | :------------: |
|    用户1     |  男  |  23  | 本科 |    1     |      5000      |
|    用户2     |  女  |  25  | 高中 |    6     |      6000      |
|    用户3     |  女  |  27  | 硕士 |    1     |      1200      |
|    用户4     |  男  |  26  | 硕士 |    1     |      1000      |
|      …       |  …   |  …   |  …   |    …     |       …        |

可以将每个用户看成一个向量 $x_i \ , i=12,...$，向量的维度由用户的性别、年龄、学历、工作年限、负债情况等信息组成，即$x_i=( x^{(1)}_i\,\, x^{(2)}_i \ ,...\ , x^{(n}_i )$，那么一种简单的判别方法就是对用户各个维度求一个加权和，并且为每一个维度赋予一个权重$w_j \ , j=1,2,...,n$，当这个加权和超过某一个门限值，就判定可以给该用户办理信用卡，低于门限值则拒绝，如：

- 如果$\sum\limits^n_{j=1}w_jx_i^{(j)} \ ≥ threshold \ $，则可以给用$x_i$办理信用卡
- 如果$\sum\limits^n_{j=1}w_jx_i^{(j)} \ \le threshold \ $，则拒绝给用$x_i$办理信用卡

可以将是否给办理的结果用“+1”和“-1”来表示，这样，上面的判决式可以进行一定变形，即不等式左右分别减去阈值“threshold”，从而可以得到一个符号函数，即：
$$
h(x_i)=sign[(\sum\limits^n_{j=1}w_jx_i^{(j)}) -\ threshold \ ]
$$
这样，公式(1)中的$h(x)$就被称为感知机函数，可以再进一步变形得到两个向量内积的形式，即：
$$
h(x_i)=sign[(\sum\limits^n_{j=1}w_jx_i^{(j)}) -\ threshold \ ] \\
=sign[(\sum\limits^n_{j=1}w_jx_i^{(j)}) +\ \mathbf{b} \ ] \\
=sign(\mathbf{w^Tx + b})
$$
其中，$\mathbf{w}=(w_1, w_2 ,..., w_n)$是各个特征权重组成的向量；$\mathbf {x = (x_1, x_2 ,..., x_n)}$是数据的特征向量；$\mathbf b = - threshold$s 是阈值（取负，也叫偏置常数）。

所以，由以上可得感知机模型的函数如下：
$$
h(x) = sign(\mathbf{w^Tx + b}) =

\begin{cases}
\ +1  ,\ \mathbf{w^Tx + b}>0  \\
\ -1 , \ \mathbf{w^Tx + b}<0  \\
\end{cases}
$$
当 $\mathbf x\ $的维度为2时，可以得到二维平面的一些点，而感知机的分界线对应的是一条直线，此时$x^{(1)} \ 和\ x^{(2)}  $分别对应 x 轴和 y 轴，常数项 b 对应的是截距，如下图：

![在这里插入图片描述](E:\MardkDown-Books\机器学习\支持向量机 SVM\SVM.assets\2019070922003044.png)

感知机对应于输入空间将实例划分为正负两类的分割超平面$\mathbf{w^Tx + b} = 0$，属于判别模型。

</br>

### 1.2、感知机学习方法

感知机模型学习的目的是为了确定参数$\mathbf {w,b}$，所以假设训练集为$T={ (\mathbf x_1,y_1) ,(\mathbf x_2,y_2) ,..., (\mathbf x_2,y_2) }$。

感知机和支持向量机的重要前提均是 <font color=#FF0000>训练数据集**线性可分**</font>，则感知机学习的目标就是求得一个能够将训练数据集中正负样本完全区分的超平面。

![在这里插入图片描述](E:\MardkDown-Books\机器学习\支持向量机 SVM\SVM.assets\20190709212518891.png)

对二维来说，即希望能到的超平面（分割直线）离红色最近的点与蓝色最近的点的距离尽可能的远，这样能够很明显的将两类进行区分。

由解析几何的知识，可知，二维平面中，点$(x, y)$到直线$Ax + By +C = 0$的距离为：
$$
\frac{|Ax + By +C |}{\sqrt{A^2 + B^2}}
$$
则，拓展到 n 维空间中，$\mathbf {w^Tx + b} = 0$，点到直线的距离为：
$$
d^{\ '} = \frac{| \mathbf {w^Tx + b} |}{ ||\mathbf w||_2} \quad , \quad ||\mathbf w|| = \sqrt{w_1^2 + w_2^2 + ... +w_n^2 }
$$
其中，$||\mathbf w||_2$也叫“L2范数”，也就是模。

对于公式（3）可得；

- 对于分类正确的样本点，有 $y_i ( \mathbf{w^Tx + b} ) > 0$ 恒成立，即：
  - $ y_i = +1$时，有$\mathbf{w^Tx + b}>0$ ;
  - $ y_i = -1$时，有$\mathbf{w^Tx + b}<0$ 。
- 对于分类错误的样本点，有 $y_i ( \mathbf{w^Tx + b} ) < 0$ 恒成立，即:
  - $ y_i = +1$时，有$\mathbf{w^Tx + b}<0$  ;
  - $ y_i = -1$时，有$\mathbf{w^Tx + b}>0$ 。

所以，分类错误的样本点到分类超平面 $\mathbf S$ 的距离等价于：
$$
d^{\ '} = \frac{| \mathbf {w^Tx + b} |}{ ||\mathbf w||_2}  \\
= \frac{|y_i|\cdot| \mathbf {w^Tx + b} |}{ ||\mathbf w||_2}  \\
= - \frac{y_i\cdot ( \mathbf {w^Tx + b} ) }{ ||\mathbf w||_2}
$$
最终，将所有错误分类点归为一个集合M，则它们到超平面的距离的总和定义为<font color=#ff000>**损失函数**</font>：
$$
L({\mathbf {w,b}}) = -\frac{1}{||\mathbf w||_2 \cdot } \cdot \sum\limits_{x_i \in M} y_i\cdot ( \mathbf {w^Tx + b} )
$$
所以，当所有错误分类点到超平面的距离最小的时候，损失函数是最小的，这是得到的模型是最优的，而又因为所有样本点到超平面的距离公式中分母$||\mathbf w||_2$是相同的，所以公式（7）的**损失函数**最终可变形为：
$$
L({\mathbf {w,b}}) = - \sum\limits_{\mathbf x_i \in M} y_i\cdot ( \mathbf {w^Tx + b} )
$$

### 1.3、感知机模型求解

对于求解感知机最优模型的问题转换为最优解，即求损失函数的最小值时对应的参数即为模型参数，采用梯度下降法进行求解，分别对参数求偏导：
$$
\begin{cases}
\frac{\partial L(\mathbf {w, b})}{\partial \mathbf w}= -\sum\limits_{x_i \in M}y_i\mathbf x_i  \\
\\
\frac{\partial L(\mathbf {w, b})}{\partial \mathbf b} = -\sum\limits_{x_i \in M}y_i  \\
\end{cases}
\\
其中， \mathbf w = (w_1, w_2 ,..., w_n) 是各个特征对应的权重向量 ；\quad
\mathbf x_i = (x^{(i)}_1, x^{(i)}_2 ,..., x^{(i)}_n)
$$

求解过程即每次随机选取一个误分类样本点$（x_i , y_i）$，对$\mathbf {w, b}$ 进行一次更新，迭代计算公式如下：
$$
\mathbf w^{‘} \leftarrow  \mathbf w + \eta y_i\mathbf x_i   \\
\mathbf b^{‘} \leftarrow \mathbf b + \eta y_i
$$
其中，$0 ≤ \eta ≤ 1$是算法的学习率，即步长。通过梯度下降法不停迭代使得损失函数$L({\mathbf {w,b}})$快速地不断减小，直到满足要求。

> **注意：**感知机模型有两个问题：
>
> 1. 使用前提是数据集必须为线性可分，当数据集线性不可分的时候，感知记得学习算法不收敛，迭代过程会发生震荡；
> 2. 感知机模型仅适用于二分类问题，在实际应用中存在一定限制。









---


$$
\begin{cases}
\ \frac{| \mathbf {w^Tx + b} |}{ ||\mathbf w||_2} ≥ d  ,\ y^{(i)} \forall +1  \\
\\
\ \frac{| \mathbf {w^Tx + b} |}{ ||\mathbf w||_2} ≥ d  ,\ y^{(i)} \forall -1  \\
\end{cases}
$$
