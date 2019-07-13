

# <center>SVM 支持向量机</center>

&#8195;&#8195;支持向量机（SVM）是一个功能强大并且全面的机器学习模型，它能够执行线性或非线性分类问题、回归问题，甚至是异常值检测任务。

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

&#8195;&#8195;可以将每个用户看成一个向量 $x_i \ , i=12,...$，向量的维度由用户的性别、年龄、学历、工作年限、负债情况等信息组成，即$x_i=( x^{(1)}_i\,\, x^{(2)}_i \ ,...\ , x^{(n}_i )$，那么一种简单的判别方法就是对用户各个维度求一个加权和，并且为每一个维度赋予一个权重$w_j \ , j=1,2,...,n$，当这个加权和超过某一个门限值，就判定可以给该用户办理信用卡，低于门限值则拒绝，如：

- 如果$\sum\limits^n_{j=1}w_jx_i^{(j)} \ ≥ threshold$，则可以给用$x_i$办理信用卡
- 如果$\sum\limits^n_{j=1}w_jx_i^{(j)} \ \le threshold$，则拒绝给用$x_i$办理信用卡

&#8195;&#8195;可以将是否给办理的结果用“+1”和“-1”来表示，这样，上面的判决式可以进行一定变形，即不等式左右分别减去阈值“$threshold$”，从而可以得到一个符号函数，即：
$$
h(x_i)=sign[(\sum\limits^n_{j=1}w_jx_i^{(j)}) -\ threshold \ ]
$$
&#8195;&#8195;这样，公式(1)中的$h(x)$就被称为感知机函数，可以再进一步变形得到两个向量内积的形式，即：
$$
h(x_i)=sign[(\sum\limits^n_{j=1}w_jx_i^{(j)}) -\ threshold \ ] \\
=sign[(\sum\limits^n_{j=1}w_jx_i^{(j)}) +\ \mathbf{b} \ ] \\
=sign(\mathbf{w^Tx + b})
$$

&#8195;&#8195;其中，$\mathbf{w}=(w_1, w_2 ,..., w_n)$是各个特征权重组成的向量；$\mathbf {x = (x_1, x_2 ,..., x_n)}$是数据的特征向量；$\mathbf b = - threshold$s 是阈值（取负，也叫偏置常数）。

&#8195;&#8195;所以，由以上可得感知机模型的函数如下：
$$
h(x) = sign(\mathbf{w^Tx + b}) =
\begin{cases}
\ +1  ,\ \mathbf{w^Tx + b}>0  \\
\ -1 , \ \mathbf{w^Tx + b}<0  \\
\end{cases}
$$

&#8195;&#8195;当 $\mathbf x$的维度为2时，可以得到二维平面的一些点，而感知机的分界线对应的是一条直线，此时$x^{(1)} \ 和\ x^{(2)}$分别对应 x 轴和 y 轴，常数项 b 对应的是截距，如下图：

<center><img src="https://img-blog.csdnimg.cn/2019071311243895.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" width=50%>


&#8195;&#8195;感知机对应于输入空间将实例划分为正负两类的分割超平面$\mathbf{w^Tx + b} = 0$，属于判别模型。

</br>

### 1.2、感知机学习方法

&#8195;&#8195;感知机模型学习的目的是为了确定参数$\mathbf {w,b}$，所以假设训练集为$T=\{ (\mathbf x_1,y_1) ,(\mathbf x_2,y_2) ,..., (\mathbf x_2,y_2) \}$。

&#8195;&#8195;感知机和支持向量机的重要前提均是 <font color=#FF0000>训练数据集**线性可分**</font>，则感知机学习的目标就是求得一个能够将训练数据集中正负样本完全区分的超平面。

<center><img src="https://img-blog.csdnimg.cn/20190713112552326.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" width=50%>

&#8195;&#8195;对二维来说，即希望能到的超平面（分割直线）离红色最近的点与蓝色最近的点的距离尽可能的远，这样能够很明显的将两类进行区分。

&#8195;&#8195;由解析几何的知识，可知，二维平面中，点$(x, y)$到直线$Ax + By +C = 0$的距离为：
$$
\frac{|Ax + By +C |}{\sqrt{A^2 + B^2}}
$$
&#8195;&#8195;则，拓展到 n 维空间中，$\mathbf {w^Tx + b} = 0$，点到直线的距离为：
$$
d = d^{\ '} = \frac{| \mathbf {w^Tx + b} |}{ ||\mathbf w||_2} \quad , \quad ||\mathbf w|| = \sqrt{w_1^2 + w_2^2 + ... +w_n^2 }
$$
&#8195;&#8195;其中，$||\mathbf w||_2$也叫 “ L2范数 ”，也就是模。

<center><img src="https://img-blog.csdnimg.cn/20190713112643903.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" width=60%>

&#8195;&#8195;所以，对于数据集的样本点，有：
$$
\begin{cases}
\ \frac{| \mathbf {w^Tx + b} |}{ ||\mathbf w||_2} ≥ d  ,\  \forall y^{(i)} = +1  \\
\\
\ \frac{| \mathbf {w^Tx + b} |}{ ||\mathbf w||_2} ≥ d  ,\ \forall y^{(i)} = -1  \\
\end{cases}
$$
&#8195;&#8195;将上式分子分母同时除以$d$，可得：
$$
\begin{cases}
\ \frac{| \mathbf {w^Tx + b} |}{ ||\mathbf w||_2d} ≥ 1  ,\  \forall y^{(i)} = +1  \\
\\
\ \frac{| \mathbf {w^Tx + b} |}{ ||\mathbf w||_2d} ≤ -1  ,\ \forall y^{(i)} = -1  \\
\end{cases}
$$
&#8195;&#8195;令$\ \mathbf {w^T_d } = \frac{| \mathbf {w^T} |}{ ||\mathbf w||_2d} , \  \mathbf {b_d} = \frac{| \mathbf {b} |}{ ||\mathbf w||_2d}$ ，所以，可得下式：
$$
\begin{cases}
\ \mathbf {w^T_dx + b_d } ≥ 1  ,\  \forall y^{(i)} = +1  \\
\\
\ \mathbf {w^T_dx + b_d } ≤ -1  ,\ \forall y^{(i)} = -1  \\
\end{cases}
$$
&#8195;&#8195;不妨令$\mathbf {w = w^T_d } \, \ \mathbf {b = b_d }$ ，所以：
$$
\begin{cases}
\ \mathbf {w^Tx + b } ≥ 1  ,\  \forall y^{(i)} = +1  \\
\\
\ \mathbf {w^Tx + b } ≤ -1  ,\ \forall y^{(i)} = -1  \\
\end{cases}
$$

&#8195;&#8195;对于公式（6）可得；

- 对于分类正确的样本点，有 $y_i ( \mathbf{w^Tx + b} ) ≥1 > 0$ 恒成立，即：
  - $y_i = +1$时，有$\mathbf{w^Tx + b} >0$ ;
  - $y_i = -1$时，有$\mathbf{w^Tx + b}<0$ 。
- 对于分类错误的样本点，有 $y_i ( \mathbf{w^Tx + b} ) ≤ -1 < 0$ 恒成立，即:
  - $y_i = +1$时，有$\mathbf{w^Tx + b}<0$  ;
  - $y_i = -1$时，有$\mathbf{w^Tx + b}>0$ 。

&#8195;&#8195;所以，分类错误的样本点到分类超平面 $\mathbf S$ 的距离等价于：
$$
d^{\ '} = \frac{| \mathbf {w^Tx + b} |}{ ||\mathbf w||_2}  \\
= \frac{|y_i|\cdot| \mathbf {w^Tx + b} |}{ ||\mathbf w||_2}  \\
= - \frac{y_i\cdot ( \mathbf {w^Tx + b} ) }{ ||\mathbf w||_2}
$$
&#8195;&#8195;最终，将所有错误分类点归为一个集合M，则它们到超平面的距离的总和定义为<font color=#ff000>**损失函数**</font>：
$$
L({\mathbf {w,b}}) = -\frac{1}{||\mathbf w||_2 \cdot } \cdot \sum\limits_{x_i \in M} y_i\cdot ( \mathbf {w^Tx + b} )
$$
&#8195;&#8195;所以，当所有错误分类点到超平面的距离最小的时候，损失函数是最小的，这是得到的模型是最优的，而又因为所有样本点到超平面的距离公式中分母$||\mathbf w||_2$是相同的，所以公式（7）的**损失函数**最终可变形为：
$$
L({\mathbf {w,b}}) = - \sum\limits_{\mathbf x_i \in M} y_i\cdot ( \mathbf {w^Tx + b} )
$$

### 1.3、感知机模型求解
&#8195;&#8195;对于求解感知机最优模型的问题转换为最优解，即求损失函数的最小值时对应的参数即为模型参数，采用梯度下降法进行求解，分别对参数求偏导：
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

&#8195;&#8195;求解过程即每次随机选取一个误分类样本点$（x_i , y_i）$，对$\mathbf {w, b}$ 进行一次更新，迭代计算公式如下：
$$
\mathbf w^{‘} \leftarrow  \mathbf w + \eta y_i\mathbf x_i   \\
\mathbf b^{‘} \leftarrow \mathbf b + \eta y_i
$$
&#8195;&#8195;其中，$0 ≤ \eta ≤ 1$是算法的学习率，即步长。通过梯度下降法不停迭代使得损失函数$L({\mathbf {w,b}})$快速地不断减小，直到满足要求。

>  **注意**：感知机模型有两个问题：
>
> 1. 使用前提是数据集必须为线性可分，当数据集线性不可分的时候，感知记得学习算法不收敛，迭代过程会发生震荡；
> 2. 感知机模型仅适用于二分类问题，在实际应用中存在一定限制。

</br>

## 2、支持向量机

&#8195;&#8195;由上述可知感知机模型，即**在数据集线性可分的条件下**，利用分割超平面$\mathbf {w^T \cdot x} + \mathbf b = 0$ 把样本点划分为两类，通过计算误分类点距离超平面距离总和作为损失函数，使其最小化从而调整超平面，直至所有误分类点被纠正正确后迭代结束。

&#8195;&#8195;因为 $\mathbf {w^T ，b}$ 的取值不同，所以得到的分割超平面也可能不相同，所以感知机模型得到的超平面可能有多个。那么，支持向量机模型就是找到一个最优的分割超平面。

&#8195;&#8195;**SVM模型和感知机模型一样**。<font color=#a0000>**SVM模型的方法是：不仅要让样本点被分割超平面分开，还希望那些离分割超平面最近的点到分割超平面的距离最小。**</font>

&#8195;&#8195;SVM 和 线性分类器对比如下：

<center><img src="https://img-blog.csdnimg.cn/20190713112836366.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70">


&#8195;&#8195;支持向量机分为两种：硬间隔支持向量机和软间隔支持向量机。

</br>

### 2.1、求解硬间隔支持向量机

输入：线性可分训练集$T=\{ (\mathbf x_1 ,y_1) ,(\mathbf x_2 ,y_2), ..., (\mathbf x_n ,y_n) \}$，且$y_i \in \{-1,1\} $。

输出：分割超平面$\mathbf {w^{T* }\cdot x + b^*} = 0$ 和分类决策函数 $h(\mathbf x) =sign(\mathbf {w^Tx + b})$。

求解步骤：

1. 构造约束优化问题.
   $$
   \min_{\alpha} \ \frac{1}{2} \cdot \sum\limits_{i=1}^M \alpha_i \alpha_j y_i y_j (\mathbf x_i \cdot \mathbf x_j) - \sum\limits_{i=1}^M \alpha_i  \\
   s.t.  \quad  \sum\limits_{i=1}^M \alpha_iy_i = 0  \quad(s.t. 意思是使得...满足...)   \\
   \alpha_i ≥ 0 , i=1,2,...,M
   $$

2. 利用SMO算法求解上面的优化问题，得到$\mathbf {\alpha}$向量的值$\mathbf {\alpha^*}$,$\alpha = (\alpha_1, \alpha_2 ,.., \alpha_M)$ 是拉格朗日乘子向量，$\alpha_i ≥0$ .

3. 求解计算$\mathbf w$向量的值$\mathbf w^*$.
   $$
   \mathbf w^*= \sum\limits_{i=1}^M \alpha^*\ y_i\ \mathbf x_i
   $$

4. 找到满足$\alpha^*_s > 0$ 对应的支持向量点$(\mathbf {x_s}, y_s )$，从而求解计算$\mathbf b$ 的值$\mathbf {b^*}$.
   $$
   \mathbf b^*= \frac{1}{S}\sum\limits_{s=1}^S [ y_s - \mathbf {w^* \cdot x^s}] 
   $$

5. 由$\mathbf w^*$ 和 $\mathbf {b^*}$ 得到分割超平面 $\mathbf {w^* \cdot x + b^*} = 0$ 和分类决策函数  $h(\mathbf x) =sign(\mathbf {w^{T* } + b})$.


</br>

### 2.2、求解软间隔支持向量机

输入：线性可分训练集$T=\{ (\mathbf x_1 ,y_1) ,(\mathbf x_2 ,y_2), ..., (\mathbf x_n ,y_n) \}$，且$y_i \in \{-1,1\} $。

输出：分割超平面$\mathbf {w^{T* } \cdot x + b^*} = 0$ 。

求解步骤：

1. 构造约束优化问题.
   $$
   \min_{\alpha} \ \frac{1}{2} \cdot \sum\limits_{i=1}^M \alpha_i \alpha_j y_i y_j (\mathbf x_i \cdot \mathbf x_j) - \sum\limits_{i=1}^M \alpha_i  \\
   s.t.  \quad  \sum\limits_{i=1}^M \alpha_iy_i = 0    \\
   0 ≤ \alpha_i ≤ c ,\  i=1,2,...,M
   $$

2. 利用SMO算法求解上面的优化问题，得到$\mathbf {\alpha}$向量的值$\mathbf {\alpha^*}$,$\alpha = (\alpha_1, \alpha_2 ,.., \alpha_M)$ 是拉格朗日乘子向量，$\alpha_i ≥0$ .

3. 求解计算$\mathbf w$向量的值$\mathbf w^*$.
   $$
   \mathbf w^*= \sum\limits_{i=1}^M \alpha^*\ y_i\ \mathbf x_i
   $$

4. 找到满足$0 < \alpha^*_s < c$ 对应的支持向量点$(\mathbf {x_s}, y_s )$，从而求解计算$\mathbf b$ 的值$\mathbf {b^*}$.
   $$
   \mathbf b^*= \frac{1}{S}\sum\limits_{s=1}^S [ y_s - \mathbf {w^* \cdot x^s}]
   $$

5. 由$\mathbf w^*$ 和 $\mathbf {b^*}$ 得到分割超平面 $\mathbf {w^{T* } \cdot x + b^*} = 0$ 和分类决策函数  $h(\mathbf x) =sign(\mathbf {w^Tx + b})$ .


### 2.3、求解非线性支持向量机
&#8195;&#8195;解决非线性支持向量机的方法：定义一个低纬特征空间到高纬特征空间的映射$\phi$，利用这个映射函数，将所有特征映射到一个更高的维度，让数据线性可分，然后再利用线性方法来优化目标函数，求分离超平面和分类决策函数。即新的优化函数变为：
$$
   \min_{\alpha} \ \frac{1}{2} \cdot \sum\limits_{i=1}^M \alpha_i \alpha_j y_i y_j (\phi(\mathbf x_i) \cdot \phi(\mathbf x_j) ) - \sum\limits_{i=1}^M \alpha_i  \\
   s.t.  \quad  \sum\limits_{i=1}^M \alpha_iy_i = 0    \\
   0 ≤ \alpha_i ≤ c ,\  i=1,2,...,M
$$

同理，相应决策函数变为如下：
$$
f(\mathbf x) = sign[(\sum\limits_{i=1}^M \alpha^* y_i \mathbf x) \cdot \mathbf x + \mathbf b^*] \\
= sign\left[ \left( \sum\limits_{i=1}^M \alpha^* y_i K( \mathbf x_i, \mathbf x ) \right) + \mathbf b^* \right] 
$$

其中，$K( \mathbf x_i, \mathbf x )$ 指核函数，常用核函数如：
1. 线性核函数：$K( \mathbf x_i, \mathbf x ) =  \mathbf x_i  \cdot \mathbf x_j$
2. 高斯核函数：$K( \mathbf x_i, \mathbf x ) = exp^{( -\frac{||\mathbf x_i  \cdot \mathbf x_j||^2}{2\delta^2} )}$
3. 多项式核函数：$K( \mathbf x_i, \mathbf x ) =  [ \gamma (\mathbf x_i  \cdot \mathbf x_j )+ \nu ]^p$


  </br>
  

## 3、支持向量机实践

### 3.1、简单支持向量机

&#8195;&#8195;通过使用 sklearn.datasets 数据集实现简单的 SVM 。

```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set()

# 1、创建模拟数据集
from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=50, centers=2,
                  random_state=0, cluster_std=0.60)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');
```
输出结果：
<center><img src="https://img-blog.csdnimg.cn/20190713113031819.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >

```python
# 2、可以有多种方法分类
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(xfit, m * xfit + b, '-k')

plt.xlim(-1, 3.5);
```

输出结果：
<center><img src="https://img-blog.csdnimg.cn/20190713113117622.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >

SVM 的思想: 假想每一条分割线是有宽度的。在SVM的框架下, 认为最宽的线为最优的分割线

```python
# 3、绘制有宽带的分割线
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
                     color='#AAAAAA', alpha=0.4)

plt.xlim(-1, 3.5);
```
输出结果：
<center><img src="https://img-blog.csdnimg.cn/2019071311314837.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >


**训练SVM**
```python
# 4、使用线性SVM和比较大的 C
from sklearn.svm import SVC # "Support vector classifier"
model = SVC(kernel='linear', C=1E10)
model.fit(X, y)
```

创建一个显示SVM分割线的函数
```python
# 5、定义 SVM 分割线函数
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
 
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model);
```
输出结果：
<center><img src="https://img-blog.csdnimg.cn/20190713113238109.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70">


<br>

### 3.2、非线性支持向量机
#### 3.2.1、SVM 中使用多项式特征

**(1)、导入数据集**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

X, y = datasets.make_moons()
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
```

输出结果：
<center><img src="https://img-blog.csdnimg.cn/20190713113312167.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70">


**(2)、通过增加噪声，增大数据集标准差**

```python
X, y = datasets.make_moons(noise=0.15, random_state=666)

plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
```
输出结果：
<center><img src="https://img-blog.csdnimg.cn/20190713113334828.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >

**(3)、使用多项式核函数的SVM**

```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

def PolynomialSVC(degree, C=1.0):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("linearSVC", LinearSVC(C=C))
    ])

poly_svc = PolynomialSVC(degree=3)
poly_svc.fit(X, y)

def plot_decision_boundary(model, axis):
    
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3]-axis[2])*100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A','#FFF59D','#90CAF9'])
    
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)
```

输出结果：
<center><img src="https://img-blog.csdnimg.cn/20190713113438552.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70">


**(4)、使用多项式核函数的SVM**

```python
from sklearn.svm import SVC

def PolynomialKernelSVC(degree, C=1.0):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("kernelSVC", SVC(kernel="poly", degree=degree, C=C))
    ])

poly_kernel_svc = PolynomialKernelSVC(degree=3)
poly_kernel_svc.fit(X, y)
```

输出结果：
<center><img src="https://img-blog.csdnimg.cn/20190713113455318.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70">


</br>

### 3.3、SVM 解决回归问题
&#8195;&#8195;SVM 解决回归问题和解决分类问题的思想正好相反。即SVM解决分类问题希望在margin中的数据点越少越好，而SVM解决回归问题则希望落在margin中的数据点越多越好。

<center><img src="https://img-blog.csdnimg.cn/20190713113540202.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" width=65%>

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

boston = datasets.load_boston()
X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

def StandardLinearSVR(epsilon=0.1):
    return Pipeline([
        ('std_scaler', StandardScaler()),
        ('linearSVR', LinearSVR(epsilon=epsilon))
    ])

svr = StandardLinearSVR()
svr.fit(X_train, y_train)
svr.score(X_test, y_test)
```
 **输出 output :  0.6358806887937369**


### 3.4、SVM 实现人脸的分类识别
&#8195;&#8195; 对输入的人脸图像使用PCA(主成分分析)将图像(看作一维向量)进行了降维处理，然后将降维后的向量作为支持向量机的输入。**PCA降维的目的**可以看作是特征提取, 将图像里面真正对分类有决定性影响的数据提取出来，从而实现支持向量机人脸的分类识别.。

**（1）、导入人脸实例数据集**
```python
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)

# 人脸实例
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],
            xlabel=faces.target_names[faces.target[i]])
```
<center><img src="https://img-blog.csdnimg.cn/20190713160722962.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70">

&#8195;&#8195; 每一幅图的尺寸为 [62×47] , 大约 3000 个像素值。
&#8195;&#8195; 我们可以将整个图像展平为一个长度为3000左右的一维向量, 然后使用这个向量做为特征. 通常更有效的方法是通过预处理提取图像最重要的特征. 一个重要的特征提取方法是PCA(主成分分析), 可以将一副图像转换为一个长度为更短的(150)向量。
```python
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
pca = PCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='linear', class_weight='balanced')
model = make_pipeline(pca, svc)
```

**（2）、将数据分为训练和测试数据集**
```python
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target,
                                                random_state=42)
```

**（3）、调参**
**通过交叉验证寻找最佳的 C (控制间隔的大小)**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'svc__C': [1, 5, 10, 50]}
grid = GridSearchCV(model, param_grid)

grid.fit(Xtrain, ytrain)

model = grid.best_estimator_
yfit = model.predict(Xtest)
```

**（4）、使用训练好的SVM做预测**
```python
fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                   color='black' if yfit[i] == ytest[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14);
```
预测结果：
<center><img src="https://img-blog.csdnimg.cn/20190713161143925.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70">

**（5）、生成报告与混淆矩阵**
```python
from sklearn.metrics import classification_report
print(classification_report(ytest, yfit,
                            target_names=faces.target_names))

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');
```
**报告数据**
<center><img src="https://img-blog.csdnimg.cn/20190713161322362.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70">

**混淆矩阵**
<center><img src="https://img-blog.csdnimg.cn/20190713161442494.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70">

</br>

## 4、SVM总结                                                                                                                                                                                                                                                    
### 4.1、优点                                                                                                                                                   
1. SVM模型具有很好的泛化能力，特别是在小样本训练集上能比其他算法得到好很多的结果。其本身的优化目标是结构化风险最小，通过引入 margin 的概念，可以得到数据分布的结构化描述，降低了对数据规模和数据分布的要求；
2. 模型只需要保存支持向量, 模型占用内存少, 预测快；
3. SVM 模型具有较强的数据理论支撑；
4. 分类只取决于支持向量, 适合数据的维度高的情况, 例如DNA数据；
5. SVM 模型引入核函数之后可以解决非线性分类问题，并且 SVM 模型还可以解决回归问题。

### 4.2、缺点
1. SVM 模型不适合解决多分类问题，对于多分类问题，只能采用一对多模式间接实现；
2. SVM 模型存在两个对结果影响比较大的超参数，如超参数惩罚项系数 c ，所以复杂度比一般非线性模型要高，需要做调参 $c$ 当数据量大时非常耗时间；
3.  训练的时间复杂度为 $\mathcal{O}[N^3]$ 或者至少 $\mathcal{O}[N^2]$, 当数据量巨大时候不合适使用。

### 4.4、SVM模型与 LR模型的异同
1. SVM 与 LR 模型都是有监督的机器学习算法，且都属于判别模型；
2. 如果不考虑 SVM 核函数情况，两者都属于是线性分类算法 ;
3. SVM模型与 LR模型的构造原理不同；
4. SVM模型与 LR模型在学习时考虑的样本点不同 ；
5. SVM 模型中自带正则化项 $\frac{1}{2}||\mathbf w||^2$，和 LR模型相比，不容易产生过拟合问题。




---
> 文中实例及参考：
> + 刘宇波老师《Python入门机器学习》
> + 《机器学习基础》
> + 贪心学院，[https://www.greedyai.com](https://www.greedyai.com)
---





