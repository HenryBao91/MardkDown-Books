# <center>PCA 原理（一）</center>



## 1、PCA 原理推导

​	假设有 M 个样本 $x_1, x_2, ..., x_m$ ，每个样本点 $x_i$ 含有 N 个特征，则每个样本数据可以表示为： $x_i =( x^{(1)}_i, (x^{(2)}_i,..., (x^{(n)}_i)$，整体样本数据在 N 维特征空间的原始坐标系为 $I=(i_1, i_2, ..., i_n)$， $I$ 是一组标准正交基，即有如下性质：
$$
\begin{aligned}
||i_s||_2 =&  \ 1 \\
i_s^T \cdot i_t =& \ 0 \ ,  \ s ≠ t
\end{aligned}
$$
​	样本点 $x_i$ 在原始坐标系中的表示为：
$$
x_i= (i_1, i_2, ...,i_n)\cdot 
\begin{pmatrix}
	x^{(1)}_i \\
	x^{(2)}_i \\
	\vdots  \\
	x^{(n)}_i
\end{pmatrix}
\ , \ i=1,2,3...,m
$$
​	假设进行线性变换之后得到的新的坐标系为 $J=(j_1, j_2, ..., j_{n'})$， $J$ 同样是一组标准正交基，即满足上述正交基的性质。则样本点 $x_i$ 在新的坐标系中的近似表示为：
$$
\mathop{{x_i}}\limits^{\sim } = (j_1, j_2, ...,j_{n'})\cdot 
\begin{pmatrix}
	z^{(1)}_i \\
	z^{(2)}_i \\
	\vdots  \\
	z^{(n')}_i
\end{pmatrix}
\ , \ i=1,2,3...,m
$$
​	根据正交基的性质， $j_s$ 可以等价于：
$$
j_s= (i_1, i_2, ...,i_n)\cdot
\begin{pmatrix}
	j_s \cdot i_1 \\
	j_s \cdot i_2 \\
	\vdots  \\
	j_s \cdot i_n \\
\end{pmatrix}
\ , \ s=1,2,3...,m'
$$
​	令：
$$
w_s= 
\begin{pmatrix}
	j_s \cdot i_1 \\
	j_s \cdot i_2 \\
	\vdots  \\
	j_s \cdot i_n \\
\end{pmatrix}
\ , \ s=1,2,3...,m'
$$
​	则 $w_s$ 是一个新的基向量，其各分量就是基向量 $j_s$ 在原始坐标系 $(i_1, i_2, ..., i_n)$ 中的投影。所以，$j_s$ 可以写为：
$$
j_s = (i_1, i_2, ...,i_n)\cdot w_s \ , \ s =1 ,2 ,..., n'
$$
​	根据正交基性质，有 $||w_s||_2\ =\ 1 \ , \ w_s^T \cdot w_t = \ 0 \ , \ s≠t$。

​	类似的有 $w_1,w_2,...,w_{n'}$，将其写成矩阵形式为：
$$
W = [w_1, w_2, ...,w_{N'}] =
\begin{pmatrix}
	j_1 \cdot i_1 & j_2 \cdot i_1 & \cdots & j_{n'} \cdot i_1\\
	j_1 \cdot i_2 & j_2 \cdot i_2 & \cdots & j_{n'} \cdot i_2\\
	\vdots  & \vdots     &   \ddots   & \vdots \\
	j_1 \cdot i_n & j_2 \cdot i_n & \cdots & j_{n'} \cdot i_n\\
\end{pmatrix}
$$
​	则 W 就称为<font color=red>**坐标变换矩阵**</font>，且有 $W = W^T, \  WW^T =I$ 。根据坐标变换矩阵，新坐标系和原始坐标系直接的关系可以表示为：
$$
(j_1, j_2, ...,j_{n'}) = (i_1, i_2, ...,i_n)\cdot W
$$
​	将其带入前面 $x_i$ 在新坐标系中的近似表达式，可得：
$$
\mathop{{x_l}}\limits^{\sim} = (j_1, j_2, ...,j_{n'})\cdot 
\begin{pmatrix}
	z^{(1)}_i \\
	z^{(2)}_i \\
	\vdots  \\
	z^{(n')}_i
\end{pmatrix}
= (i_1, i_2, ...,i_n) W \cdot 
\begin{pmatrix}
	z^{(1)}_i \\
	z^{(2)}_i \\
	\vdots  \\
	z^{(n')}_i
\end{pmatrix}
$$
​	再将其与 $x_i$ 在原始坐标系中的表达式 $x_i = (i_1, i_2, ...,i_n)\cdot \begin{pmatrix}x^{(1)}_i \\ x^{(2)}_i \\\vdots  \\  x^{(n)}_i \end{pmatrix}$ 比较可知，通过坐标变换来降维，相当于是用 $Wz_i$ 去近似表示了 $x_i$ ，使：
$$
x_i = W z_i
$$
​	即：
$$
z_i = W^{-1}x_i=W^Tx_i
$$
​	则有：
$$
z_i = w_s^Tx_i \ , \quad\quad\quad\quad s=1,2,3...,n'
$$


 	一般，$n'$ 会远小于 $n$ ，这样就可以达到降维的目的了。将维度由 $M$ 降到 $M'$ ，相当于人为的丢弃了部分坐标。我们的要求是：**基于将为后的坐标重构样本时，得到的重构样本与原始原本坐标尽量相同。对于样本点 $x_i$ 来说，即要使 $W z_i$ 和 $x_i$ 的距离最小化，推广到整体样本点，即：
$$
min \sum\limits_{i=1}^M|| W z_i - x_i ||^2_2
$$
​		先计算 $|| W z_i - x_i ||^2_2$，即
$$
\sum\limits_{i=1}^M|| W z_i - x_i ||^2_2 \\
\begin{aligned} 
&= (\sum\limits_{i=1}^Mx_i^Tx_i) - tr[W^T(\sum\limits_{i=1}^Mx_i^Tx_i)W]  \\
&=  (\sum\limits_{i=1}^Mx_i^Tx_i) -  tr(W^TXX^TW)
\end{aligned}
$$
​		因为对于给定 M 个样本，$\sum\limits^n_{i=1}x_i^Tx_i$ 是一个固定值，因此最小化上面的结果等价于：
$$
min \ - tr(W^TXX^TW)  \\
s.t. \quad W^TW=I
$$
​		构造拉格朗日函数：
$$
L(W) = - tr(W^TXX^TW) + \lambda(W^TW-I)
$$
​		对 $W$ 求导，可得：
$$
- XX^TW + \lambda W = 0
$$
​		移项可得；
$$
 XX^TW = \lambda W
$$
​		可以看出，**坐标变换矩阵 $W$ 为 $XX^T$ 的 $M'$ 个特征向量组成的矩阵，而 $\lambda$ 为 $XX^T$ 的特征值。当我们将原始数据集从 $N$ 维降到 $N’$ 维时，只需要找到 $XX^T$ 最大的 $N’$ 个特征值对应的特征向量，将其组成坐标变换矩阵（投影矩阵） $W$ ，然后利用$z_i = W^Tx_i$ 即可实现降维的目的。





---
> 文中实例及参考：
>
> + 《机器学习基础》

---