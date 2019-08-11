# <center>PCA</center>



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

​	