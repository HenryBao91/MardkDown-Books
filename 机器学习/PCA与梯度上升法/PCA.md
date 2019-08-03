# <center>PCA</center>



## 1、主成分分析

### 1.1、主成分分析介绍

​		主成分分析（PCA）是一种非常重要的<font color='red'>**无监督降维学习方法**</font>。其基本思想：找出原始数据最主要的方面来替代原始数据，使得在损失少部分原始信息的基础上极大地降低原始数据的维度。

​		这一方法利用正交变换把由线性相关变量表示的观测数据转换为由少数几个由线性无关变量表示的数据，线性无关的变量称为主成分。主成分的个数通常小于原始变量的个数，所以主成分分析属于降维方法。

​		主成分分析有以下特点：

- 是一个无监督的机器学习算法

- 主要用于数据的降维

- 通过降维，可以发现更便于理解的特征

- 其他应用：可视化、去噪

  ​	对于二维平面的一些数据点：

  ![1564822310404](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\1564822310404.png)

  ​	对于上述样本点降维，可以分别只保留特征1和特征2，即分别降到特征1或者特征2的维度上，如下图：

  ![111111111111111111111](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\111111111111111111111.jpg)

  ​		降维后的效果如下图：

  ![22222222221](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\22222222221.jpg)

  ​	由图对比可知，左边的降维方案更好，因为点和点之间的距离比较大，也就是点与点之间有较大的区分度，并且保留了之前的区分度，而右侧的区分度则比较小。

  ​	降维遵循两个基本原则：

    1. 样本点到降维后的方向直线的距离更近；

    2. 样本点在降维后方向直线上的投影尽可能地分开，即同时满足紧凑性和可分性的要求。

       所以，就需要找到让样本间间距最大的轴，可以找到更好的降维方向，如下：

  ![未命名_meitu_1](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\未命名_meitu_1-1564823269612.jpg)

  

### 1.2、主成分分析降维方法

​		在主成分分析中采用方差的方法来定义样本间距，从而找到样本间间距最大的轴。方差是一个描述样本分布疏密程度的一个统计量。
$$
Var(x) = \frac{1}{m} \sum\limits^m_{i=1} (x_i - \overline x)^2
$$
​		所以，找到上述轴的方法，就是是的样本空间的所有样本点映射到这个轴之后，方差最大。方法有如下几步：

​		**第一步：**将样本数据的均值归为0（demean），即相当于把坐标轴进行了平移：

![1564823805997](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\1564823805997.png)

​		此时，样本的均值 $\overline x = 0$ ，所以方差公式变为 $Var(x) = \frac{1}{m} \sum\limits^m_{i=1} x_i ^2$

​		**第二步：**得到轴的方向，在二维平面中，如 $w=(w_1,w_2)$ ，使得所有样本映射到 w 之后，有方差最大，此时方差为（此时，映射后的均值$\overline X_{project}=0$）：
$$
\begin{aligned}
Var(X_{project}) &= \frac{1}{m}\sum\limits^m_{i=1}|| X_{project}^{(i)} - \overline X_{project} ||^2 \\
&= \frac{1}{m}\sum\limits^m_{i=1}|| X_{project}^{(i)} ||^2 
\end{aligned}
$$
​		使得 $Var(X_{project}) = \frac{1}{m}\sum\limits^m_{i=1}|| X_{project}^{(i)} ||^2 $ 最大，推导如下：

​	![1564824576179](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\1564824576179.png)

​		即将原始样本点 X 映射到 w 轴之后投影的长度，每个样本点代表的向量与 w 轴方向向量进行点乘之后得到的投影平方和最大。

​		**目标：**求 $w$ 使得  $Var(X_{project})$ 最大，对于 N 维情况下：
$$
\begin{aligned}
Var(X_{project}) &= \frac{1}{m}\sum\limits^m_{i=1}( X^{(i)} \cdot w)^2 \\
&= \frac{1}{m}\sum\limits^m_{i=1}( X^{(i)}_1 \cdot w_1 + X^{(i)}_2 \cdot w_2 +... +X^{(i)}_n \cdot w_n)^2  \\
&= \frac{1}{m}\sum\limits^m_{i=1}( \sum\limits^n_{j=1} X^{(i)}_j \cdot w_j )^2
\end{aligned}
$$
​		求一个目标函数的最优化问题，使用梯度上升法解决。梯度上升法即梯度代表方向，对应目标函数 $J$ 增大的最快方向：
$$
+\eta \cdot \frac{dJ}{d \theta}
$$
​		PCA 降维除了使用梯度上升法之外，同样可以使用数学求解的方法。

​		同时，对于线性回归和主成分分析两种方法区别的对比如下：

![未命名_meitu_1](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\未命名_meitu_1-1564825230723.jpg)



​		

## 1.3、主成分分析法推导

​		**目标：**求 $w$ ，使得 $f(X)=\frac{1}{m}\sum\limits^m_{i=1}( X^{(i)}_1 \cdot w_1 + X^{(i)}_2 \cdot w_2 +... +X^{(i)}_n \cdot w_n)^2$ 最大。

$$
\nabla f = 
\begin{pmatrix}
	\frac{\partial f}{\partial w_1} \\
	\frac{\partial f}{\partial w_2} \\
	\vdots  \\
	\frac{\partial f}{\partial w_n} 
\end{pmatrix}
= \frac{2}{m}
\begin{pmatrix}
        \sum\limits^m_{i=1}( X^{(i)}_1 \cdot w_1 + X^{(i)}_2 \cdot w_2 +... +X^{(i)}_n \cdot w_n)X^{(i)}_1  \\
        \sum\limits^m_{i=1}( X^{(i)}_1 \cdot w_1 + X^{(i)}_2 \cdot w_2 +... +X^{(i)}_n \cdot w_n)X^{(i)}_2  \\
        \vdots  \\
        \sum\limits^m_{i=1}( X^{(i)}_1 \cdot w_1 + X^{(i)}_2 \cdot w_2 +... +X^{(i)}_n \cdot w_n)X^{(i)}_n   \\
\end{pmatrix} \\
$$

​		上式进行展开并进行向量化化简如下：
$$
\begin{aligned}
\nabla f &= 
\begin{pmatrix}
	\frac{\partial f}{\partial w_1} \\
	\frac{\partial f}{\partial w_2} \\
	\vdots  \\
	\frac{\partial f}{\partial w_n} 
\end{pmatrix}
= \frac{2}{m}
\begin{pmatrix}
\sum\limits^n_{i=1} ( X^{(i)}_i \cdot w_i ) \cdot X^{(i)}_1 \\
\sum\limits^n_{i=1} ( X^{(i)}_i \cdot w_i ) \cdot X^{(i)}_2 \\
\vdots  \\
\sum\limits^n_{i=1} ( X^{(i)}_i \cdot w_i ) \cdot X^{(i)}_n \\
\end{pmatrix} \\

&= \frac{2}{m}(X^{(1)} w, X^{(2)} w , ... , X^{(m)} w)
\cdot
\begin{pmatrix}
\mathbf X_1^{(1)} & \mathbf X_2^{(1)}  & \cdots  & \mathbf X_n^{(1)} \\
\mathbf X_1^{(2)} & \mathbf X_2^{(2)}  & \cdots  & \mathbf X_n^{(2)} \\
\vdots  & \vdots &  \ddots  & \vdots    \\
\mathbf X_1^{(m)} & \mathbf X_2^{(m)}  & \cdots  & \mathbf X_n^{(m)}
\end{pmatrix}  \\
&= \frac{2}{m}\cdot (\mathbf Xw)^T \cdot \mathbf X  \\
&= \frac{2}{m}\cdot \mathbf X^T \cdot (\mathbf Xw)
\end{aligned}
$$

​		

## 2、梯度上升法求解PCA

### 2.1、普通数据测试

1. 数据准备：

```python
import numpy as np
import matplotlib.pyplot as plt
X = np.empty((100, 2))
X[:,0] = np.random.uniform(0., 100., size=100)
X[:,1] = 0.75 * X[:,0] + 3. + np.random.normal(0, 10., size=100)
plt.scatter(X[:, 0], X[:, 1])
plt.show()
```

![1564831955463](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\1564831955463.png)

	2. demean :

```python
def demean(X):
    # X 是一个矩阵，每一行代表一个样本，所以这里操作实际是每一个样本中的每一个特征
    # 都要减去这个特征对应的均值
    return X - np.mean(X, axis=0)
    # axis=0 ,在行方向上求均值，最终求的结果就是每一列的均值，求均值得到一个1*n向量
    
X_demean = demean(X)
plt.scatter(X_demean[:, 0], X_demean[:, 1])
plt.show()
```

![1564832216224](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\1564832216224.png)

> ```python
> np.mean(X_demean[:,0])  =  -7.531752999057062e-15
> np.mean(X_demean[:,1])  =  -1.2363443602225743e-14
> ```

3. 梯度上升法实现

```python
# 求解目标函数
def f(w, X):
    return np.sum((X.dot(w)**2)) / len(X)

# 求目标函数梯度数学求解
def df_math(w, X):
    return X.T.dot(X.dot(w)) * 2. / len(X)

# 求目标函数梯度调试方法
def df_debug(w, X, epsilon=0.0001):  
    # 在PCA中，w 是指的方向向量，模是1，每一个维度都很小，所以epsilon小一些
    res = np.empty(len(w))
    for i in range(len(w)):
        w_1 = w.copy()
        w_1[i] += epsilon
        w_2 = w.copy()
        w_2[i] -= epsilon
        res[i] = (f(w_1, X) - f(w_2, X)) / (2 * epsilon)
    return res

def direction(w):   # w 应是单位方向向量，模应该等于1
    # 在进行完每次w = w + eta * gradient之后，w的模有可能不为1
    #这里对 w 的模进行归一
    return w / np.linalg.norm(w)  # norm 即求向量模的函数

def gradient_ascent(df, X, initial_w, eta, n_iters = 1e4, epsilon=1e-8):
    w = direction(initial_w) 
    cur_iter = 0

    while cur_iter < n_iters:
        gradient = df(w, X)
        last_w = w
        w = w + eta * gradient
        w = direction(w) # 注意1：每次求一个单位方向，否则需要更多次的迭代搜索
        if(abs(f(w, X) - f(last_w, X)) < epsilon):
            break
            
        cur_iter += 1

    return w
```

	4. 调用梯度上升法

```python
# 注意2：不能用0向量开始
initial_w = np.random.random(X.shape[1])
print(initial_w)
```

> ````python
> array([0.37409902, 0.48471663])
> ````

```python
# 注意3：不能使用StandardScaler标准化数据，因为PCA过程本身是要求一个轴，是原始数据
# 映射到这个轴上之后样本的方差最大，而标准化之后数据的方差为1，则方差最大值就不存在
# 这样，就无法求PCA
eta = 0.001

gradient_ascent(df_math, X_demean, initial_w, eta)
```

> ```python
> array([0.80736806, 0.59004814])
> ```

```python
plt.scatter(X_demean[:,0], X_demean[:,1])
# 由于 w 是单位向量，非常小，所以乘以30
plt.plot([0, w[0]*30], [0, w[1]*30], color='r') 
plt.show()
```

![1564839082382](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\1564839082382.png)



### 2.2、极端数据测试

1. 原始数据：

```python
X2 = np.empty((100, 2))
X2[:,0] = np.random.uniform(0., 100., size=100)  # 没有噪音
X2[:,1] = 0.75 * X2[:,0] + 3.

plt.scatter(X2[:,0], X2[:,1])
plt.show()
```

![1564839149239](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\1564839149239.png)

2. 进行PCA降维：

```python
X2_demean = demean(X2)
w2 = gradient_ascent(df_math, X2_demean, initial_w, eta)
plt.scatter(X2_demean[:,0], X2_demean[:,1])
plt.plot([0, w2[0]*30], [0, w2[1]*30], color='r')
plt.show()
```

![1564839198604](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\1564839198604.png)





## 3、求前N个主成分

### 3.1、求其他主成分方法

​	求其他主成分的方法：对数据进行改变，例如求第二个主成分，即将数据在第二个主成分上的分量减去，在新的数据上求出的第一主成分也就是原来数据相应的第二主成分，同理，可求第二、第三主成分等。

![1564843443203](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\1564843443203.png)

​		

### 3.2、求前N主成分实践

1. 原始数据

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.empty((100, 2))
X[:,0] = np.random.uniform(0., 100., size=100)
X[:,1] = 0.75 * X[:,0] + 3. + np.random.normal(0, 10., size=100)

def demean(X):
    return X - np.mean(X, axis=0)

X = demean(X)
plt.scatter(X[:,0], X[:,1])
plt.show()
```

![1564844019178](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\1564844019178.png)

2. 梯度上升法求主成分

```python
def f(w, X):
    return np.sum((X.dot(w)**2)) / len(X)

def df(w, X):
    return X.T.dot(X.dot(w)) * 2. / len(X)

def direction(w):
    return w / np.linalg.norm(w)

def first_component(X, initial_w, eta, n_iters = 1e4, epsilon=1e-8):
    
    w = direction(initial_w) 
    cur_iter = 0

    while cur_iter < n_iters:
        gradient = df(w, X)
        last_w = w
        w = w + eta * gradient
        w = direction(w) 
        if(abs(f(w, X) - f(last_w, X)) < epsilon):
            break
            
        cur_iter += 1

    return w
```

3. 求解第一主成分

```python
initial_w = np.random.random(X.shape[1])
eta = 0.01
w = first_component(X, initial_w, eta)
```

> ```python
> initial_w = array([0.62172241, 0.84712175])
> w = array([0.76301162, 0.64638476])
> ```

4. 原始数据减去第一主成分

```python
X2 = np.empty(X.shape)
X2 = X - X.dot(w).reshape(-1, 1) * w
    
plt.scatter(X2[:,0], X2[:,1])
plt.show()         
```

![1564845597979](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\1564845597979.png)

5. 求解第二主成分

```python
w2 = first_component(X2, initial_w, eta)

# 第一主成分方向和第二主成分方向是垂直的，点乘应该等于0 ，以下结果几乎为0
w.dot(w2)
```

>```python
>w2 = array([-0.64638109,  0.76301474])
>w.dot(w2) = 4.81511868927198e-06 ≈ 0 
>```

6. 封装求解前N主成分函数

```python
def first_n_components(n, X, eta=0.01, n_iters = 1e4, epsilon=1e-8):
    X_pca = X.copy()
    X_pca = demean(X_pca)
    res = []
    for i in range(n):
        initial_w = np.random.random(X_pca.shape[1])
        w = first_component(X_pca, initial_w, eta)
        res.append(w)
        
        X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w
        
    return res

# 调用函数求第二主成分
first_n_components(2, X)
```

>```python
>[array([0.7630117 , 0.64638468]),    	# 第一主成分
> array([-0.64638205,  0.76301392])]		# 第二主成分
>```