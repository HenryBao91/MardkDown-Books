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



## 4、高纬度向低纬度映射

### 4.1、推导

$$
X =
\begin{pmatrix}
\mathbf X_1^{(1)} & \mathbf X_2^{(1)}  & \cdots  & \mathbf X_n^{(1)} \\
\mathbf X_1^{(2)} & \mathbf X_2^{(2)}  & \cdots  & \mathbf X_n^{(2)} \\
\vdots  & \vdots &  \ddots  & \vdots    \\
\mathbf X_1^{(m)} & \mathbf X_2^{(m)}  & \cdots  & \mathbf X_n^{(m)}
\end{pmatrix}
\quad \quad \quad 
W_k =
\begin{pmatrix}
\mathbf X_1^{(1)} & \mathbf X_2^{(1)}  & \cdots  & \mathbf X_n^{(1)} \\
\mathbf X_1^{(2)} & \mathbf X_2^{(2)}  & \cdots  & \mathbf X_n^{(2)} \\
\vdots  & \vdots &  \ddots  & \vdots    \\
\mathbf X_1^{(k)} & \mathbf X_2^{(k)}  & \cdots  & \mathbf X_n^{(k)}
\end{pmatrix}  \\
$$

​	高纬度到低纬度降维：
$$
\ \quad\quad X \cdot W_k^T \\
(m \cdot n )  \ *  (n \cdot k)    \ \Rightarrow  \ m\cdot k
$$
​	低纬度到高纬度恢复,	**低->高，恢复（有损失）**：
$$
 \quad\quad X_k \cdot W_k^T \\
(m \cdot k )  \ *  (k \cdot n)    \ \Rightarrow  \ m\cdot n
$$


### 4.2、降维实践

1. 封装降维代码类：

```python
# -*- coding: utf-8 -*-
'''
# @Time : 2019/7/29 22:17 
# @Author : Hong Zhen
# @Software: PyCharm
'''
import numpy as np

class PCA:

    def __init__(self, n_components):
        """初始化PCA"""
        assert n_components >= 1, "n_components must be valid"
        self.n_components = n_components
        self.components_ = None


    def fit(self, X, eta=0.01, n_iters=1e4):
        """获得数据集X的前n个主成分"""
        assert self.n_components <= X.shape[1], \
        "n_components must not be greater than the feature number of X"


        def demean(X):
            return X - np.mean(X, axis=0)

        def f(w, X):
            return np.sum((X.dot(w) ** 2)) / len(X)

        def df(w, X):
            return X.T.dot(X.dot(w)) * 2. / len(X)

        def direction(w):
            return w / np.linalg.norm(w)

        def first_component(X, initial_w, eta=0.01, n_iters=1e4, epsilon=1e-8):

            w = direction(initial_w)
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = df(w, X)
                last_w = w
                w = w + eta * gradient
                w = direction(w)
                if (abs(f(w, X) - f(last_w, X)) < epsilon):
                    break

                cur_iter += 1

            return w

        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))
        for i in range(self.n_components):
            # 初始搜索方向 w
            initial_w = np.random.random(X_pca.shape[1])
            # 搜索此时的 PCA 对应的主成分
            w = first_component(X_pca, initial_w, eta, n_iters)
            self.components_[i, :] = w

            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

        return self


    def transform(self, X):
        """将给定的X，映射到各个主成分分量中"""
        assert X.shape[1] == self.components_.shape[1]  # 列

        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        """将给定的X，反向映射回原来的特征空间"""
        assert X.shape[1] == self.components_.shape[0]  # 行

        return X.dot(self.components_)


    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components
```

2. 降维测试

```python
X = np.empty((100, 2))
X[:,0] = np.random.uniform(0., 100., size=100)
X[:,1] = 0.75 * X[:,0] + 3. + np.random.normal(0, 10., size=100)

pca = PCA(n_components=2)
pca.fit(X)
print(pca.components_)
```

> ```python
> # 两个坐标轴的方向
> array([[ 0.79412129,  0.60775931],
>        [ 0.60776146, -0.79411964]])
> ```

```python
pca = PCA(n_components=1)
pca.fit(X)
X_reduction = pca.transform(X)
print(X_reduction.shape)
```

> ```python
> (100, 1)
> ```

```python
# 数据恢复
X_restore = pca.inverse_transform(X_reduction)
print(X_restore.shape)

# 恢复数据后会丢失数据信息
plt.scatter(X[:,0], X[:,1], color='b', alpha=0.5)
plt.scatter(X_restore[:,0], X_restore[:,1], color='r', alpha=0.5)
plt.show()
```

> ```python
> (100, 2)
> ```

![1565012256746](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\1565012256746.png)

### 4.3、scikit-learn中的PCA

#### 4.3.1、降维与数据恢复

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(X)
# 和自己求的主成分的方向相反，因为sklearn中实现pca的方向不是使用梯度上升法
# 由于只是方向相反，并不影响降维
pca.components_
```

> ```python
> array([[-0.79601531, -0.60527648]])
> ```

```python
# 降维
X_reduction = pca.transform(X)
print(X_reduction.shape)
```

> ```python
> array([[-0.79601531, -0.60527648]])
> ```

```python
# 恢复数据
X_reduction = pca.transform(X)
X_reduction.shape
```

> ```python
> (100, 2)
> ```

```python
# 绘制结果相同
plt.scatter(X[:,0], X[:,1], color='b', alpha=0.5)
plt.scatter(X_restore[:,0], X_restore[:,1], color='r', alpha=0.5)
plt.show()
```

![1565012496755](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\1565012496755.png)

#### 4.3.2、真实数据测试

1. 准备数据

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()   # 使用手写识别数据
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
print(X_train.shape)
```

> ```python
> (1347, 64)
> ```

2. 原始进行分类测试

```python
%%time
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
print(knn_clf.score(X_test, y_test))
```

> ```python
> Wall time: 22.9 ms
> 0.9866666666666667
> ```

3. 使用PCA降维

```python
pca = PCA(n_components=2)
pca.fit(X_train)

# 不能根据 X_test 再重新训练一个 pca ，所以 必须使用训练数据集得到的 pca
# transform(X_test) ， 这样才能验证整个算法的准确度

X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)

%%time 
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_reduction, y_train)

print(knn_clf.score(X_test_reduction, y_test))
```

> ```python
> # 通过降维后将大大节省计算时间
> Wall time: 998 µs
> # 精度也被降低
> 0.6066666666666667
> ```

#### 4.3.3、PCA所解释的方差

​		上面测试降维到2维之后，测试准确率只有0.606，过于低下，在sklearn-PCA中提供了解释方差的方法。

```python
pca.explained_variance_ratio_
```

> ```python
> array([0.14566817, 0.13735469])
> ```

​		上面的输出代表降维到2维之后，在每个维度上保留的数据可以解释的方差的比例，所以降到2维后，整个数据维度上可解释的方差为：$0.14 + 0.13$左右。

```python
pca = PCA(n_components=X_train.shape[1])  # 64 个维度
pca.fit(X_train)
# 对于 64 个主成分来说，依次可以解释的方差的值，可以表示每个维度相应的重要程度
pca.explained_variance_ratio_
```

> ```python
> array([1.45668166e-01, 1.37354688e-01, 1.17777287e-01, 8.49968861e-02,
>        5.86018996e-02, 5.11542945e-02, 4.26605279e-02, 3.60119663e-02,
>        3.41105814e-02, 3.05407804e-02, 2.42337671e-02, 2.28700570e-02,
>       		 ...			...			...			...
>        1.23186515e-06, 1.05783059e-06, 6.06659094e-07, 5.86686040e-07,
>        1.71368535e-33, 7.44075955e-34, 7.44075955e-34, 7.15189459e-34])
> ```

​		上面可以求解出原始数据64个维度上，每个维度可解释的方差为比例为多少。可以对各个维度可解释方差累积和的图像：

```python
plt.plot([i for i in range(X_train.shape[1])], 
         [np.sum(pca.explained_variance_ratio_[:i+1]) for i in range(X_train.shape[1])])
plt.show()
```

![1565014853419](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\1565014853419.png)

#### 4.3.4、选择留可解释方差比例降维

​		当不知道需要降至多少维，能够满足可解释方差比例时，可以通过sklearn中的方式实现：

```python
pca = PCA(0.95)   # 代表要做pca之后能解释95%以上的方差
pca.fit(X_train)
print(pca.n_components_)
```

>```python
>28   # 代表降维至 28维可以满足95%的可解释方差比例
>```

```python
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)
%%time 
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_reduction, y_train)
print(knn_clf.score(X_test_reduction, y_test))
```

> ```python
> Wall time: 3.02 ms
> 0.98
> ```

#### 4.3.4、PCA对数据进行降维可视化

```python
pca = PCA(n_components=2)
pca.fit(X)
X_reduction = pca.transform(X)

for i in range(10):   # 自动为每一次绘制指定不同颜色
    plt.scatter(X_reduction[y==i,0], X_reduction[y==i,1], alpha=0.8)
plt.show()
```

![1565015527695](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\1565015527695.png)



## 5、MNIST 数据集实践

1. 下载MNIST数据集

```python
import numpy as np 
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')
mnist
```

> ```python
> {'COL_NAMES': ['label', 'data'],
>  'DESCR': 'mldata.org dataset: mnist-original',
>  'data': array([[0, 0, 0, ..., 0, 0, 0],
>         [0, 0, 0, ..., 0, 0, 0],
>         [0, 0, 0, ..., 0, 0, 0],
>         ..., 
>         [0, 0, 0, ..., 0, 0, 0],
>         [0, 0, 0, ..., 0, 0, 0],
>         [0, 0, 0, ..., 0, 0, 0]], dtype=uint8),
>  'target': array([ 0.,  0.,  0., ...,  9.,  9.,  9.])}
> ```

2. 数据准备

```python
X, y = mnist['data'], mnist['target']
X_train = np.array(X[:60000], dtype=float)
y_train = np.array(y[:60000], dtype=float)
X_test = np.array(X[60000:], dtype=float)
y_test = np.array(y[60000:], dtype=float)
```

> ```python
> # 训练集大小
> X_train.shape = (60000, 784)
> y_train.shape = (60000,)
> 
> # 测试集大小
> X_test.shape = (10000, 784)
> y_test.shape = (10000,)
> ```

3. 使用KNN分类

```python
from sklearn.neighbors import KNeighborsClassifier

knnknn_clf = KNeighborsClassifier()
%time knn_clf.fit(X_train, y_train)
```

> ```python
> CPU times: user 57.6 s, sys: 681 ms, total: 58.3 s
> Wall time: 59.4 s
> ```
>
> 
>
> ```python
> KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
>            metric_params=None, n_jobs=1, n_neighbors=5, p=2,
>            weights='uniform')
> ```

```python
%time knn_clf.score(X_test, y_test)
```

>```python
>CPU times: user 14min 20s, sys: 4.3 s, total: 14min 24s
>Wall time: 14min 29s
>```
>
>
>
>```python
>0.96879999999999999
>```

4. 使用 PCA 降维后进行分类

```python
from sklearn.decomposition import PCA 

pca = PCA(0.90)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)

knn_clf = KNeighborsClassifier()
%time knn_clf.fit(X_train_reduction, y_train)
```

> ```python
> CPU times: user 588 ms, sys: 5.23 ms, total: 593 ms
> Wall time: 593 ms
> ```
>
> 
>
> ```python
> KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
>            metric_params=None, n_jobs=1, n_neighbors=5, p=2,
>            weights='uniform')
> ```

```python
%time knn_clf.score(X_test_reduction, y_test)
```

> ```python
> CPU times: user 1min 55s, sys: 346 ms, total: 1min 56s
> Wall time: 1min 56s
> ```

**降维去除了噪音，有可能准确率更高！**

> ```python
> 0.9728
> ```



## 6、PCA 降噪

**降维的过程可以理解成是去噪**

**手写识别的例子**

1. 准备数据

```python
from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y = digits.target 

# 添加噪音
noisy_digits = X + np.random.normal(0, 4, size=X.shape) # 均值为0，方差为4
```

2. 绘制其中部分数字（加噪音）

```python
example_digits = noisy_digits[y==0,:][:10]
for num in range(1,10):
    example_digits = np.vstack([example_digits, noisy_digits[y==num,:][:10]])
print(example_digits.shape)

plot_digits(example_digits)
```

> ```python
> (10, 64)
> ```

![1565095438470](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\1565095438470.png)

3. PCA 降噪

```python
pca = PCA(0.5).fit(noisy_digits)
print(pca.n_components_)

components = pca.transform(example_digits)
filtered_digits = pca.inverse_transform(components)
plot_digits(filtered_digits)
```

> ```python
> 12
> ```

![1565095745802](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\1565095745802.png)



## 7、特征脸

### 7.1、人脸识别库

		1. 导入数据

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

faces = fetch_lfw_people()
```

![1565096445682](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\1565096445682.png)

```python
 # 对于每一个样本以二维的形式展现出来，第一个维度是样本总数， 2914 = 62 * 47
faces.images.shape 
```

> ```python
> (13233, 62, 47)
> ```

2. 绘制数据

```python
random_indexes = np.random.permutation(len(faces.data))
X = faces.data[random_indexes]
example_faces = X[:36,:]   # 取出前 36 个
print(example_faces.shape)  # (36, 2914)

plot_faces(example_faces)
```

![1565096962578](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\1565096962578.png)



### 7.2、特征脸

​		特征脸，即将主成分中的每一行都当做一个样本来看待，如果存储的是人脸图像数据，那么主成分越靠前越能反映人脸的特征。

```python
%%time
from sklearn.decomposition import PCA 
pca = PCA(svd_solver='randomized')   # 采用随机的方式，求所有主成分
pca.fit(X)
```

> ```python
> CPU times: user 3min 24s, sys: 8.6 s, total: 3min 33s
> Wall time: 2min 2s
> ```

​		将主成分当做样本，相当于每一张人脸都是各个主成分的线性组合。

```python
plot_faces(pca.components_[:36,:])  # 越往后越清晰
```

![1565097457245](E:\MardkDown-Books\机器学习\PCA与梯度上升法\PCA.assets\1565097457245.png)

​	



---
> 文中实例及参考：
> + 刘宇波老师《Python入门机器学习》
> + 《机器学习基础》

---

