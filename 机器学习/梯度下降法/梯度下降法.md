# <center>梯度下降法</center>

## 1、梯度下降法

## 1.1、梯度下降

​		梯度下降是一种非常通用的<font color='red'>优化算法</font>，能够为大范围的问题寻找最优解。**梯度下降的中心思想就是：迭代地调整参数从而使成本函数最小化**。

特点：

- 梯度下降法不是一个机器学习算法

- 梯度下降法是一种基于搜索的最优化方法

- 作用：最小化一个损失函数

- 梯度上升法：最大化一个效用函数

  使用梯度下降方法的原因：**很多机器学习的模型是无法直接求到最优解**。

<center><img src="https://img-blog.csdnimg.cn/20190726225039630.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >


​		对于上图损失函数来说，在损失函数上任取一点（下图蓝点），可以得到该点的损失函数 $J$ 值和该点的导数。由数学知识可知，如果该点的导数不为0，则该点肯定不在函数的极值点上，导数通常可以写为$\frac{dJ}{d\theta}$。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019072622525146.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70)

​		对于该问题，导数代表 $\theta$ 变化时，$J$ 相应的变化。导数可以代表方向，对应 $J$ 增大的方向，由于导数是负值，所以 $J$ 增大的方向应该是在 $\theta$ 轴的负方向上，也就是 $\theta$ 减小的时候。对应的还需要一个  $\theta$ 移动的步长，即 $\eta$ ，梯度下降中一个重要参数是每一步的步长，这取决于超参数学习率。叫做梯度下降法的原因：对于一维的的可以直接求导表征，但是对于多维的需要求各个方向的导数，也就是函数的梯度。

## 1.2、梯度下降法关键点

### 1.2.1、学习率

​		对于 $\eta$ ：

-   $\eta$ 称为学习率（Learning rate）
-   $\eta$ 的取值影响获得最优解的速度
-   $\eta$ 取值不合适，甚至得不到最优解
-   $\eta$ 是梯度下降法的一个超参数

1. 学习率 $\eta$ 太小，则算法需要经过大量迭代才能收敛，这将耗费很长时间。

<center><img src="https://img-blog.csdnimg.cn/20190726225313767.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >

2. 学习率 $\eta$ 太高，则算法可能会发散，值越来越大，最后无法找到好的解决方案。

<center><img src="https://img-blog.csdnimg.cn/2019072622550955.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >


### 1.2.2、梯度陷阱

​		对于线性回归算法而言，线性回归模型的 MSE 成本函数恰好是一个凸函数，这意味着连接曲线上任意两个点的线段永远不会跟曲线相交，即不存在局部最小，只存在一个全局最小值。

<center><img src="https://img-blog.csdnimg.cn/20190726225553884.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >


**并不是对于所有的函数都有唯一的极值点**。

 解决方法：

- 多次运行，随机初始化点
- 梯度下降法的初始点也是一个超参数

### 1.2.3、特征值缩放

​		即使损失函数是凸函数，但如果不同特征的尺寸差别巨大，如下图，那它可能是一个非常细长的“碗”，图中左边训练集上特征1和特征2具有相同的数值规模，而右边的训练集上，特征1的值则比特征2小得多。

<center><img src="https://img-blog.csdnimg.cn/20190726225634409.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >

​		由此可见，左图的梯度下降法直接走向最小值，可以快速到达。而在右图中，先是沿着与全局最小值方向近乎垂直的方向前进，接下来是一段几乎平坦的尝尝的山谷，最终虽然也能够到达最小值，但是需要花费大量的训练时间。

​		<font color='darkblue'>**应用梯度下降法时，需要保证所有特征值的大小比例差不多，可以使用sklearn中的StandartScaler类，否则收敛的时间会很长。**</font>



## 2、梯度下降法模拟

### 2.1、梯度下降法实验

```python
import numpy as np
import matplotlib.pyplot as plt

plot_x = np.linspace(-1, 6, 141)
plot_y = (plot_x - 2.5)**2 - 1   # y 是 x 的简单二次函数

plt.plot(plot_x, plot_y) 
plt.show()
```

图像：

<center><img src="https://img-blog.csdnimg.cn/20190726225703575.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >


```python
# 函数导数：
def dJ(theta):
    return 2*(theta - 2.5)

# 损失函数：
def J(theta):
    return (theta - 2.5)**2 - 1

eta = 0.1
epsilon = 1e-8

theta = 0.0
while True:
    grandient = dJ(theta)  # 梯度
    last_theta = theta     # 当前 theta
    theta = theta - eta * grandient  # 新移动的 theta
    # 判断 theta 是否来到了导数最小值的点
    if( abs(J(theta) - J(last_theta)) < epsilon ):  
        break

print(theta)
print(J(theta))
```

> **output:**
> > ```python
> > 2.499891109642585
> > -0.99999998814289
> > ```

对 $\theta$ 的观察：

```python
theta = 0.0
theta_history = [theta]
while True:
    gradient = dJ(theta)
    last_theta = theta
    theta = theta - eta * gradient
    theta_history.append(theta)
    
    if(abs(J(theta) - J(last_theta)) < epsilon):
        break

plt.plot(plot_x, J(plot_x))
plt.plot(np.array(theta_history), J(np.array(theta_history)), color="r", marker='+')
plt.show() 
```

绘制图像：

<center><img src="https://img-blog.csdnimg.cn/20190726225744153.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >




### 2.2、梯度下降法封装

```python
theta_history = []

def gradient_descent(initial_theta, eta, epsilon=1e-8):
    theta = initial_theta
    theta_history.append(initial_theta)

    while True:
        gradient = dJ(theta)
        last_theta = theta
        theta = theta - eta * gradient
        theta_history.append(theta)
    
        if(abs(J(theta) - J(last_theta)) < epsilon):
            break
            
def plot_theta_history():
    plt.plot(plot_x, J(plot_x))
    plt.plot(np.array(theta_history), J(np.array(theta_history)), color="r", marker='+')
    plt.show()
```

1. 减小$\theta$ 观察输出效果：

```python
eta = 0.01
theta_history = []
gradient_descent(0, eta)
plot_theta_history()
```

​	绘制输出图像（$\theta$ 更加密集）：

<center><img src="https://img-blog.csdnimg.cn/20190726225828138.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >


2. 增大$\theta$ 观察输出效果：

```python
eta = 0.8
theta_history = []
gradient_descent(0, eta)
plot_theta_history()
```

​	绘制输出图像（$\theta$ 更加稀疏，幅度更大）：

<center><img src="https://img-blog.csdnimg.cn/20190726225851479.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >



3. 继续增大$\theta$ 观察输出效果：

```python
eta = 1.1   #  eta 设置的过大
theta_history = []
gradient_descent(0, eta)
plot_theta_history()
```

​	编译器报错：

<center><img src="https://img-blog.csdnimg.cn/20190726225917392.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >


修改损失函数和梯度下降法代码，避免报错：

```python
def J(theta):
    try:
        return (theta-2.5)**2 - 1.
    except:
        return float('inf')   # 进行异常检测，如果错误返回浮点数最大值
    
 #  n_iters 代表运行最大循环次数  
def gradient_descent(initial_theta, eta, n_iters = 1e4, epsilon=1e-8):
    
    theta = initial_theta
    i_iter = 0
    theta_history.append(initial_theta)

    while i_iter < n_iters:    # 避免死循环
        gradient = dJ(theta)
        last_theta = theta
        theta = theta - eta * gradient
        theta_history.append(theta)
    
        if(abs(J(theta) - J(last_theta)) < epsilon):
            break
            
        i_iter += 1
        
    return    
```

重新执行上面代码：

```PYTH
eta = 1.1
theta_history = []
gradient_descent(0, eta, n_iters=10)
plot_theta_history()
```

输出图像：

<center><img src="https://img-blog.csdnimg.cn/20190726230018190.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >



## 3、多元回归中的梯度下降法

<center><img src="https://img-blog.csdnimg.cn/20190726230044311.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >


​	对于多元线性回归来说，$\theta$ 需要用向量来表示，代表多个参数；对于最简单的一元线性回归来说，$\theta$ 也包含有两个值，即 $\theta = (\theta_0, \theta_1)$ ，相当于 $y = \theta_0 + \theta_1\cdot x $ 。此时，损失函数为梯度为：
$$
\nabla J = (\frac{\partial J}{\partial \theta_0}, \frac{\partial J}{\partial \theta_1}, ... , \frac{\partial J}{\partial \theta_n})
$$

### 3.1、梯度下降法的推导

​	**目标**：使 $\sum\limits^m_{i=1}(y^{(i)} - \hat y^{(i)} )^2$  尽可能的小。

​	其中，$\hat y^{(i)}  = \theta_0 + \theta_1 \mathbf X_1^{(i)} + \theta_2 \mathbf X_2^{(i)} + ... + + \theta_n \mathbf X_n^{(i)}$ 。

​	所以，求解目标可以转换为： $\sum\limits^m_{i=1}(y^{(i)} -  \theta_0 + \theta_1 \mathbf X_1^{(i)} + \theta_2 \mathbf X_2^{(i)} + ... + + \theta_n \mathbf X_n^{(i)} )^2$  尽可能小。
​	由于此时求解的变量是 $\theta$ , 所以求导过程中为了直观交换一下顺序，同时把 $\hat y^{(i)}$ 写成向量形式。

$$
\nabla J(\theta) = 
\begin{pmatrix}
\frac{\partial J}{\partial \theta_0}   \\
\\
\frac{\partial J}{\partial \theta_1}    \\
\\
\frac{\partial J}{\partial \theta_2}    \\
\vdots   \\
\frac{\partial J}{\partial \theta_n}   \\
\end{pmatrix}
= \begin{pmatrix}
\sum\limits^m_{i=1}2(y^{(i)} - \mathbf X_b^{(i)}\theta  )\cdot(-1)  \\
\sum\limits^m_{i=1}2(y^{(i)} - \mathbf X_b^{(i)}\theta  )\cdot(-X_1^{(i)})  \\
\sum\limits^m_{i=1}2(y^{(i)} - \mathbf X_b^{(i)}\theta  )\cdot(-X_2^{(i)})  \\
\vdots   \\
\sum\limits^m_{i=1}2(y^{(i)} - \mathbf X_b^{(i)}\theta  )\cdot(-X_n^{(i)})  \\
\end{pmatrix}
= 2
\begin{pmatrix}
\sum\limits^m_{i=1}(\mathbf X_b^{(i)}\theta  -y^{(i)})  \\
\sum\limits^m_{i=1}(\mathbf X_b^{(i)}\theta -y^{(i)})\cdot X_1^{(i)}  \\
\sum\limits^m_{i=1}(\mathbf X_b^{(i)}\theta -y^{(i)})\cdot X_2^{(i)}  \\
\vdots   \\
\sum\limits^m_{i=1}(\mathbf X_b^{(i)}\theta -y^{(i)})\cdot X_n^{(i)}  \\
\end{pmatrix}
$$

​	由于在上述求损失函数梯度公式中，每一项均是求和，所以，损失函数梯度的大小与 m 有关，这显然梯度大小与样本数量有关，这实际是不合理的，实际应该是和梯度无关的，所以将目标函数除以m，得：

$$
J(\theta) =  \frac{2}{m}\sum\limits^m_{i=1}(y^{(i)} - \hat y^{(i)} )^2
$$
​	即，<font color="red">$J(\theta) = MSE(y, \hat y)$</font> ，有时也取 $J(\theta) =  \frac{1}{2m}\sum\limits^m_{i=1}(y^{(i)} - \hat y^{(i)} )^2$ ，目的均是为了方便计算。

​	最终梯度公式变为：
$$
\nabla J(\theta) = 
\begin{pmatrix}
\frac{\partial J}{\partial \theta_0}   \\
\\
\frac{\partial J}{\partial \theta_1}    \\
\\
\frac{\partial J}{\partial \theta_2}    \\
\vdots   \\
\frac{\partial J}{\partial \theta_n}   \\
\end{pmatrix}
= \frac{2}{m}
\begin{pmatrix}
\sum\limits^m_{i=1}(\mathbf X_b^{(i)}\theta  -y^{(i)})  \\
\sum\limits^m_{i=1}(\mathbf X_b^{(i)}\theta -y^{(i)})\cdot X_1^{(i)}  \\
\sum\limits^m_{i=1}(\mathbf X_b^{(i)}\theta -y^{(i)})\cdot X_2^{(i)}  \\
\vdots   \\
\sum\limits^m_{i=1}(\mathbf X_b^{(i)}\theta -y^{(i)})\cdot X_n^{(i)}  \\
\end{pmatrix}
$$



### 3.2、线性回归梯度下降法

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)
x = 2 * np.random.random(size=100)
y = x * 3. + 4. + np.random.normal(size=100)

X = x.reshape(-1, 1)

plt.scatter(x, y)
plt.show()
```

​	为了可视化的方便，这里 x 是使用的一维数组表示，同时为了方便拓展到多维，将 x 进行reshape 转换成 X 。

> ```python
> X.shape
> 
> (100 , 1)
>```
>```python
> y.shape
> 
>  (100 ,)
>  ```

​	数据图像：

<center><img src="https://img-blog.csdnimg.cn/20190726230421981.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" width=70% >



​	**使用梯度下降法训练：**

```python
# 求解损失函数
def J(theta, X_b , y):
    try:
        return np.sum((y - X_b.dot(theta))**2) / len(X_b)
    except:
        return float('inf')
    
# 求解损失函数梯度
def dJ(theta, X_b, y):
    res = np.empty(len(theta))   # 定义一个长度为len(theta)的空数组
    res[0] = np.sum(X_b.dot(theta) - y)  # 第一个元素单独求
    for i in range(1, len(theta)):       # 第1~n个元素单独求
        res[i] = (X_b.dot(theta) - y).dot(X_b[:,i])
    return res * 2 / len(X_b)

# 梯度下降求参数
def gradient_descent(X_b, y, initial_theta, eta, n_iters = 1e4, epsilon=1e-8):
    
    theta = initial_theta
    cur_iter = 0

    while cur_iter < n_iters:
        gradient = dJ(theta, X_b, y)
        last_theta = theta
        theta = theta - eta * gradient
        if(abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
            break
            
        cur_iter += 1

    return theta
```

​	求解实现程序：

```python
X_b = np.hstack([ np.ones((len(x), 1)), x.reshape(-1,1) ])
initial_theta = np.zeros(X_b.shape[1])  # theta 初始化为 0
eta = 0.01

theta = gradient_descent(X_b, y, initial_theta, eta)
```

**输出 ：theta**

> > ```python
> > array([ 4.02145786,  3.00706277])
> > ```

​	

### 3.3、封装梯度下降法

​	封装梯度下降法代码：

```python
import numpy as np
from .metrics import r2_score

class LinearRegression:

    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None
        self.intercept_ = None
        self._theta = None


    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            res = np.empty(len(theta))
            res[0] = np.sum(X_b.dot(theta) - y)
            for i in range(1, len(theta)):
                res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            return res * 2 / len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):

            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self



    def __repr__(self):
        return "LinearRegression()"

```

​	测试封装后的代码实现：

```python
from myML.LinearRegression import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit_gd(X, y)
```

​	**输出：**

> > ```python
> > lin_reg.coef_  = array([ 3.00706277])
> > ```
> > ```python
> > lin_reg.intercept_  =  4.021457858204859
> > ```



## 4、梯度下降法向量化与数据标准化

### 4.1、向量化公式推导

​	由上可知，梯度下降法计算公式可以进行变形，推导如下：

$$
\nabla J(\theta)  
= \frac{2}{m}
\begin{pmatrix}
\sum\limits^m_{i=1}(\mathbf X_b^{(i)}\theta  -y^{(i)})  \\
\sum\limits^m_{i=1}(\mathbf X_b^{(i)}\theta -y^{(i)})\cdot X_1^{(i)}  \\
\sum\limits^m_{i=1}(\mathbf X_b^{(i)}\theta -y^{(i)})\cdot X_2^{(i)}  \\
\vdots   \\
\sum\limits^m_{i=1}(\mathbf X_b^{(i)}\theta -y^{(i)})\cdot X_n^{(i)}  \\
\end{pmatrix}
=\frac{2}{m}
\begin{pmatrix}
\sum\limits^m_{i=1}(\mathbf X_b^{(i)}\theta -y^{(i)})\cdot X_0^{(i)}   \\
\sum\limits^m_{i=1}(\mathbf X_b^{(i)}\theta -y^{(i)})\cdot X_1^{(i)}  \\
\sum\limits^m_{i=1}(\mathbf X_b^{(i)}\theta -y^{(i)})\cdot X_2^{(i)}  \\
\vdots   \\
\sum\limits^m_{i=1}(\mathbf X_b^{(i)}\theta -y^{(i)})\cdot X_n^{(i)}  \\
\end{pmatrix}
=\frac{2}{m} \cdot \mathbf X_b^T \cdot (\mathbf X_b \theta - y)
$$
​	
其中，$X_0^{(i)} \equiv 1$ ，对上式进一步化简得：


$$
\nabla J(\theta) 
=( \mathbf X_b^{(1)}\theta -y^{(1)} , \mathbf X_b^{(2)}\theta -y^{(2)}, ..., \mathbf X_b^{(m)}\theta -y^{(m)}) \cdot
\begin{pmatrix}
\mathbf X_0^{(1)} & \mathbf X_1^{(1)}  & \cdots  & \mathbf X_n^{(1)} \\
\mathbf X_0^{(2)} & \mathbf X_1^{(2)}  & \cdots  & \mathbf X_n^{(2)} \\
\vdots  & \vdots &  \ddots  & \vdots    \\
\mathbf X_0^{(m)} & \mathbf X_1^{(m)}  & \cdots  & \mathbf X_n^{(m)}
\end{pmatrix}  \\
$$
​	
​	对上式化简并进行转置（转置原因：由于梯度是列向量，但是在 numpy 的计算中不区分，所以进行一下转置），最终可得：
​	
$$
\nabla J(\theta) 
= \frac{2}{m} \cdot {(\mathbf X_b \theta - y)}^T \cdot \mathbf X_b \\
\\
= \frac{2}{m} \cdot \mathbf X_b^T \cdot (\mathbf X_b \theta - y) \\
$$


### 4.2、代码实现

```python
# 向量化求梯度
def dJ(theta, X_b, y):
    return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(y)  # 这里 len(y) = len(X_b)
```

​	使用波士顿房价数据，并使用向量化的梯度下降法进行预测：

```python
import numpy as np
from sklearn import datasets

boston = datasets.load_boston()
X = boston.data
y = boston.target

X = X[y < 50.0]
y = y[y < 50.0]
```

1. **采用标准方程的方式训练线性回归：**

```python
from myML.LinearRegression import LinearRegression

lin_reg1 = LinearRegression()
%time lin_reg1.fit_normal(X_train, y_train)
lin_reg1.score(X_test, y_test)
```

 输出情况：

> ```python
> CPU times: user 45.6 ms, sys: 5.74 ms, total: 51.3 ms
> Wall time: 56.4 ms
>     
> 0.81298026026584658    
> ```



2. **采用梯度下降法训练线性回归：**

```python
lin_reg2 = LinearRegression()
lin_reg2.fit_gd(X_train, y_train)
```

​	此时运行程序会报错：

<center><img src="https://img-blog.csdnimg.cn/20190726230955347.png" >


通过打印查看数据可以看到，训练的模型的参数都是 nan ：


```python
lin_reg2.coef_
```

> ```python
> array([ nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,
>         nan,  nan])
> ```

​	通过打印训练集数据查看下前3行数据：

<center><img src="https://img-blog.csdnimg.cn/20190726231052824.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >

​	由数据可知，训练集中数据样本之间差别太大，有的维度的数据是几百，有的是0.x，所以最终求导的梯度可能也是非常大的，使用默认的 $\eta$ 的步长还是较大，使得最终的训练结果可能还是不收敛的。通过改变 $\eta$ 的值可以验证。

 ```python
# 减小步长 eta 的值，并且增大循环迭代的次数，以提高准确度
%time lin_reg2.fit_gd(X_train, y_train, eta=0.000001, n_iters=1e6) 

lin_reg2.score(X_test, y_test)
 ```

​	输出结果：

> ```python
> CPU times: user 48.4 s, sys: 265 ms, total: 48.6 s
> Wall time: 49.9 s
>     
> 0.75418523539807636
> ```



### 4.3、数据归一化

​	在使用标准方程求解线性回归模型的时候并没有进行数据归一化，这是因为线性回归模型的求解转换成了数学公式的计算，涉及的搜索过程很少，所以不需要。但是在使用梯度下降法时，如果数据数值不在一个维度上将会影响梯度的结果，而梯度的结果再乘以步长 $\eta$ 才是真正每次走的步长，就可能出现每一步太大或太小，就会出现上述结果不收敛或者收敛速度太慢的情况。**所以，<font color='red'>使用梯度下降法之前，最好对数据进行归一化.</font>**

<center><img src="https://img-blog.csdnimg.cn/20190726231120273.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >


### 4.3.1、数据归一化梯度下降法

```python
from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()
standardScaler.fit(X_train)           # 首先使用 StandardScaler 对 X_train 进行 fit
X_train_standard = standardScaler.transform(X_train)  # 通过 transform 将数据归一化

lin_reg3 = LinearRegression()
%time lin_reg3.fit_gd(X_train_standard, y_train)

# 计算训练结果的 R2 值
# 由于对训练数据集进行了归一化，所以对测试集也需要归一化
X_test_standard = standardScaler.transform(X_test)
lin_reg3.score(X_test_standard, y_test)
```

​	输出结果：

> ```python
> CPU times: user 237 ms, sys: 4.37 ms, total: 242 ms
> Wall time: 258 ms
>     
> # 训练结果的 R2 值   
> 0.81298806201222351
> ```



### 4.3.2、梯度下降法的优势

​	从上述实例可以看出，使用标准方程解和使用梯度下降法的训练过程时间分别是 56.4 ms 和 258 ms 。使用规模比较大的虚拟数据进行测试

```python
m = 1000
n = 5000

big_X = np.random.normal(size=(m, n))
true_theta = np.random.uniform(0.0, 100.0, size=n+1)
big_y = big_X.dot(true_theta[1:]) + true_theta[0] + np.random.normal(0., 10., size=m)
```

1. 标准方程解测试：

```python
big_reg1 = LinearRegression()
%time big_reg1.fit_normal(big_X, big_y)
```

> ```python
> Wall time: 16.21 s
> ```

2. 梯度下降法测试：

```python
big_reg2 = LinearRegression()
%time big_reg2.fit_gd(big_X, big_y)
```

> ```python
> Wall time: 5.31 s
> ```

​	通过对比，当特征数和数据量较大的时候，梯度下降法耗时远小于标准方程法求解时间。



## 5、批量梯度下降（BGD)

​	在上述梯度下降的求解方法，都是基于完整的训练集 X 的。所以这种梯度下降法也叫**批量梯度下降法**。
$$
\nabla J(\theta) 
 = \nabla MSE(\theta) 
 = \frac{2}{m} \cdot \mathbf X_b^T \cdot (\mathbf X_b \theta - y) 
$$
​	**批量梯度下降法**：每一步都使用整批训练数据。所以，面对非常庞大的训练集时，算法会变得非常慢。但是，梯度下降法随特征数量的增加，如对于训练同样的线性回归模型，使用批量梯度下降法要比标准（正规）方程法快的多。

<center><img src="https://img-blog.csdnimg.cn/20190726231210495.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >


​	但是，批量梯度下降法的受学习率 $\eta$ 的制约，如上左、右图，学习率太低或者太高都不好。学习率太低最终可以得到最终解，但是需要太久的训练时间；学习率太高，算法容易发散，可能离最终结果越来越远。

​	要寻找合适的学习率，可以使用网格搜索的方法。但是，需要限制迭代次数，这样网格搜索可以淘汰哪些收敛耗时太长的模型。一个简单的办法，就是在开始时设置一个非常大的迭代次数，当梯度向量的值变得很微小时中断算法，即认为这时梯度下降基本上达到了最小值。


</br>


## 6、随机梯度下降（SGD)

### 6.1、随机梯度下降推导

批量梯度下降的主要问题是它要用整个训练集来计算每一步的梯度，所以训练集很大时，算法会特别慢。而随机梯度下降，每一步在训练集中随机选择一个样本点数据，并且仅基于该样本点数据来计算梯度。这样每次迭代都只需要计算很少量的数据。

$$
\nabla J(\theta) 
= \frac{2}{m}
\begin{pmatrix}
\sum\limits^m_{i=1}(\mathbf X_b^{(i)}\theta -y^{(i)})\cdot X_0^{(i)}   \\
\sum\limits^m_{i=1}(\mathbf X_b^{(i)}\theta -y^{(i)})\cdot X_1^{(i)}  \\
\sum\limits^m_{i=1}(\mathbf X_b^{(i)}\theta -y^{(i)})\cdot X_2^{(i)}  \\
\cdots   \\
\sum\limits^m_{i=1}(\mathbf X_b^{(i)}\theta -y^{(i)})\cdot X_n^{(i)}  \\
\end{pmatrix}
=2 \cdot
\begin{pmatrix}
(\mathbf X_b^{(i)}\theta -y^{(i)})\cdot X_0^{(i)}   \\
(\mathbf X_b^{(i)}\theta -y^{(i)})\cdot X_1^{(i)}  \\
(\mathbf X_b^{(i)}\theta -y^{(i)})\cdot X_2^{(i)}  \\
\cdots   \\
(\mathbf X_b^{(i)}\theta -y^{(i)})\cdot X_n^{(i)}  \\
\end{pmatrix}
=2 \cdot \mathbf (X_b^{(i)})^T \cdot (\mathbf X_b^{(i)} \theta - y^{(i)})
$$


​	即对 BGD 公式计算每次只取一行样本数据进行计算，**当作搜索的方向**<font color='darkred'>**（并不是实际梯度的方向，因为此时方程已经不是损失函数的梯度）。**</font>

由于算法的随机性质，它比批量梯度下降要不规则得多。损失函数将不再是缓缓降低直到抵达最小值，而是不断上上下下，但是从整体来看，还是在慢慢下降。所以，**在随机梯度下降法中，学习率的取值变得非常重要**。因为，如果学习率一直取一个固定值，可能已经来到最优的点位置了，但是由于学习率还是固定值，所以步长也是固定的，可能错过了最优点。那么，最简单的方法就是让学习率随着迭代次数逐渐递减。

​	搜索过程如下如所示：

<center><img src="https://img-blog.csdnimg.cn/20190726231355927.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >


​	所以，对学习率进行改变，得到上图中学习率公式，其中 $a、b$ 相当于随机梯度下降法中的超参数。经验上比较好的选取为： $a=5 ；b=50$ 。这种方法叫做**模拟退火的方法**（类似于冶金时熔化的金属慢慢冷却的退火过程。确定每个迭代学习率的函数叫作学习计划。如果学习率降得太快，可能会陷入局部最小值，甚至是停留在走向最小值的半途中。如果学习率降得太慢，需要太长时间才能跳到差不多最小值附近），最终得：
$$
\eta = \frac{t_0}{i\_iters \ + \ t_1 }
$$



### 6.2、随机梯度下降实践

```python
import numpy as np
import matplotlib.pyplot as plt

m = 100000

x = np.random.normal(size=m)
X = x.reshape(-1,1)
y = 4.*x + 3. + np.random.normal(0, 3, size=m)

plt.scatter(x, y)
plt.show()
```

样本数据可视化：
<center><img src="https://img-blog.csdnimg.cn/20190726231433963.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >


1. 使用批量梯度下降法：
循环终止的条件有两个：
   ①、循环迭代次数 ；
   ②、当前损失函数比上一轮减小的值小于设置的阈值$\epsilon$ . 
  
```python
def J(theta, X_b, y):
    try:
        return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
    except:
        return float('inf')
    
def dJ(theta, X_b, y):
    return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(y)

def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):

    theta = initial_theta
    cur_iter = 0

    while cur_iter < n_iters:
        gradient = dJ(theta, X_b, y)
        last_theta = theta
        theta = theta - eta * gradient
        if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
            break

        cur_iter += 1

    return theta
```

```python
%%time
X_b = np.hstack([np.ones((len(X), 1)), X])
initial_theta = np.zeros(X_b.shape[1])
eta = 0.01
theta = gradient_descent(X_b, y, initial_theta, eta)

print(theta)
```

> ```python
> Wall time: 1.75 s
>     
> array([2.97614182, 4.01141538])    
> ```


2. 使用随机梯度下降法：

   循环终止的条件有一个：

   ①、循环迭代次数 .（由于梯度的改变方向是随机的，所以不能保证损失函数一直在减小，有可能是跳跃的）

```python
def dJ_sgd(theta, X_b_i, y_i ):  # X_b_i, y_i 分别是一行和一个对应的数据
    return 2 * X_b_i.T.dot(X_b_i.dot(theta) - y_i)

def sgd(X_b, y, initial_theta, n_iters):

    t0, t1 = 5, 50
    def learning_rate(t):   # 学习率函数，t 是当前迭代次数
        return t0 / (t + t1)

    theta = initial_theta
    for cur_iter in range(n_iters):
        rand_i =  np.random.randint(len(X_b))   # 随机一个样本对应的索引
        gradient = dJ_sgd(theta, X_b[rand_i], y[rand_i])  # 传入该索引对应的数据
        theta = theta - learning_rate(cur_iter) * gradient
    
    return theta
```

```python
%%time
X_b = np.hstack([np.ones((len(X), 1)), X])
initial_theta = np.zeros(X_b.shape[1]) 
theta = sgd(X_b, y, initial_theta, n_iters=len(X_b)//3)  # 循环次数是样本总量的 1/3

print(theta)
```

> ```python
> Wall time: 200 ms
>     
> array([2.97476237, 4.0101637 ])    
> ```



### 6.3、随机梯度下降综合

封装 SGD 代码：

```python
import numpy as np 

class LinearRegression:

    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    # n_iters 代表对所有样本循环几轮    
    def fit_sgd(self, X_train, y_train, n_iters=50, t0=5, t1=50):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert n_iters >= 1

        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i * (X_b_i.dot(theta) - y_i) * 2.

        def sgd(X_b, y, initial_theta, n_iters=5, t0=5, t1=50):

            def learning_rate(t):
                return t0 / (t + t1)

            theta = initial_theta
            # 用 m 来表示迭代次数，每一次迭代称为一轮
            m = len(X_b)
            
            for i_iter in range(n_iters):  # 第 i_iter 轮
                
                # 每一轮循环过程对样本进行乱序，从而保证尽可能每轮对所有样本遍历一遍
                indexes = np.random.permutation(m)
                
                X_b_new = X_b[indexes,:]
                y_new = y[indexes]
                
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    # 当前学习率计算对应的次数 ： i_iter * m + i
                    theta = theta - learning_rate(i_iter * m + i) * gradient

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.random.randn(X_b.shape[1])
        self._theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    ...
```



1. 封装我们自己的SGD

```python
import numpy as np
import matplotlib.pyplot as plt

m = 100000

x = np.random.normal(size=m)
X = x.reshape(-1,1)
y = 4.*x + 3. + np.random.normal(0, 3, size=m)

from playML.LinearRegression import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit_sgd(X, y, n_iters=2)

print(lin_reg.coef_)
print(lin_reg.intercept_)
```

> ```python
> lin_reg.coef_ = array([3.99986556])
> 
> lin_reg.intercept_ = 3.0042962116349594
> ```



2. 真实数据中使用SGD

```python
from sklearn import datasets
from playML.model_selection import train_test_split
from playML.LinearRegression import LinearRegression

boston = datasets.load_boston()
X = boston.data
y = boston.target

X = X[y < 50.0]
y = y[y < 50.0]

X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)
# 使用真实的数据进行梯度下降法，需要对样本特征数据进行归一化
from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)

lin_reg = LinearRegression()
%time lin_reg.fit_sgd(X_train_standard, y_train, n_iters=2)
lin_reg.score(X_test_standard, y_test)
```

> ```python
> Wall time: 4.96 ms
> 
> 0.7857275413602652
> ```



3. scikit-learn 中的 SGD

```python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(n_iter=100)
%time sgd_reg.fit(X_train_standard, y_train)
sgd_reg.score(X_test_standard, y_test)
```

> ```python
> Wall time: 3.99 ms
> 
> 0.812414332027749
> ```



## 7、小批量梯度下降法

​	小批量梯度下降法：每一步的梯度计算，既不是基于整个训练集（如批量梯度下降），也不是基于单个样本点（如随机梯度下降），而是基于一小部分随机的样本数据，也就是小批量。相比随机梯度下降，小批量梯度下降的主要优势在于可以从矩阵运算的优化中获得显著的性能提升，特别是需要用到图形处理器时。

​	小批量梯度下降法在参数空间层面的前也不像SGD那样不稳定，特别是批量较大时。所以小批量梯度下降最终会比 SGD 更接近最小值一些。但是另一方面，它可能会受到局部最小值的限制问题。



## 8、关于梯度下降法调试

### 8.1、调试的推导

​	对于一维而言，函数上某点的梯度（导数），也就是该点切线的斜率。通过在该点正负方向各取相邻的一点，这两点连线的斜率可以近似作为该点切线的斜率。

<center><img src="https://img-blog.csdnimg.cn/20190730223849814.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >

这样的方法同样适用于高维的场景，对于 $\theta = (\theta_0 , \theta_1, ..., \theta_n)$ ，其梯度为：$\frac{\partial J}{\partial  \theta} = (\frac{\partial J}{\partial  \theta_0},\frac{\partial J}{\partial  \theta_1}, ... , \frac{\partial  J}{\partial \theta_n})$ 。以对 $\theta_0$ 求导为例，在 $\theta_0$ 左右各取两点：
$$
\begin{cases}
\theta_0^+ = (\theta_0 + \epsilon, \theta_1, ..., \theta_n)  \\
\\
\theta_0^- = (\theta_0 - \epsilon, \theta_1, ..., \theta_n) \\
\end{cases}
$$
所以损失函数 $J$ 对 $\theta_0$ 求导的结果为：
$$
\frac{\partial J}{\partial \theta_0}  \frac{J(\theta_0^+) - J(\theta_0^-)}{2 \ \epsilon}
$$

以此类推，但是这样求解的时间复杂度非常高，所以这种方法仅仅作为小数据量的调试手段，先调试测试一下大概结果，再通过梯度下降法求解并对比结果正确性。



### 8.2、梯度的调试

1. 使用的数据：

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)
X = np.random.random(size=(1000, 10))

true_theta = np.arange(1, 12, dtype=float)  # 真实 theta: 1-11

X_b = np.hstack([np.ones((len(X), 1)), X])    # 在 X 前面加上一列 1
y = X_b.dot(true_theta) + np.random.normal(size=1000) 
```

2. 数学方法和调试方法：

```python
def J(theta, X_b, y):
    try:
        return np.sum((y - X_b.dot(theta))**2) / len(X_b)
    except:
        return float('inf')
    
    
# 数学方法
def dJ_math(theta, X_b, y):
    return X_b.T.dot(X_b.dot(theta) - y ) * 2. / len(y)

# 调试方法
def dJ_debug(theta, X_b, y, epsilon=0.01):
    res = np.empty(len(theta))
    for i in range(len(theta)):
        theta_1 = theta.copy()
        theta_1[i] += epsilon
        theta_2 = theta.copy()
        theta_2[i] -= epsilon
        res[i] = (J(theta_1, X_b, y) - J(theta_2, X_b, y)) / (2 * epsilon)
    return res
```

3. 分别测试输出：

```python
X_b = np.hstack([np.ones((len(X), 1)), X])
initial_theta = np.zeros(X_b.shape[1])  # 初始值设成全0 
eta = 0.01
```

- 数学方法求解 	

```python
%time theta = gradient_descent(dJ_debug, X_b, y, initial_theta, eta)
print(theta)
```

> ```python
> Wall time: 2.84 s
> ```
>
> ```python
> array([ 1.10059643,  1.9566545 ,  2.82202437,  4.16594877,  5.05120906,
>         5.93194605,  6.81204783,  7.94520523,  9.0933952 ,  9.99854904,
>        10.90008725])
> ```

- 调试方法求解：

```python
%time theta = gradient_descent(dJ_math, X_b, y, initial_theta, eta)
print(theta)
```

> ```python
> Wall time: 466 ms
> ```
>
> ```python
> array([ 1.10059643,  1.9566545 ,  2.82202437,  4.16594877,  5.05120906,
>         5.93194605,  6.81204783,  7.94520523,  9.0933952 ,  9.99854904,
>        10.90008725])
> ```

​	通过测试可以看出：

（1）、两种方法最终都能得到正确的结果；

（2）、但是调试的方法耗时比数学梯度下降法求解慢很多；

（3）、调试方法是与损失函数 $J$ 无关的，它适用于所用的函数 .


<br>

## 9、更多关于梯度下降法

1. **上述一共总结了三种常用梯度下降法：**

- 批量梯度下降法 Batch Gredient Descent
- 随机梯度下降法 Stochastic Gredient Descent
- 小批量梯度下降法 Mini-Batch Gredient Descent

2. **随机的优势：**

- 跳出局部最优解
- 更快的运行速度
- 机器学习领域很多算法都要使用随机的特点：随机搜索、随机森林

 

---
> 文中实例及参考：
> + 刘宇波老师《Python入门机器学习》
> + 《机器学习实战》

---

