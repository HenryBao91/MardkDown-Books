# <center>k 近邻算法</center>

&#8195;&#8195;k 近邻（k-Nearset Neighbor，简称 kNN）学习是一种常用的**监督学习**方法，其工作机制非常简单：给定测试样本，基于某种距离度量找出训练集中与其最靠近的 k 个训练样本，然后基于这 k 个“邻居”的信息来进行观测。通常，在 **分类任务** 中可使用“投票法”，即将这 k 个样本点中出现最多的类别标记作为预测结果；在 **回归任务** 中可使用“平均法”，将这 k 个样本的实值输出标记的平均值作为预测结果；还可以基于距离远近进行加权平均或加权投票，距离越近的样本权重越大。

<br/>

## 1、kNN 分类算法

​	K 近邻的思想：对于任意一个新的样本点，可以在 M 个已知类别标签的样本点中选取 K 个与其距离最接近的点作为它的最近邻点，然后统计这 K 个最近邻点的类别标签，采取多数投票表决的方式，即把这 K 个最近邻点中占绝大多数类别的点所对应的类别拿来当做要预测点的类别。

### 1.1、kNN 算法特点

- 思想极度简单
- 应用数学知识非常少
- 效果好
- 可以解释机器学习算法过程中的很多细节问题
- 更完整的刻画机器学习应用的流程
- 既可以解决**分类问题、也可以解决回归问题**

### 1.2、距离
#### 1.2.1 欧拉距离（常用）

&#8195;&#8195;对于两个点a、b的距离计算：

1. 二维距离：
   $$
   \sqrt{(x^a_1 - x^b_1)^2 + (x^a_2 - x^b_2)^2}
   $$

2. 三维距离：
   $$
   \sqrt{(x^a_1 - x^b_1)^2 + (x^a_2 - x^b_2)^2 + (x^a_3 - x^b_3)^2}
   $$

3. 多维距离：
   $$
   \sqrt{(x^a_1 - x^b_1)^2 + (x^a_2 - x^b_2)^2 + ... +
   (x^a_n - x^b_n)^2} \\
   =\sqrt{ \sum_{i=0}^{n}{(x^a_i - x^b_i)^2}}
   $$

#### 1.2.2 **更多的距离定义**
1. 向量空间余弦相似度 Cosine Similarity
3. 调整余弦相似度 Adjusted Cosine Similarity
4. 皮尔森相关系数 Pearson Correlation Coefficient
5. Jaccard 相关系数 Jaccard Coefficient
5.  汉明距离：两个字符串中不相同位数的数目

<br/>



## 1.3、分类决策规则

​	K 近邻算主要使用的多数表决规则，使用 0-1 损失函数来衡量，误分类的概率为：
$$
P(Y ≠ f(X)) = 1- P(Y = f(X))
$$
​	其中，$f(X)$ 就是分类决策函数。

​	对于给定的预测样本实例$x_j$，假设最后预测它的分类为$c_r$ ，即 $f(x_j)= c_r$。再假设 $x_j$ 最近邻的 K个训练样本实例 $x_i, \ i=(1,2,..,K)$ ，则误分类概率：
$$
L = \frac{1}{K}\sum\limits_{x_i \in N_k}I(y_i ≠ c_r) = 1 -  \frac{1}{K}\sum\limits_{x_i \in N_k}I(y_i = c_r)
$$
​	其中，$I$ 为指示函数，即 $I(Ture) = 1,\ I(False) = 0$ 。

​	目标是使用误分类率 L 最小，等价于：
$$
max \frac{1}{K}\sum\limits_{x_i \in N_k}I(y_i = c_r)
$$
​	所以，误分类率就是训练数据的核心思想，K 近邻里面的多数表决规则等价于使训练数据的经验风险最小化。



### 1.4、kNN 分类应用实例

- **判断是否为恶性肿瘤**

<center><img src="https://img-blog.csdnimg.cn/20190628154405234.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FAFFFF,t_70">

&#8195;&#8195;图中横、纵坐标分别为患肿瘤的大小和时间，红色样本点为恶性肿瘤，蓝色样本点为良性肿瘤。新输入绿色测试样例，选取 k=3，即根据离测试样例最近的3个样本点之间的距离判断该样本点是否为恶性肿瘤。

- **判断张三能否获取offer**

<center><img src="https://img-blog.csdnimg.cn/20190628154455401.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FAFFFF,t_70">

&#8195;&#8195;​张三要参加⼀一家公司的⾯面试，但它通过各种渠道得知了了拿到offer 的⼈人的情况和没有拿到offer ⼈人的情况。我们⼀一起来预测⼀一下张三是否会拿到offer?

<br/>

## 1.5、kNN 分类算法过程

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190628154516179.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70)

  &#8195;&#8195; 机器学习训练模型的过程通常叫做 **fit** ，也叫拟合。
&#8195;&#8195; 对于kNN算法训练并没有得到什么模型，可以说kNN是机器学习算法中唯一一个不需要训练过程的算法。也就是说输入样例可以直接输入送个训练数据集，直接找到距离最近的一个点。
&#8195;&#8195; k近邻算法是非常特殊，可以被认为是没有模型的算法。为了和其他算法统一，可以认为训练数据集就是模型本身。
&#8195;&#8195;  所以，对于kNN来说，训练集就是模型。

<br/>



## 2、kNN 算法分类

&#8195;&#8195;**k 近邻分类算法实现过程：**
- 输入：训练集$T={ (x_1,y_1) ,(x_2,y_2) ,..., (x_n,y_n)  }$，其中，$x_i=( x^1_i, x^2_i ,..., x^n_i)$为第 i 个训练样本实例，$y_i\in \{ c_i, c_2 ,..., c_r \}$ 为 $x_i$对应的类别标签选定的 K 值。
- 输出：待预测实例 $x_j$ 所属的类别 $x_j$。

&#8195;&#8195;**步骤如下：**

&#8195;&#8195; 第1步：根据选定的 K 值，选择合适的距离度量方式，在训练集 T 中找出待预测实例 $x_j$ 的 K 个最近邻点   $x_j$ ，这 K 个训练样本实例构成的集合记为 $N_k$ 。

&#8195;&#8195; 第2步：根据多数表决规则决定待预测实例  $x_j$ 所属的类别  $y_j$ ，即
$$
 y_j = arg \, max \frac{1}{K} \sum_{x_i \in N_k} I(y_i = c_r)
$$

### 2.1 kNN 二分类的简单实现
​	&#8195;&#8195;首先通过自己随机生成的一些二维坐标点数据，实现 kNN 算法的简单二分类问题。

​	1.  训练样本数据和测试数据：
```python
import numpy as np
import matplotlib.pyplot as plt

# 训练数据
raw_data_X = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343808831, 3.368360954],
              [3.582294042, 4.679179110],
              [2.280362439, 2.866990263],
              [7.423436942, 4.696522875],
              [5.745051997, 3.533989803],
              [9.172168622, 2.511101045],
              [7.792783481, 3.424088941],
              [7.939820817, 0.791637231]
             ]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

# 转换成 np 数组
X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)

# 测试数据
x = np.array([8.093607318, 3.365731514])

plt.scatter(X_train[y_train==0,0], X_train[y_train==0,1], color='g')
plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1], color='r')
plt.scatter(x[0], x[1], color='b')
plt.show()
```

<center><img src="https://img-blog.csdnimg.cn/20190628125002225.png">


2. kNN 训练过程：

```python
from math import sqrt

distances = []

distances = [sqrt(np.sum((x_train - x)**2))
             for x_train in X_train]

# 将 distances 排序，将索引list返回
nearest = np.argsort(distances) 

# neighbor = 6
k = 6
topk_y = [y_train[neighbor] for neighbor in nearest[:k]]

from collections import Counter
votes = Counter(topk_y)
```

3. 预测：

```python
predict_y = votes.most_common(1)[0][0]

# 预测结果
predict_y
```
<img src="https://img-blog.csdnimg.cn/20190628160926831.png " width="800">

<br/>

### 2.2 kNN 三分类的简单实现

&#8195;&#8195; 采用sklearn数据库，实现三分类问题。

 1. 导入训练数据：

    ```python
    from sklearn import datasets
    from collections import Counter  # 为了做投票
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    # 导入iris数据
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2003)
    ```

 2.  训练过程：

```python
def euc_dis(instance1, instance2):
    """
        计算两个样本instance1和instance2之间的欧式距离
        instance1: 第一个样本， array型
        instance2: 第二个样本， array型
        """
    # TODO
    dist = np.sqrt(sum((instance1 - instance2)**2))
    return dist


def knn_classify(X, y, testInstance, k):
    """
        给定一个测试数据testInstance, 通过kNN算法来预测它的标签。 
        X: 训练数据的特征
        y: 训练数据的标签
        testInstance: 测试数据，这里假定一个测试数据 array型
        k: 选择多少个neighbors? 
        """
    # TODO  返回testInstance的预测标签 = {0,1,2}
    distances = [euc_dis(x, testInstance) for x in X]
    kneighbors = np.argsort(distances)[:k]
    count = Counter(y[kneighbors])
    return count.most_common()[0][0]
```

3. 预测：

    ```python
    # 预测结果。    
    predictions = [knn_classify(X_train, y_train, data, 3) for data in X_test]
    correct = np.count_nonzero((predictions==y_test)==True)
    #accuracy_score(y_test, clf.predict(X_test))
    print ("Accuracy is: %.3f" %(correct/len(X_test)))
    ```

<br/>

### 2.3 kNN 多分类的实现

&#8195;&#8195; 采用 sklearn 中手数字识别数据集和 sklearn 中 kNN 分类器实现多分类，并进行准确度检测。

​	1. 数据导入与处理：

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt

digits = datasets.load_digits()
# sklearn 将均数据封装成 字典的形式
digits.keys()

X = digits.data
y = digits.target

# 随意取一行查看下数据，比如取 666 行
some_digit = X[666] 

# 转换成 8x8 的矩阵
some_digit_image = some_digit.reshape(8, 8)

plt.imshow(some_digit_image, cmap = matplotlib.cm.binary)
```

<center><img src="https://img-blog.csdnimg.cn/20190628161028258.png" width="300">



 2. 划分数据集预测并计算准确度：

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import kNeighborsClassifier
    from sklearn.metrics import accuracy_score
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
    
    knn_clf = kNeighborsClassifier(n_neighbors=3)
    knn_clf.fit(X_train, y_train)
    y_predict = knn_clf.predict(X_test)
    
    accuracy_score(y_test, y_predict)
    ```
    

<img src="https://img-blog.csdnimg.cn/2019062816104158.png" width="900">

<br/>

## 3、kNN 算法决策边界与超参数

### 3.1 kNN的决策边界

&#8195;&#8195; 决策边界问题：
<center><img src="https://img-blog.csdnimg.cn/20190628161209589.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70">

&#8195;&#8195; 决策边界问题代码实现：

```python
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from sklearn.neighbors import kNeighborsClassifier

# 生成一些随机样本
n_points = 100
X1 = np.random.multivariate_normal([1,50], [[1,0],[0,10]], n_points)
X2 = np.random.multivariate_normal([2,50], [[1,0],[0,10]], n_points)
X = np.concatenate([X1,X2])
y = np.array([0]*n_points + [1]*n_points)
print (X.shape, y.shape)

# kNN模型的训练过程
clfs = []
neighbors = [1,3,5,9,11,13,15,17,19]
for i in range(len(neighbors)):
    clfs.append(kNeighborsClassifier(n_neighbors=neighbors[i]).fit(X,y))
    
# 可视化结果
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(3,3, sharex='col', sharey='row', figsize=(15, 12))
for idx, clf, tt in zip(product([0, 1, 2], [0, 1, 2]),
                        clfs,
                        ['kNN (k=%d)'%k for k in neighbors]):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)
    
plt.show()
```
&#8195;&#8195;运行结果：
		![在这里插入图片描述](https://img-blog.csdnimg.cn/20190628161227201.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFADFF,t_70)

<br/>

### 3.2 超参数

#### 3.2.1 超参数与获取方法

&#8195;&#8195; 决策边界决定“线性分类器器”或者“⾮非线性分类器器”

&#8195;&#8195; 由上决策边界的实现，可知 kNN 的决策边界：k值的影响。

​	**1. 超参数和模型参数**

- 超参数：在算法运行前需要决定的参数
- 模型参数：算法过程中学习的参数

  kNN 算法中没有模型参数
  kNN 算法中的 k 是典型的超参数

​	**2. 寻找好的超参数**

- 领域知识
- 经验数值
- 实验搜索

#### 3.2.3 超参数实例
1. 寻找最好的 k 的实例：

```python
import numpy as np
from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y = digits.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

best_score = 0.0
best_k = -1
for k in range(1, 11):
    knn_clf = kNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(X_train, y_train)
    score = knn_clf.score(X_test, y_test)   # 求当前k的分类准确度
    if score > best_score:
        best_k = k
        best_score = score
        
print("best_k =", best_k)
print("best_score =", best_score)   
```

输出结果：
<img src="https://img-blog.csdnimg.cn/20190628161249609.png" width ="700">

2. 寻找 kNN 中另一个超参数 p 
&#8195;&#8195; 在 kNN 关于距离的计算中通常使用的是欧拉距离，除了欧拉距离之外还有其他类型的距离，如曼哈顿距离。

&#8195;&#8195; 曼哈顿距离:

<center><img src="https://img-blog.csdnimg.cn/20190629132405544.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70" >

二维：
$$
\sum_{i=1}^{n}|X_i^a-X_i^b|
$$
其中图中，红、蓝、黄，三条线都表示曼哈顿距离，而绿色的线表示欧拉距离。

所以，距离表示：
$$
曼哈顿距离：
\sum_{i=1}^{n}|X_i^a-X_i^b|  =  [\sum_{i=1}^{n}|X_i^a-X_i^b|]^\frac{1}{1}  \\
\\
欧拉距离：
\sqrt{\sum_{i=1}^{n}(X_i^a-X_i^b)^2}  =  [{\sum_{i=1}^{n}|X_i^a-X_i^b|^2 }]^\frac{1}{2} 
\\
所以两者具有相同的表达形式：
 [{\sum_{i=1}^{n}|X_i^a-X_i^b|^p }]^\frac{1}{p} 
$$
$$
闵可夫斯基距离：
 [{\sum_{i=1}^{n}|X_i^a-X_i^b|^p }]^\frac{1}{p} 
$$

&#8195;&#8195; 由上可知，欧拉距离和曼哈顿距离均是闵可夫斯基距离的一种特殊情况，同时也获得了一个超参数 p （即，模型参数的参数）

```python
%%time

best_score = 0.0
best_k = -1
best_p = -1

# 网格搜索
for k in range(1, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(n_neighbors=k, weights="distance", p=p)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test, y_test)
        if score > best_score:
            best_k = k
            best_p = p
            best_score = score
        
print("best_k =", best_k)
print("best_p =", best_p)
print("best_score =", best_score)
```

输出结果：
<img src="https://img-blog.csdnimg.cn/20190629140109387.png" width ="500">

**这种 k-p 循环搜索的策略，搜索规模为 k * p 的矩阵组合，这种搜索策略方法叫：网格搜索 。** </br>


</br>
                              
                           
## 4、kNN 算法解决回归问题
<center><img src="https://img-blog.csdnimg.cn/20190629201201378.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70">

**解决回归问题方法：**
&#8195;&#8195;最简单的方法，要预测绿色点的预测值，就是找到离他最近的 K 个节点，如上图，K=3，即求这三个蓝色点的值的平均值。也可以使用距离的权值，进行加权均值。


&#8195;&#8195;**k 近邻回归算法实现过程：**
- 输入：训练集$T={ (x_1,y_1) ,(x_2,y_2) ,..., (x_n,y_n)  }$，其中，$x_i=( x^1_i, x^2_i ,..., x^n_i)$为第 i 个训练样本实例，$y_i\in \{ c_i, c_2 ,..., c_r \}$ 为 $x_i$对应的类别标签选定的 K 值。
- 输出：待预测实例 $x_j$ 的值 $x_j$。

&#8195;&#8195;**步骤如下：**

&#8195;&#8195; 第1步：根据选定的 K 值，选择合适的距离度量方式，在训练集 T 中找出待预测实例 $x_j$ 的 K 个最近邻点   $x_j$ ，这 K 个训练样本实例构成的集合记为 $N_k$ 。

&#8195;&#8195; 第2步：根据取平均规则决定待预测实例  $x_j$ 所属的的输出值  $y_j$ ，即
$$
 y_j = \frac{1}{K} \sum_{ i = 1}^{K} y_i 
$$

</br>

### 4.1 数据归一化
&#8195;&#8195; 在数据进行处理进行解决归一化问题之前，通常需要对数据进行预处理工作，其中一项就是**数据归一化**。
 		

**样本间的距离被发现时间所主导**
|       | 肿瘤大小<br />（厘米） | 发现时间<br />（天） |
| :---: | :--------------------: | :------------------: |
| 样本1 |           1            |         200          |
| 样本2 |           5            |         100          |

**解决方案**：将所有的数据映射到同一尺度  

<br/>

#### 4.1.1 最值归一化 normalization

&#8195;&#8195; 最值归一化：把所有数据映射到 0-1 之间

$$
x_{scale}=\frac{x-x_{min}}{x_{max}-x_{min}}
$$
&#8195;&#8195; 适用于分布有明显便捷的情况，受outlier影响比较大
最值归一化代码：

```python
import numpy as np
import matplotlib.pyplot as plt

 # 矩阵大小为 50 x 2
X = np.random.randint(0 ,100 ,(50, 2))
X = np.array(X , dtype=float)

# 第一列
X[:, 0] = (X[:,0] - np.min(X[:,0])) / (np.max(X[:,0]) - np.min(X[:,0]))
# 第二列
X[:, 1] = (X[:,1] - np.min(X[:,1])) / (np.max(X[:,1]) - np.min(X[:,1]))

plt.scatter(X[:,0], X[:,1])
plt.show()
```
归一化可视化结果：
<center><img src="https://img-blog.csdnimg.cn/20190629144352535.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70">

#### 4.1.2 均值方差归一化 standardization

&#8195;&#8195; 数据分布没有明显的边界，有可能存在极端数据值
&#8195;&#8195; 均值归一化：把所有数据归一化到均值为0，方差为1的分布中
$$
x_{scale}=\frac{x-x_{min}}{s}
$$

均值方差归一化代码：

```python
import numpy as np
import matplotlib.pyplot as plt

 # 矩阵大小为 50 x 2
X2 = np.random.randint(0 ,100 ,(50, 2))
X2 = np.array(X , dtype=float)

# 第一列
X2[:,0] = (X2[:,0] - np.mean(X2[:,0])) / np.std(X2[:,0]) 
# 第二列
X2[:,1] = (X2[:,1] - np.mean(X2[:,1])) / np.std(X2[:,1])  

plt.scatter(X2[:,0], X2[:,1])
plt.show()
```
归一化可视化结果：
<center><img src="https://img-blog.csdnimg.cn/20190629144606737.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70">


### 4.2 测试数据归一化
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190628161339738.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70)
&#8195;&#8195; 测试数据是模拟真实环境
- 真实环境很有可能无法得到所有测试数据的均值和方差
- 对数据的归一化也是算法的一部分

$$
x_{scale}=\frac{X_{test}-\overline{X_{train}}}{std_{train}}
$$

### 4.3 kNN 回归

1. 使用数据内容：

<center><img src="https://img-blog.csdnimg.cn/20190628161409999.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FBBFFF,t_70">

2. 数据预处理：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190628161421676.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70)

3. 数据转换：
<center><img src="https://img-blog.csdnimg.cn/2019062816143383.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70">

</br>

4. kNN 回归实现(核心代码)：

```python
   from sklearn.neighbors import kNeighborsRegressor
   from sklearn.model_selection import train_test_split
   from sklearn import preprocessing
   from sklearn.preprocessing import StandardScaler
   import numpy as np
   
   X = df[['Construction Year', 'Days Until MOT', 'Odometer']]
   y = df['Ask Price'].values.reshape(-1, 1)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)
   
   X_normalizer = StandardScaler() # N(0,1)
   X_train = X_normalizer.fit_transform(X_train)
   X_test = X_normalizer.transform(X_test)
   
   y_normalizer = StandardScaler()
   y_train = y_normalizer.fit_transform(y_train)
   y_test = y_normalizer.transform(y_test)
   
   knn = kNeighborsRegressor(n_neighbors=2)
   knn.fit(X_train, y_train.ravel())
   
   #Now we can predict prices:
   y_pred = knn.predict(X_test)
   y_pred_inv = y_normalizer.inverse_transform(y_pred)
   y_test_inv = y_normalizer.inverse_transform(y_test)
```

<center><img src="https://img-blog.csdnimg.cn/20190628161502607.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70">

</br>


## 5、更多有关 kNN 问题
### 5.1、kNN 缺点

1. 最大的缺点：效率低下

   如果训练集有 m 个样本，n 个特征，则预测每一个新的数据，需要 O(m*n)

   **优化：使用树结构：kD-Tree，Ball-Tree**

2. 缺点2：高度数据相关

3. 缺点3：预测结果不具有可解释性

4. 缺点4：维数灾难

&#8195;&#8195;随着维度的增加，“看似相近”的两个点之间的距离越来越大。**解决方法：降维**

|  1维   |           0到1的距离           |   1   |
| :----: | :----------------------------: | :---: |
|  2维   |       (0,0)到(1,1)的距离       | 1.414 |
|  3维   |     (0,0,0)到(1,1,1)的距离     | 1.73  |
|  64维  | (0,0,...,0)到(1,1,...,1)的距离 |   8   |
| 1000维 | (0,0,...,0)到(1,1,...,1)的距离 |  100  |

</br>

### 5.2、kNN 处理大数据量

- 近似算法：不再寻求完全准确的解，可以适当损失精确率

- 利用类似哈希算法– Locality Sensitivity Hashing (LSH)

  **核心思想**：把样本分在不同的bucket, 使得距离比较近的样本较大概率在同一个bucket里。
  

</br>

### 5.3、kNN 处理样本的重要性
<center><img src="https://img-blog.csdnimg.cn/20190628213736635.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FBBFFF,t_70">

&#8195;&#8195; 如上图，当选取 k = 3 时，此时得到的距离测试点（绿色）最近的三个点中有两个蓝色，一个红色，如果只采用按个数投票的方式，则明显是该测试点应属于蓝色一类，但从图中可以观察（或者实际情况中），测试点明显离红色非常近，此时不能单单依靠个数进行判决，还应加入权重。
&#8195;&#8195; 最简单的方式，比如距离的倒数作为权重等方式。

**权重计算：**
$$
\omega(X, X_i)=exp(-\lambda|X - X_i|^2_2)
$$
</br>
**预测：**
$$
P_r(y|X)=\frac{\sum_{i=1}^{n}{  \omega(X, X_i) \delta(y, y_i)  } }{  \sum_{i=1}^{n}{  \omega(X, X_i)  }  }
$$
$$
\delta(y, y_i) =
\left \{ 
\begin{array}{c}
1  \qquad   y = y_i \\ 
0  \qquad   y \ne y_i 
\end{array}
\right.
$$

</br>

### 5.4、kNN 注意点
&#8195;&#8195; k 近邻有 3 个参数对结果影响比较大：**一个是数据归一化；另一个是 k 值的选择，k 一般选择奇数；还有一个是距离的度量方式。**
&#8195;&#8195; 如果不先对数据进行归一化，那么多个特征的取值范围相差较大时，就会发生距离偏移，最终结果也会受到影响。
     

---
**机器学习整体流程：**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190629202256244.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvbmd6aGVuOTE=,size_16,color_FFFFFF,t_70)


</br>

---

> 文中实例参考：
> + 贪心学院，[https://www.greedyai.com](https://www.greedyai.com)
> + 刘宇波老师《Python入门机器学习》

