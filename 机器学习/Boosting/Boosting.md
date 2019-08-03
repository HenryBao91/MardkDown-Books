#<center>Boosting</center>



## 1、Boosting

## 1.1、Boosting算法

​		Boosting算法核心思想：

![1564665572566](E:\MardkDown-Books\机器学习\Boosting\Boosting.assets\1564665572566.png)



### 1.2、Boosting实例

​	使用Boosting进行年龄预测：

![1564665960597](E:\MardkDown-Books\机器学习\Boosting\Boosting.assets\1564665960597.png)

![1564665982604](E:\MardkDown-Books\机器学习\Boosting\Boosting.assets\1564665982604.png)

![1564666022744](E:\MardkDown-Books\机器学习\Boosting\Boosting.assets\1564666022744.png)

![1564666031943](E:\MardkDown-Books\机器学习\Boosting\Boosting.assets\1564666031943.png)





## 2、XGBoosting

​		XGBoost 是 GBDT 的一种改进形式，具有很好的性能。

![1564666162880](E:\MardkDown-Books\机器学习\Boosting\Boosting.assets\1564666162880.png)



### 2.1、XGBoosting 推导

​		经过 k 轮迭代后，GBDT/GBRT 的损失函数可以写成 $L(y, f_k(x))$，将$f_k(x)$视为遍历（是一个复合型变量），对 $L(y, f_k(x))$ 在 $f_{k-1}(x)$ 处进行二阶泰勒展开，可得：
$$
L(y, f_k(x)) \\ ≈ L(y, f_{k-1}(x) ) + \frac{\partial L(y, f_{k-1}(x) )}{\partial f_{k}(x)}[f_k(x) \ -  f_{k-1}(x) ] \ + \ \frac{1}{2} × \frac{\partial ^2 L(y, f_{k-1}(x) )}{\partial ^2 f_{k-1}(x)}[f_k(x) \ -  f_{k-1}(x) ]^2
$$
​		然后取 $g = \frac{\partial L(y, f_{k-1}(x) )}{\partial f_{k}(x)}$ ，$h = \frac{\partial ^2L(y, f_{k-1}(x) )}{\partial ^2f_{k}(x)}$ ，即 $g$ 和 $h$ 分别代表一阶导数和二阶导数，所以，展开式可以变形得：
$$
L(y, f_k(x))  ≈ L(y, f_{k-1}(x) ) + g[f_k(x) \ -  f_{k-1}(x) ] \ + \ \frac{1}{2}h[f_k(x) \ -  f_{k-1}(x) ]^2
$$
​		又因为在 GBDT 中，利用前向分布算法，有 $f_k(x) \ =  f_{k-1}(x) + T_k(x)$，即：
$$
f_k(x) \ -  f_{k-1}(x) \ = T_k(x)
$$
​		代入上式，可得：
$$
L(y, f_k(x))  ≈ L(y, f_{k-1}(x) ) + g \cdot T_k(x) \ + \ \frac{1}{2}h \cdot T^2_k(x)
$$
​		上面的损失函数还是对于一个样本数据而言，对于整体样本数据，损失函数可得：
$$
L = \sum\limits^N_{i=1} L(y, f_k(x))  ≈ \sum\limits^N_{i=1} [ L(y, f_{k-1}(x) ) + g \cdot T_k(x) \ + \ \frac{1}{2}h \cdot T^2_k(x) ]
$$
​		等式右边中的第一项 $L(y, f_{k-1}(x) )$ 只与前 k-1 轮有关，第 k 轮优化中可将该项视为常数。在 GBDT 中的损失函数上再加上一项与第 k 轮的基学习器 CART 决策树相关的正则化项 $\Omega (T_k(x))$ 防止过拟合，可得到新的第 k 轮迭代时的等价损失函数：
$$
L_k =  \sum\limits^N_{i=1} [ \ g_i \cdot T_k(x_i) \ + \ \frac{1}{2}h_i \cdot T^2_k(x_i) \ ] \ + \Omega (T_k(x))
$$
​		**所以，$L_k$ 就是 XGBoost 模型的损失函数。**

​		假设第 k 棵 CART 回归树其对一个的叶子区域样本子集为 $D_{k1}, D_{k2}, ... ,D_{kT}$，且第 $j$ 个小单元 $D_{kj}$ 中仍然包含 $N_{kj}$ 个样本数据，则计算每个小单元里面的样本的输出均值为：
$$
\overline c_{kj} = \frac{1}{N_{kj}}\sum\limits_{x_i \in D_{kj}} y_i
$$
​		得到：
$$
T_k(x) = \sum\limits_{j=1}^T \overline c_{kj} \cdot I( x_i \in D_{kj} )
$$
​		正则化项 $\Omega (T_k(x))$ 的构造如下：
$$
\Omega (T_k(x)) = \gamma \ T + \frac{1}{2}\gamma \sum\limits_{j=1}^T \overline c_{kj}^2
$$
​		其中，  参数 $T$ 为 $T_k(x)$ 决策树的叶子节点的个数，参数 $\overline c_{kj} ，j=1,2,...,T$，是第 $j$ 个叶子节点的输出均值；$\gamma $ 和 $\lambda$ 是两个权衡因子。叶子节点的数量及权重因子一起用来控制决策树模型的复杂度。

​		将 $T_k(x)$ 和 $\Omega (T_k(x))$ 一起带入 $L_{k}$ ，可得：
$$
L_k =  \sum\limits^N_{i=1} [ \ g_i \cdot T_k(x_i) \ + \ \frac{1}{2}h_i \cdot T^2_k(x_i) \ ] \ + \Omega (T_k(x)) \\
= \sum\limits^N_{i=1} [ \ g_i \cdot T_k(x_i) \ + \ \frac{1}{2}h_i \cdot T^2_k(x_i) \ ] \ +\gamma \ T + \frac{1}{2}\gamma \sum\limits_{j=1}^T \overline c_{kj}^2 \\
=  \sum\limits^N_{i=1}[ (\sum\limits_{x_i \in D_{kj}}g_i)\cdot \overline c_{kj} + \frac{1}{2}(\sum\limits_{x_i \in D_{kj}} h_i + \lambda) (\overline c_{kj})^2] + \gamma \ T
$$

​		由上公式可知，XGBoost 模型对应的损失函数 $L_k$ 主要与原损失函数的一阶、二阶梯度在当前模型的值 $g_i 、h_i$ 及第 $k$ 棵 CART 树的叶子节点参数值 $\overline c_{kj}$ 有关，而 $g_i 、h_i$ 与第 $k$ 轮迭代无关，所以将其当做常数，在要训练第 $k$ 棵 CART 树，只需要考虑 $\overline c_{kj}$ 参数。

​		对  $c_{kj}$ 求导并令其导数等于 0 ，可得：
$$
\frac{\partial L_k}{\partial \overline c_{kj}} = \sum\limits^T_{j=1}[ (\sum\limits_{x_i \in D_{kj}}g_i) + (\sum\limits_{x_i \in D_{kj}} h_i + \lambda)  \cdot \overline c_{kj}] = 0 \\
$$

$$
\overline c_{kj} = -\frac{\sum\limits_{x_i \in D_{kj}}g_i}{\sum\limits_{x_i \in D_{kj}} h_i + \lambda}
$$

​		将其带入上式可得第 $k$ 轮迭代时等价的损失函数为：
$$
L_k =  - \frac{1}{2}\sum\limits^T_{j=1}[ \frac{(\sum\limits_{x_i \in D_{kj}}g_i)^2}{(\sum\limits_{x_i \in D_{kj}} h_i + \lambda)} ] + \lambda T
$$
​		实际上，第 $k$ 轮迭代的损失函数的优化过程对应的就是第 $k$ 棵树的分裂过程：每次分裂对应于将属于某个叶子结点下的训练样本分配到两个新的叶子节点上；而损失函数满足样本之间的累积性，所以可以通过将分裂前叶子节点上所有样本的损失函数与分裂之后两个新叶子节点上的样本的损失函数进行比较，以此作为各个特征分裂点的打分标准；最后选择一个生成该树的最佳分裂方案。

​		在实践中，当训练数据量较大时，不可能穷举出每一棵树进行打分来选择最好的，因为这个过程计算量太大。所以，通常采用贪心算法的方式来进行逐层选择最佳分裂点。假设一个叶子节点 $I$ 分裂成两个新的叶子节点 $I_L$ 和 $I_R$ ，则该节点分裂产生的增益为：
$$
G_{split} = \frac{1}{2}[ \frac{(\sum\limits_{x_i \in I_{L}}g_i)^2}{(\sum\limits_{x_i \in I_{L}} h_i + \lambda)} + \frac{(\sum\limits_{x_i \in I_{R}}g_i)^2}{(\sum\limits_{x_i \in I_{R}} h_i + \lambda)} - \frac{(\sum\limits_{x_i \in I}g_i)^2}{(\sum\limits_{x_i \in I} h_i + \lambda)} ] - \gamma
$$
​		上面的 $G_{split}$ 表示的就是一个叶子节点 $I$ 按照某特征下的某分裂点分成两个新的叶子节点 $I_L$ 和 $I_R$ 后可以得到的“增益”，该增益值与模型的损失函数值成负相关关系（因为在损失函数的基础上取了负号），即该值越大，就表示按照该分裂方式分裂可以使模型的整体损失减小得越多。

​		所以，XGBoost 采用的是解析解思维，即对损失函数进行二阶泰勒展开，求得解析解，然后用这个解析解作为“增益”来辅助简历 CART 回归树，最终使得整体损失达到最优。

 



## 3、XGBoosting 实践

1. 读取数据

```python
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import xgboost as xgb

df = pd.read_csv('LoanStats3a 2.csv',low_memory=False,skiprows=1)
print(df.shape)
df.head()
```

> ```python
> (42538, 144)
> ```

![1564806207260](E:\MardkDown-Books\机器学习\Boosting\Boosting.assets\1564806207260.png)

2. 数据预处理

```python
df = df.iloc[:,2:111]  # 删掉很多空的列
empty_cols = [i for i in range(45,72)]   # 删除更多的列
df = df.drop(df.columns[empty_cols],axis=1)
df.shapedf = df[(df['loan_status']=="Fully Paid") | (df['loan_status']=="Charged Off")]
df['loan_status'] = df['loan_status'].map({'Fully Paid':0, 'Charged Off':1})
df=df.dropna(axis=1) #340000 is minimum number of non-NA values
df
```

![1564806446117](E:\MardkDown-Books\机器学习\Boosting\Boosting.assets\1564806446117.png)

3. 数据独热编码

```python
df_grade = df['grade'].str.get_dummies().add_prefix('grade: ')
# 把类型独热编码
df_subgrad = df['sub_grade'].apply(str).str.get_dummies().add_prefix('sub_grade: ')
df_home = df['home_ownership'].apply(str).str.get_dummies().add_prefix('home_ownership: ')
df_addr = df['addr_state'].apply(str).str.get_dummies().add_prefix('addr_state: ')
df_term = df['term'].apply(str).str.get_dummies().add_prefix('term: ')

# 添加独热编码数据列
df = pd.concat([df, df_grade, df_subgrad, df_home, df_addr, df_term], axis=1)
# 去除独热编码对应的原始列
df = df.drop(['grade', 'sub_grade', 'home_ownership', 'addr_state', 'int_rate', 'term', 'zip_code','purpose','initial_list_status','initial_list_status','pymnt_plan','issue_d','earliest_cr_line','verification_status'], axis=1)
```

4. 准备数据

```python
# 准备数据
X = df.drop('loan_status', axis=1)
y = df['loan_status']
print (X.shape, y.shape)
```

> ```python
> (39786, 122) (39786,)
> ```

5. 预测的实现

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
xg_classifier = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

xg_classifier.fit(X_train,y_train)
xg_classifier.score(X_test, y_test)
```

> ```python
> 0.9889419452123649
> ```

