#<center>Boosting</center>





​		XGBoost 是 GBDT 的一种改进形式，具有很好的性能。



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
=  \sum\limits^N_{i=1}[ (\sum\limits_{x_i \in D_{kj}}g_i)\cdot c_{kj} + \frac{1}{2}(\sum\limits_{x_i \in D_{kj}} h_i + \lambda) (c_{kj})^2] + \gamma \ T 
$$
 





