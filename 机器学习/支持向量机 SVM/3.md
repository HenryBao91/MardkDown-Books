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