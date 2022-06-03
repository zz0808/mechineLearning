### Linear Regression

 数据集格式$(x^{(i)}, y^{(i)})$, 大小为 $m$, 第i条数据表示为$i$, 对于第i条数据输入$x^{(i)}$, 假设有 $n$ 个特征, 即对于 $x$,
$$x = \{x_0, x_1, ..., x_n\}$$
上面 $x$ 的维度为 $n+1$, 这是为了简化计算, $x_0$ 常等于 1, 使得 $\theta_0$ 是一个常数项。
为了线性拟合数据，我们假设:
$$h_{\theta}=\theta_0x_0 + \theta_1x_1 + .., + \theta_nx_n=\theta^Tx$$ 
这里 $h^{(i)}_{\theta}$ 是LR回归对 $x^{(i)}$的预测结果, 这里 $x_0 = 1$。

为了验证预测结果的可靠性，即和真实结果的接近程度，引入代价函数：
$$J(\theta) = \frac{1}{m}\sum_{i=0}^{m}(h_{\theta}(x)^{{(i)}}-y^{(i)})^2$$
 $J(\theta)$ 越小, 意味着线性回归模型对数据集的拟合程度越好，所以我们的目标是选择一组合适的$\theta=[\theta_0,\theta_{1}, ..., \theta{n}]$ 使得最小化 $J(\theta)$。

#### Gradient Descent

我们通过梯度下降方法更新 $J(\theta)$，来获取其最优解：
$$ \begin{aligned} \theta_j &= \theta_j-\alpha \cdot \frac{1}{m} \frac{\partial J(\theta)}{\partial \theta_{j}} 
\end{aligned} $$
当 $m=1$时

$$ \begin{aligned} \frac{\partial J(\theta)}{\partial \theta_{i}} &= 
2 \cdot \frac{1}{2} (h_{\theta}-y) \cdot \frac{\partial} {\partial \theta{{j}}}(h_{\theta}{x}-y) \\ &= 2 \cdot \frac{1}{2} (h_{\theta}-y) \frac{\partial} {\partial \theta{{j}}}(\sum_{i=0}^{m}\theta_{i}x_{i}-y) \\ &= (h_{\theta}-y)x_j \end{aligned}  $$

所以有

$$ \theta_j = \theta_j-\alpha \cdot \frac{1}{m} (h_{\theta}-y)x_{j} $$

这就是梯度下降。

#### Normal equation

正规方程是另一种获取 $\theta$的方法.

$$\theta=(X^{T}X)^{-1} \cdot (X^{T}y)$$

如果 $X^{T}X$ 不可逆, 一般因为:
+  冗余特征, 其中两个特征密切相关（即线性相关）
+ 特征太多 (e.g. m ≤ n). 可以考虑删除部分特征，或者使用“正则化”。

#### Reference
[Mechine Learning](https://www.coursera.org/learn/machine-learning/home/week/2)
