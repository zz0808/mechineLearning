### Logistic Resression

在机器学习中，逻辑回归常用于二元分类， 对于数据集 $(x^{(i)}, y^{(i)})$, 其中输出 $y^{(i)} \in \{0,1\}$, 逻辑回归模型表示为：
$$h_{\theta}(x)=g(\theta^{T}x)$$
$$g(z)=\frac{1}{1+e^{-z}}$$
预测结果范围在 (0,1), 这意味着，如果 $h_{\theta}(x)=0.7$, 表示预测为1的概率为70%，所以我们可以认为预测输出为 "1", 即, 
$$y = \begin{cases} 
0 & h_{\theta}(x) < 0.5 \\
1 & h_{\theta}(x) \geq 0.5 
\end{cases}$$
也就是：
$$
h_{\theta}(x)=P(y=1|x;\theta) = 1 - P(y=0|x;\theta)
$$ 

#### Cost function

逻辑回归的代价函数为
$$ J(\theta) = \frac{1}{m} \sum_{i=1}^{m}Cost(h_{\theta}(x^{(i)}),y^{(i)})$$ 

其中

$$Cost(h_{\theta}(x^{(i)}),y^{(i)}) = \begin{cases} 
-log(h_{\theta}(x)) & y = 1 \\ -log(1 - h_{\theta}{(x)}) & y = 0 \\
\end{cases}$$
当 $y=1$ 并且 $h_{\theta}(x)$ 接近 1, $J(\theta)$越小，显然这是合理的,对于 $y=0$时也是这样. 总结来说,
$$J(\theta)=-\frac{1}{m} \sum_{i=1}^{m}(y^{(i)} \cdot log(h_{\theta}(x^{(i)})) + (1-y^{(i)}) \cdot log(1 - h_{\theta}{(x^{(i)})}))$$

#### Gradient Descent
重复更新：
$$\begin{split} \theta_{j} &= \theta_{j} - \alpha \cdot \frac{\partial}{\partial \theta_{j}}J(\theta) \\ &=  \theta_{j} - \frac{\alpha}{m} \cdot \sum_{i=1}^{m}((h_{\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)})\end{split}$$

向量化：

$$\left[ \begin{matrix} \frac{\partial J(\theta)}{\partial \theta_{0}} \\ \frac{\partial J(\theta)}{\partial \theta_{1}} \\ \vdots \\ \frac{\partial J(\theta)}{\partial \theta_{n}} \end{matrix}\right] = \frac{1}{m}\left[ \begin{matrix} \sum_{i=1}^{m}((h_{\theta}(x^{(i)})-y^{(i)})\cdot x^{(i)}_{0}) \\ \sum_{i=1}^{m}((h_{\theta}(x^{(i)})-y^{(i)})\cdot x^{(i)}_{1}) \\ \vdots \\ \sum_{i=1}^{m}((h_{\theta}(x^{(i)})-y^{(i)})\cdot x^{(i)}_{j}) \end{matrix}\right] = \frac{1}{m}\sum_{i=1}^{m}((h_{\theta}(x^{(i)})-y^{(i)})\cdot x^{(i)})=\frac{1}{m}X^T\cdot (h_{\theta(x)}-y)$$ 

这里

$$ h_{\theta(x)}-y = \left[ \begin{matrix} h_{\theta}(x^{(0)})-y^{(0)} \\ h_{\theta}(x^{(1)})-y^{(1)} \\ \vdots \\  h_{\theta}(x^{(m)})-y^{(m)}\end{matrix} \right]$$

#### 向量化正则化后的逻辑回归

$$J(\theta) = -\frac{1}{m}\sum_i^m(y^{(i)}\cdot log(h_{\theta}(x^{(i)}))+(1-y^{(i)}) \cdot log(1-h_{\theta}(x^{(i)}))) + \frac{\lambda}{2m}\sum_{j=1}^{m}\theta^{2}_{j}$$

so 

$$ \frac{\partial J(\theta)}{\partial \theta_{j}}= \begin{cases} 
\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)}-y^{(i)})\cdot x_j^{(i)}) & j=0 \\
 \frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)}-y^{(i)})\cdot x_j^{(i)}) + \frac{\lambda}{m}\theta_{j} & j > 0 
\end{cases}$$ 


#### Reference
[mechine learning](https://www.coursera.org/learn/machine-learning/home/week/3)

