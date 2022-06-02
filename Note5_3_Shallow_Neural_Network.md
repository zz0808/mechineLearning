#### activation function
tanh(z): 特殊的，sigmoid用于 二元分类:
$$ tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \in (-1,1)$$
对z求导为$1-tanh(z)^2$:
$$ \begin{split} tanh(z)' &= (1-\frac{2e^{-z}}{e^z+e^{-z}})' \\ &= 2\frac{e^{-z}(e^z+e^{-z})+e^{-z}(e^z-e^{-z})}{(e^z+e^{-z})^2} \\ &= 4\frac{e^{-z}e^z}{(e^z+e^{-z})^2} \\ &= \frac{(e^z+e^{-z})^2-({e^z}^2+{e^{-z}}^2-2e^ze^{-z})}{(e^z+e^{-z})^2} \\ &= 1 - tanh(z)^2\end{split}$$

relu(z): z为0时，在计算机中，是0附近的很小的邻域，理论上应该有两段，默认导数为0. 因为其导数是常数，所以相对tanh速度更快。
$$relu(z) = max(0,z)$$

leaky Relu: relu的缺点时当z为负时，其参数接下来不会被更新，因此leaky relu对z为负的情况下做了修正:
$$g(z)=max(az,z),a是很小的正数$$

#### 为什么需要非线形映射

对于恒等线形映射，无论隐藏层有多少，它最终计算的都是：
$$\hat{y}=wx+b$$
多个线性隐藏层的组合仍然是线程隐藏层，所以没有隐藏层和很多的隐藏层都能学习到相同的效果。

