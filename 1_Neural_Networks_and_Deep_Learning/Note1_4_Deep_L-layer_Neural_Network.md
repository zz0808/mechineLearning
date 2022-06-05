#### Notation

这里，使用$n^{(l)}$表示第l层神经元个数，$a^{(l)}$表示第l层使用的激活函数, $W^{(l)}$表示第l层的权重矩阵，$b^{(l)}$表示第l层的偏置矩阵。
注意到，$W^{(l)}$和$dW^{(l)}$的维度为$(n^{(l)},n^{(l-1)})$，$b^{(l)}$和$db^{(l)}$的维度为$(n^{(l)},1)$。
注意到，以上时针对1个输入数据的情况，对于训练集m，知道：$$W = [W^{(1)},W^{(2)},...,W^{(m)}]$$
所以可以知道，对于整个数据集：$b^{(l)}$的维度会通过广播机制变为为$(n^{(l)},m)$，$Z^{(l)}$和$A^{(l)}$维度为$(n^{(l)},m)$，$dZ^{(l)}$和$dA^{(l)}$维度因此也为$(n^{(l)},m)$，

#### Parameters and Hyperparameters

参数：权重矩阵W和偏置b，是最终学习到的东西。
超参数：如学习率$\alpha$、隐藏层个数，每个隐藏层中神经元的数量等，这些是你自己设计的，他决定了网络的设计框架，并且，同样他决定了参数的最终值。

一般编程步骤：idea ---> code ---> experiment

