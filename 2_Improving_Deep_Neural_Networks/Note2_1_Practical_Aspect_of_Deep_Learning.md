#### Setting up your Machine Learning Application

##### Train/dev/test set

训练集用于训练模型，验证集用于在多个模型中评估哪一个效果最好，测试集用于对结果最好的模型进行评估，这样使得评估不会有误差。
当样本较少时，往往三者的比例为6/2/2，当时数据较大时，比如百万级别，dev/test的比例会相对减少，比如99/1/1。一个要点是尽量保证三者具有相同的数据
分布。

##### Bias/Variance

高偏差，意味着欠拟合，高方差意味着过拟合。一般可以通过训练集和验证集的误差来判断，如果训练集误差很低而验证集误差相对较高，则意味着过拟合，模型在新的数据
上泛化能力并不好，即高方差的；如果训练集误差较高而验证集误差与之相对接近，则意味着欠拟合，模型在训练集表现并不好，即高偏差的；如果训练集误差较高而验证集误差与之相对更高，则表现为即高偏差又高偏差，这意味着模型是及其糟糕的；如果训练集和验证集上的误差都比较低，这意味着低偏差低方差，这是理想的模型。

一般我们假设误差为0时，称为理想误差（也叫贝叶斯误差：接近0%），以上误差的相对值都是基于贝叶斯误差和两者数据集分布接近而言的，即如果理想误差为15%，就不能认为训练集误差接近15%时是高偏差的。

对于高偏差问题，可以通过加深网络深度和增加迭代次数尝试解决，多数情况而言，加深网络深度总是有效的，对于高方差问题，可以通过增加数据规模和正则化尝试解决，但有时候增加数据规模并不容易，因此正则化也被经常采用。

#### Regularizing your network

##### Regularization

L2正则化：对于cost function，添加$\frac{\lambda}{2m}||w||^2_2$，超参数$\lambda$称为正则化参数，为什么没有对偏置b正则化呢，也是可以添加的，但往往会忽略，因为参数w相对b具有较高的维度，当高方差时，意味着模型泛化能力差，模型的参数没有很好的拟合，w会有很多的参数，而b只有一个参数，所以w占的权重更大，而b相对占的权重比较小，所以一般只在正则化考虑w，b由于占的参数比较少，即使添上也不会起多大作用。

L1正则化：对于cost function，添加$\frac{\lambda}{2m}||w||_1$。使用L1正则化使得模型变得稀疏，即w会有很多0，有些人认为有利于压缩模型，但在实际中这种方式收效甚微，起码在压缩模型方面。所以L2正则化更常使用。

为什么L1会使得W矩阵稀疏，可以从导数方面来看，假设w只有一个，在w=0处的导数为$d_0$，在不引入范数时有
$$\begin{matrix} \frac{\partial L(w)}{\partial w} = d_0, & w=0\end{matrix}$$
引入范数后，容易知道对于L2范数
$$\begin{split} \frac{\partial J(w)}{\partial w} = d_0=\frac{\lambda}{m}w+d_0 \\ \end{split}$$
对于L1范数
$$\frac{\partial J(w)}{\partial w} = \begin{cases}  
d_0 - \frac{\lambda}{2m} &  w=0左边 \\
d_0 + \frac{\lambda}{2m} & w=0右边
\end{cases}$$
可以看到对于L1范数，在w=0处导数有突变，也就是w在此处有极小值点，优化时可能优化到w=0附近。

##### Why regularization reduces overfitting

假如$\lambda$很大，在反向传播过程中有很多w会被更新为0或者接近0。这意味着网络中很多神经元的影响可以被忽略了，即一个大的神经网络此时可被视为一个小的网络。
当W很小的时候 $Z=WX+b$也会很小，如果使用tanh激活函数，他就类似线性的，不能拟合比较复杂的函数，所以能防治过拟合。

##### Dropout regularization

随机丢弃(dropout)通过遍历网络每一层，并设置每个神经元被丢弃的概率，这样，某些神经元会被冻结，在fp和bp的时候不考虑他们。每次训练一个样例，都在训练一个较小的网络，从而减少过拟合。

常见dropout实现方式为反向随机失活：
+ 对于神经网络第$l$层，其输出为$a_l$；
+ 设置和$a_l$大小相同的矩阵$b_l$，它的元素取值是（0,1）随机数，然后让他小于keep_prob，其意义是神经元不失活概率，使它的元素变为0或1（False or True），之后和$a_l$作点乘；
+ $a_l$ / keep_prob作为下一层的计算。这是因为$a_l$丢弃了一些神经元（比如丢弃了20%），为了使得$a_l$的期望值（均值）大致不变，这样做可以弥补20%。

值得注意的是，测试阶段不需要使用随机失活，这是因为，你不想让你的结果是随机的，使用随机失活会增加噪声。

##### 其他正则化方法

1. 增加数据量，但是有时是件困难的事，所以可以通过旋转\缩放\扭曲原始图像等方法增加伪数据。
2. 早终止法(early stopping), 画出迭代次数关于训练代价和测试代价的函数曲线，一般测试代价函数会先下降，然后在某次迭代上上升，可以在这个时候停止训练。

#### Setting up your optimization problem

##### 输入归一化

归一化分为两个步骤，一是减平均值
$u=\frac{1}{m}\sum x_i, x_i = x_i - u$
如果把输入样本画在坐标系上，这使得所有输入在原点周围。
二是归一化方差，方差不均可能会有每个特征的绝对值相差比较大，处理完后，各个特征的绝对值会比较接近。

之所以要归一化，是因为特征取值范围如果差异较大，则从数学角度较大范围的特征会占较大比重，但在实际过程中可能并非如此。并且在训练时会影响梯度方向，降低训练效率。

##### 梯度消失/梯度爆炸

是指在训练时损失函数的导数会变得很小或者很大。








