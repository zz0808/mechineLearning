
We use $(x^{(i)},y^{(i)})$ represent to dataset, and $x_{(i)} \in R^n$, in binary classfication $y^{(i)} \in \{0,1\}$, neural network input in there is $X=[x^{(1)},x^{(2)},...,x^{(m)}]$, so the shape of X is $n \times m$, and output y is $1 \times m$.

#### Logistic Regression in nn

Given $x \in R^n$, want $\hat{y} = P(y=1|x)$, and parameters $w \in R^n$, $b \in R$, so $\hat{y}=g(w^T \cdot x + b )$, $g(z)=\frac{1}{1+e^{-z}}$. in some other implementation, parameter b and w and are tot separate. x will add a $x_{(0)}$, and be shape of $(n+1,1)$, so do w, and $w^{(0)}$ equal to b. We wont do that there.

##### Cost Function
loss function: $L(\hat{y},y)=-ylog(\hat{y})-(1-y)log(1-\hat{y})$.
cost function: $J(w,b)=\frac{1}{m}\sum_{i=1}^{m}L(\hat{y},y)=-\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}log(\hat{y}^{(i)})+(1-y^{(i)})log(1-\hat{y}^{(i)}))$ 

L1 Loss:
$$L1(\hat{y},y)=\sum_{i=1}^{m}|| y^{(i)} - \hat{y}^{(i)}||$$
Note that the difference of loss function and cost function is the loss function compute the error of single training example,and cost function compute the average of loss function of the entire trainning set.

##### Gradient Descent

Cost function is nonconvex function. we will find a $(w,b)$ to minimize $J(w,b)$.

Update rule: $$w=w-\alpha \frac{\partial J(w,b)}{\partial w}, b=b-\alpha \frac{\partial J(w,b)}{\partial b} \tag{1.1}$$.

Compute Graph

![](./img/compute_graph.png)

for examle above, we know:

$$\begin{split} & da = -\frac{y}{a}+\frac{1-y}{1-a} \\ & dz = a-y \\ & dw_1=x_1(a-y) \\ &dw2=x_2(a-y) \\ & db = a-y \end{split}$$
update by (1.1)


