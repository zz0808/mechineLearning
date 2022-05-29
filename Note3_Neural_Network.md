### Neural Network

We assume dataset $X$, and its dimension is $m \times n$. and hypothesis $h_{\theta}(x)=g(\theta x)$, $g(z)=\frac{1}{1+e^{-z}}$.

#### Cost Function

$$J(\theta)=-\frac{1}{m} \sum_{i=1}^{m}(y^{(i)} \cdot log(h_{\theta}(x^{(i)}))+(1-y^{(i)}) \cdot log(1-h_{\theta}(x^{(i)}
))$$
so
$$X= \left[ \begin{matrix} x^{(1)^T} \\ x^{(2)^T} \\ \vdots \\ x^{(m)^T} \end{matrix} \right], \theta= \left[ \begin{matrix} \theta^{(0)} \\ \theta^{(1)} \\ \vdots \\\theta^{(n)}\end{matrix} \right]
$$
here expand $X$, now $x^{(i)}=\{x^{(i)}_{0},x^{(i)}_{1}, \ldots,x^{(i)}_{n} \}$,  $\theta^{(i)} $ and $x^{(1)^T} $ both are vector, so
$$X\theta=\left[ \begin{matrix} \theta^T x^{(1)} \\ \theta^T x^{(2)} \\ \vdots \\ \theta^T x^{(m)} \end{matrix} \right]$$

#### Gradient

$$\frac{\partial J(\theta)}{\partial \theta_{j}}=\frac{1}{m}\sum_{i=1}^{m}((h_{\theta}(x^{(i)})-y^{(i)})\cdot x^{(i)}_{j})$$

To vectorize this operation over the dataset, we can write

$$\left[ \begin{matrix} \frac{\partial J(\theta)}{\partial \theta_{0}} \\ \frac{\partial J(\theta)}{\partial \theta_{1}} \\ \vdots \\ \frac{\partial J(\theta)}{\partial \theta_{n}} \end{matrix}\right] = \frac{1}{m}\left[ \begin{matrix} \sum_{i=1}^{m}((h_{\theta}(x^{(i)})-y^{(i)})\cdot x^{(i)}_{0}) \\ \sum_{i=1}^{m}((h_{\theta}(x^{(i)})-y^{(i)})\cdot x^{(i)}_{1}) \\ \vdots \\ \sum_{i=1}^{m}((h_{\theta}(x^{(i)})-y^{(i)})\cdot x^{(i)}_{j}) \end{matrix}\right] = \frac{1}{m}\sum_{i=1}^{m}((h_{\theta}(x^{(i)})-y^{(i)})\cdot x^{(i)})=\frac{1}{m}X^T\cdot (h_{\theta(x)}-y)$$ 

where 

$$ h_{\theta(x)}-y = \left[ \begin{matrix} h_{\theta}(x^{(0)})-y^{(0)} \\ h_{\theta}(x^{(1)})-y^{(1)} \\ \vdots \\  h_{\theta}(x^{(m)})-y^{(m)}\end{matrix} \right]$$

#### Vectorizing regularized logistic regression

$$J(\theta) = -\frac{1}{m}\sum_i^m(y^{(i)}\cdot log(h_{\theta}(x^{(i)}))+(1-y^{(i)}) \cdot log(1-h_{\theta}(x^{(i)}))) + \frac{\lambda}{2m}\sum_{j=1}^{m}\theta^{2}_{j}$$

so 

$$ \frac{\partial J(\theta)}{\partial \theta_{j}}= \begin{cases} 
\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)}-y^{(i)})\cdot x_j^{(i)}) & j=0 \\
 \frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)}-y^{(i)})\cdot x_j^{(i)}) + \frac{\lambda}{m}\theta_{j} & j > 0 
\end{cases}$$ 

