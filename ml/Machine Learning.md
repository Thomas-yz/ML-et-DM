## Regression Model

##### Liner case

$h_\theta(x)=\theta_0+\theta_1x$

$J(\theta)=\frac{1}{2m}\sum_{n=1}^m(h_\theta(x^{(i)}-y^{(i)}))^2$

## Gradient Descent

##### Basis 

to step down a hill slowly:walking: or​ ​quickly:runner:

##### Algorithm

repeat until convergence{

​	$\theta_j:=\theta_j-\alpha\frac{\partial }{\partial \theta_j}J(\theta_0,\theta_1)$ (for j=0 and j=1)

}

remember to update $\theta$ simultaneously

##### Explaining

if your $\alpha$ is appropriate, this algorithm will lead you to a "local" minimum, when slope equal to 0

##### Linear case

repeat until convergence{

​	$\theta_0:=\theta_0-\alpha\frac{1}{m}\sum_{n=1}^m(h_\theta(x^{(i)}-y^{(i)}))$

​	$\theta_1:=\theta_1-\alpha\frac{1}{m}\sum_{n=1}^m(h_\theta(x^{(i)}-y^{(i)}))\cdot x^{(i)}$

}

having one global optima

## Matrix et Vector

#### Basis

Vector: An n $\times$ 1 Matrix

uppercase -> matrix || lowercase -> vector

##### Multiplying 

$A_{m\times n}\times B_{n\times o}=C_{m\times o}$

specifically,an m$\times$n matrix multiplying n$\times$1 matrix,

​	$A_{m\times n}\times x_{n\times 1}=y_{m\times 1}$,

making it an m-dimensional vector

##### Properties

$A\times B \neq B\times A$

$(A\times B)\times C = A\times (B\times C)$

Denoted $I$

## N-Dimension Regression Model