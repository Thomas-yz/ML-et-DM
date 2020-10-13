## Outlier versus Anomaly

### we define k nearest neighbors: $distance_k(O)$

$distance_k(A,B)=max\{distance_k(B),d(A,B)\}$

$lrd(A)=1/(\frac{\sum_{B \in N_k(A)}distance_k(A,B)}{|N_K(A)|})$

$LOF_k(A)=\frac{\sum\frac{lrd(B)}{lrd(A)}}{|N_K(A)|}=\frac{\sum lrd(B)}{|N_K(A)|}/lrd(A)$

*higher LOF means higher possibility of being outlier*

## Data Preprocessing

##### Duplicate Data

the outline: create keys => sort => merge

create keys: to detect key words being able to differ all data

sort: to sort data, making similar data together

merge: to detect and eliminate repeating items

##### Coping with Imbalanced Dataset

When should we be aware of imbalanced dataset? Look at this scenario. You are assigned to cope with a mission detecting whether a tumor is benign or not. The case is that, the probability of the tumor being malignant is only 1%. After training your model, it classified all tumor benign, with an accuracy 99%, which makes no sense at all. This is what we call an imbalanced dataset.

G-mean: $(Acc^+\times Acc^-)^{\frac{1}{2}}$, $Acc^+=\frac{TP}{TP+FN}$, $Acc^-=\frac{TN}{TN+FP}$

F-measure: $\frac{2\times precision \times recall}{precision+recall}$

With two measures above, we are able to detect whether a model is doing a good job detecting those incredible odds.

##### Normalization

**min-max** $v'=\frac{v-min}{max-min}(new_max-new_min)+new_min$

**z-score** $v'=\frac{v-\mu}{\sigma}$

##### Data Description

**arithmetic mean** $\frac{1}{n}\sum^n_ix_i$

**median** $P(X≤m)=P(X≥m)=\frac{1}{2}$

**variance** $Var(x)=E[(X-\mu)^2]$

*<————when variance is high, we use median more often than arithmetic mean————>*

**Pearson's product moment correlation coefficient** $r_{A,B}=\frac{\sum(A-\overline A)(B-\overline B)}{(n-1)\sigma_A\sigma_B}$

*<————If $r_{A,B}>0$, A and B are positively correlated.————>*

*<————If $r_{A,B}<0$, A and B are negatively correlated.————>*

*<————If $r_{A,B}=0$, there is no **linear** correlation between A and B.————>*

**Pearson's chi-square test** $\chi^2=\sum\frac{(Observed-Expected)^2}{Expected}$

## Feature Selection

##### PCA

<img src="..\dm\2\PCA.png" alt="PCA" style="zoom: 50%;" />

$J(e)=\sum^n_{k=1}{||x'_k-x_k||^2}=-\sum^n_{k=1}{e^tx_kx^t_ke}+\sum^n_{k=1}{||x_k||^2}$ 

to get $max(e^tSe) s.t. ||e||=1$, $(S=x_kx^t_k)$

we get $Se=\lambda e$, which is an eigenvector problem

##### LDA

$J(w)=\frac{|\mu_1-\mu_2|^2}{S^2_1+S^2_2}=\frac{w^TS_Bw}{w^TS_ww}$

to get $\frac{d}{dw}[J(w)]=0$

we get $S^{-1}_WS_Bw=Jw$$Se=\lambda e$, which is also an eigenvector problem

## Naïve Bayes

classification: supervised learning

basis: $P(A|B)=\frac{P(B|A)P(A)}{P(B)}$

##### Naïve Bayes Classifier

$\omega_{MAP}=arg_{\omega_i\in\omega}maxP(\omega_i|a_1,a_2,...,a_n)$

$\omega_{MAP}=arg_{\omega_i\in\omega}max\frac{P(a_1,a_2,...,a_n)P(\omega_i)}{P(a_1,a_2,...,a_n)}$

$\omega_{MAP}=arg_{\omega_i\in\omega}maxP(a_1,a_2,...,a_n)P(\omega_i)$

*for naïve Bayes,we suppose as' being conditionally independent*

$\omega_{MAP}=arg_{\omega_i\in\omega}maxP(\omega_i)\prod_jP(a_j|\omega_i)$

##### Laplace smoothing

thoughts: we should not make 0 count as outlier

$P(a_jk|\omega_i)=\frac{|a_j=a_{jk}\wedge \omega=\omega_i|+1}{|\omega=\omega_i|+|a_j|}$

##### Bag of words

thoughts: we put words from the articles we research into a bag

$P(V_k|\omega_i)=\frac{n_k+1}{n+|Vocabulary|}$

## Decision Tree 

##### Entropy

$Entropy(S)=-\sum_{i=1}^c{p_ilog(p_i)}$ a measurement on information

##### Information gain

we use information to measure the information we get from a branch

$Gain(S,A)=Entropy(S)-\sum_{v\in S}{\frac{|S_v|}{S}Entropy(S_v)}$ 

*the higher the better*

##### ID3

Create a *Root* node for the tree.

If *Examples* have the same target attribute T, return *Root* with label=T.

If *Attributes* is empty, return *Root* with label=the most common value of *Target_attribute* in *Examples*.



A <- the attribute from *Attributes* that best classifies *Examples*.

The decision attribute for *Root* <- A.



For each possible value vi of A

​	■ Add a new tree branch below *Root*, corresponding to A= vi.

​	■ Let *Examples* (*v_i*) be the subset of Examples that have value vi for A.

​	■ If *Examples* (*v_i*) is empty: Below this new branch add a leaf node with label=the most common value of *Target_attribute* in *Examples*.

​	■ Else below this new branch add the subtree: 

​		□ **ID3(Examples(v_i), Target_attribute, Attributes-{A})**

##### Overfitting and Pruning

What results into overfitting: Random noise and insufficient samples

Solution: 

​	■ Stop growing the tree earlier

​	■ Allow the tree to overfit the data and then post-prune the tree

##### Entropy Bias

Food for thought: We have worst attributes with best Entropy, such as birthday

Solution: don't let them be arrtibute

## Neural Networks

##### Perceptron

TL;NW

<img src="..\dm\3\perceptron.png" alt="PCA" style="zoom: 50%;" />

##### Delta Rule(Gradient Descent)(Batch Learning)

$E(\overrightarrow w)=\frac{1}{2}\sum_{d\in D}(t_d-o_d)^2 $

$\nabla E(\overrightarrow w)=[\frac{\partial E}{\partial w_0},\frac{\partial E}{\partial w_1},...,\frac{\partial E}{\partial w_n}]$

$w_i\leftarrow w_i+\Delta w_i$ where $\Delta w_i=-\eta \frac{\partial E}{\partial w_i}$

$\frac{\partial E}{\partial w_i}=\sum_{d\in D}(t_d-o_d)(-x_id)$

##### Batch Learning versus Stochastic Learning

$w_i\leftarrow w_i+\Delta w_i$

Batch Learning's way:		$\Delta w_i=-\eta \frac{\partial E}{\partial w_i}$

Stochastic Learning's way:	$\Delta w_i=-\eta (t-o)x_i$

##### Multilayer Perceptron

TL;NW

##### Backpropagation Rule

$E_d(\overrightarrow w)=\frac{1}{2}\sum_{k\in outputs}(t_k-o_k)^2 $

 $\Delta w_{ji}=-\eta \frac{\partial E_d}{\partial w_{ji}}$

$\frac{\partial E_d}{\partial w_{ji}}=\frac{\partial E_d}{\partial net_j}\cdot \frac{\partial net_j}{\partial w_{ji}}=\frac{\partial E_d}{\partial net_j}x_{ji}$

training rule for output units: 

$\Delta w_{ji}=-\eta\frac{\partial E_d}{\partial w_{ji}}=\eta (t_j-o_j)o_j(1-o_j)x_{ji}$

training rule for hidden units: 

$\delta_j=o_j(1-o_j)\sum_{k\in Downstream(j)}\delta_kw){kj}$

$\Delta w_{ji}=\eta\delta_jx_{ji}$