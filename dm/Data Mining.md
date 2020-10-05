## Outlier and Noise

### we define k nearest neighbors: $distance_k(o)$

$lrd(A)=1/(\frac{\sum_{B in Dataset}(A)distance_k(A,B)}{|N_K(A)|})$

$LOF_k(A)=\frac{\sum\frac{lrd(B)}{lrd(A)}}{|N_K(A)|}$

*point with highest LOF means it being outlier*

## Data Preprocessing

##### Duplicating Data

the outline: create keys => sort => merge

create keys: to detect key words being able to differ all data

sort: to sort data

merge: to detect and eliminate repeating items

##### Coping with Imbalanced Dataset

When should we be aware of imbalanced dataset? Look at this scenario. You are assigned to cope with a mission detecting whether a tumor is benign or not. The case is that, the probability of the tumor being malignant is only 1%. After training your model, it classified all tumor benign, with an accuracy 99%, which makes no sense at all. This is what we call an imbalanced dataset.

G-mean: $(Acc^+\times Acc^-)^{\frac{1}{2}}$, $Acc^+=\frac{TP}{TP+FN}$, $Acc^-=\frac{TN}{TN+FP}$

F-measure: $\frac{2\times precision \times recall}{precision+recall}$

With two measures above, we are able to detect whether a model is doing a good job detecting those incredible odds.

##### Normalization

**min-max** $v'=\frac{v-min}{max-min}(new_max-new_min)+new_min$

**z-score** $v'=\frac{v-\mu}{\sigma}$

##### Mean

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



## Naïve Bayes

classification: supervised learning

basis: $P(A|B)=\frac{P(B|A)P(A)}{P(B)}$

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



##### Overfitting and Pruning



##### Entropy Bias