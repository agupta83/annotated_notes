---
toc: false
layout: post
description: Notes on MLE
categories: [statistics, mle]
title: Notes on Maximum likelihood estimation
---

The process of estimating parameter $\theta$ from data $\mathcal{D}$ is called model
fitting and MLE is one of the most common approach. MLE picks the parameter
{% cite pml1book %}
that assigns highest probability to the training data.

$$
\hat{\theta}_{mle} \triangleq \arg\max_\theta p(\mathcal{D}|\theta)
$$

Under **iid** assumption the likelihood is given by

$$
p(\mathcal{D}|\theta) = \prod_{i=1}^N p(y_i|x_i; \theta)
$$

The **log likelihood** is defined as

$$
\begin{aligned}
LL(\theta) &\triangleq \text{log } p(\mathcal{D}|\theta)\\ 
           &= \sum_{i=1}^N \text{log } p(y_i|x_i; \theta)
\end{aligned}
$$

Maximizing some function $f(\cdot)$ is same as minimizing $-f(\cdot)$. The **negative log likelihood** is given by

$$
\begin{aligned}
NLL(\theta) &\triangleq - \text{log } p(\mathcal{D}|\theta)\\ 
            &= -\sum_{i=1}^N \text{log } p(y_i|x_i; \theta)
\end{aligned}
$$

Minimizing NLL will give us the MLE 

$$
\begin{aligned}
\hat{\theta}_{mle} &= \argmin_\theta - \text{log } p(\mathcal{D}|\theta)\\ 
                   &= \argmin_\theta -\sum_{i=1}^N \text{log } p(y_i|x_i; \theta)
\end{aligned}
$$

### MLE as point estimate to Bayesian posterior

MLE can be thought as a point estimate of Bayesian posterior $p(\theta|\mathcal{D})$
under uniform prior. The model of the posterior is given by

$$
\begin{aligned}
\hat{\theta}_{map} &= \argmax_\theta \text{log } p(\theta|\mathcal{D}) \\
                   &= \argmax_\theta \text{log } p(\mathcal{D}|\theta) + \text{log } p(\theta) \\
                   &= \argmax_\theta \text{log } p(\mathcal{D}|\theta) + const \text{  \{uniform prior\}}\\
                   &= \hat{\theta}_{mle}
\end{aligned}
$$

### MLE is equivalent to minimizing KL 
Computing MLE is equivalent to minimizing **Kullback Leibler (KL) divergence** with
empirical distribution of data. The empirical distribution of data for unsupervised 
case is given by

$$
p_{data} \triangleq = \frac{1}{N}\sum_{i=1}^N\delta(y - y_i)
$$

When estimating parameter, we want model $q(y) = p(y|\theta)$ that is similar to 
empirical distribution. We can measure the goodness of the fitted model by calculating
distance between $p_{data}$, the empirical distribution and $q$, the model.

$$
\begin{aligned}
\mathbb{KL}(p||q) &= \sum_{y} p(y) \text{ log } \frac{p(y)}{q(y)}\\
                  &= \sum_{y} p(y) \text{ log } p(y) - \sum_y p(y) \text{ log } q(y)\\
                  &= \sum_{y} p_{data}(y) \text{ log } p_{data}(y) - \sum_y p_{data}(y) \text{ log } q(y)\\
                  &= -\mathbb{H}(p_{data}) - \frac{1}{N}\sum_{i=1}^N\text{ log } q(y)\\
                  &= -\mathbb{H}(p_{data}) - \frac{1}{N}\sum_{i=1}^N\text{ log } p(y_i|\theta)\\
                  &= const + NLL(\theta)
\end{aligned}
$$

### Examples
#### Bernoulli distribution

Let $\theta = p(y=head)$ then NLL of this distribution is given by

$$
\begin{aligned}
NLL(\theta) &= -\sum_{i=1}^{N}\text{log } p(y_i; \theta) \\
            &= -\sum_{i=1}^{N}\text{log }\big(\theta^{\mathbb{I}(y_i = 1)}\cdot(1 - \theta)^{\mathbb{I}(y_i = 0)}\big)\\
            &= -\sum_{i=1}^{N} \mathbb{I}(y_i = 1) log(\theta) + \mathbb{I}(y_i = 0) log(1 - \theta)\\
            &= -\big[N_{head} log(\theta) + N_{tail} log(1 - \theta)\big]
\end{aligned}
$$

MLE can be found by solving $\frac{d}{d\theta} NLL(\theta) = 0$

$$
\frac{d}{d\theta} NLL(\theta) = -\frac{N_{head}}{\theta} + \frac{N_{tail}}{(1 - \theta)} = 0
$$

Solving for $\theta$ gives

$$
\hat{\theta}_{mle} = \frac{N_{head}}{N_{head} + N_{tail}}
$$

#### Categorical distrbution
Categorical distribution is generalization of bernouli to $k$-category. The NLL is 
given by 

$$
NLL(\theta) = -\sum_{k}N_{k}log(\theta_k)
$$

To compute MLE we have to minimize NLL with contraint $\sum_k^K \theta_k = 1$. Using 
lagrangian


$$
\begin{aligned}
    \mathcal{L}(\theta, \lambda) &\triangleq -\sum_{k}N_{k}log(\theta_k) -\lambda\Big(1 -\sum_k^K \theta_k \Big)\\
    \frac{\partial\mathcal{L}}{\partial\lambda} &= \Big(1 -\sum_k^K \theta_k\Big)=0\\
    \frac{\partial\mathcal{L}}{\partial\theta_k} &= -\frac{N_k}{\theta_k}+ \lambda=0 \implies N_k = \lambda\theta_k
\end{aligned}
$$

Using sum to one contraint we have

$$
\begin{aligned}
\sum_k N_k &= N\\
\sum_k \lambda\theta_k &= N\\
\lambda \sum_k\theta_k &= N \\
\lambda &= N
\end{aligned}
$$

Thus MLE is given by

$$
\hat{\theta_k} = \frac{N_k}{\lambda} = \frac{N_k}{N}
$$

# References
{% bibliography --cited %}