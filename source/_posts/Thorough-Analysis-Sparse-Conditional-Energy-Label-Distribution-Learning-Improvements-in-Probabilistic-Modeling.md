---
title: >-
  Thorough Analysis | Sparse Conditional Energy Label Distribution Learning:
  Improvements in Probabilistic Modeling
date: 2025-10-08 15:34:12
categories:
  - Thorough Analysis
tags:
  - Label Distribution Learning
mathjax: true
---

A close reading of Geng et al., “Sparsity Conditional Energy Label Distribution Learning for Age Estimation” (2016): starting from a conditional energy model, we derive a closed-form transformation that turns the exponential sum over binary latent variables into a “gated product”; we build a joint objective of KL fitting + sparse gating, derive explicit gradients and SGD updates for $b_j$, $u_{jr}$, and $\omega_r$; and we summarize and analyze the experiments.

<!-- more -->

*Paper link: [Sparsity Conditional Energy Label Distribution Learning for Age Estimation](https://www.ijcai.org/Proceedings/16/Papers/322.pdf)*

*Prerequisite article link:*

*[Thorough Analysis | From Single Labels to Label Distributions: A New Approach for Facial Age Estimation](https://jiashun-ye.github.io/en-blog/2025/03/16/Thorough-Analysis-From%20Single-Label%20to%20Label%20Distribution%20A%20New%20Approach%20to%20Facial%20Age%20Estimation/)*

SCE-LDL advances from the maximum-entropy IIS-LDL to “energy + sparse gating” for fine-grained probabilistic modeling. It preserves the philosophy of label distribution learning while significantly improving expressiveness and robustness.

# Sparsity-Conditional Energy Label Distribution Learning

## Deriving the Conditional Probability

```c++
1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    1
    1
    1
    1
    1
    1
    1
    1
    1
    1
    1
    1
    1
    
    
    1
    1
    1
    1
    1
    1
    1
    
    1
    1
    
    1
    1
    
    1
    1
    
    11
    
    1
    1
    1
    1
    
```



The conditional energy model is defined as

$$
p(y\mid x)=\frac{\sum_{h}\exp\{-E(x,y,h)\}}{\sum_{y}\sum_{h}\exp\{-E(x,y,h)\}},
$$

where $h\in\{0,1\}^R$ is a binary latent vector. The energy function is

$$
E(x,y,h)= -\sum_{r=1}^{R} h_r f_r(x;\omega_r)-\sum_{r=1}^{R}\sum_{j=1}^{l} h_r u_{jr} y_j-\sum_{j=1}^{l} b_j y_j.
$$

The symbols mean:

- $x$: input feature vector. In the experiments, visual features such as BIF (Bio-Inspired Features) are used.  
- $y$: age index vector in one-hot form; if the true age falls in class $j$, then only $y_j=1$ and the rest are $0$.  
- $h\in\{0,1\}^R$: binary latent vector of length $R$, interpretable as a set of “feature extractor” gates.  
- $R$: the number of latent variables (or extractors), set by design; $l$ is the total number of age classes.  
- $f_r(x;\omega_r)$: the response of the $r$-th extractor to $x$, with parameters $\omega_r$; can be linear, quadratic, or sigmoid.  
- $u_{jr}$: the weight connecting latent unit $r$ to age $j$, controlling the influence of that extractor on that age.  
- $b_j$: bias term for age $j$.

The full parameter set is $\theta=\{b_j,u_{jr},\omega_r\}$.  
In the conditional probability below, $\exp\{f_r(x;\omega_r)+u_{jr}\}$ can be viewed as the “activation” of extractor $r$ on age $j$. When $f_r(x;\omega_r)+u_{jr}\ll0$, this value is close to $0$, meaning the extractor mismatches this age and is suppressed; when it is large, the activation is strong and multiplicatively amplifies the contribution. Setting $h_r=1$ means the extractor “opens the gate and participates”, while $h_r=0$ means it is “closed”; the whole model thus exhibits an energy structure composed of multiple gated units.

We care about the conditional probability of a fixed age $j$. Using the one-hot property and keeping only the $y_j=1$ terms, we get

$$
p(y_j=1\mid x)=
\frac{\sum_{h}\exp\!\Big(\sum_{r=1}^{R}h_r[f_r(x;\omega_r)+u_{jr}]+b_j\Big)}
{\sum_{k=1}^{l}\sum_{h}\exp\!\Big(\sum_{r=1}^{R}h_r[f_r(x;\omega_r)+u_{kr}]+b_k\Big)}.
$$

The key is to simplify the sum over the vector $h$. Let $A_{rj}=f_r(x;\omega_r)+u_{jr}$. Because the energy is additive in $h_r$ with no interaction terms, we have

$$
\exp\!\Big(\sum_{r}h_rA_{rj}\Big)=\prod_{r}\exp(h_rA_{rj}).
$$

The sum in the numerator can be expanded as nested binary sums:

$$
\sum_{h_1=0}^{1}\cdots\sum_{h_R=0}^{1}\;\prod_{r=1}^{R}\exp(h_rA_{rj}).
$$

Consider the innermost variable $h_R$. Treat everything independent of $h_R$ as a constant:

$$
\sum_{h_R}\exp(h_RA_{Rj})=1+e^{A_{Rj}}.
$$

Thus the innermost sum collapses to $(1+e^{A_{Rj}})$. Proceeding to $h_{R-1}$ yields another factor $(1+e^{A_{R-1,j}})$. Peeling layer by layer, each step leaves a factor $(1+e^{A_{rj}})$. Ultimately,

$$
\sum_{h}\prod_{r=1}^{R}\exp(h_rA_{rj})=\prod_{r=1}^{R}(1+e^{A_{rj}}).
$$

Similarly, the denominator has $\prod_{r=1}^{R}(1+e^{A_{rk}})$ for each age $k$. Therefore, the closed-form conditional probability is

$$
p(y_j=1\mid x)=
\frac{e^{b_j}\prod_{r=1}^{R}\big[1+\exp\{f_r(x;\omega_r)+u_{jr}\}\big]}
{\sum_{k=1}^{l} e^{b_k}\prod_{r=1}^{R}\big[1+\exp\{f_r(x;\omega_r)+u_{kr}\}\big]}.
$$

This derivation shows that, instead of summing over $2^R$ latent configurations, we reduce the problem to a product of $R$ bracketed terms. Intuitively, each latent unit $h_r$ acts as a switch: when $f_r(x;\omega_r)+u_{jr}\ll 0$, the corresponding bracket approximates $1$, i.e., the gate closes and contributes nothing; when it is large, the bracket exceeds $1$ and multiplicatively amplifies the contribution. The combination of all latent units thus forms a “$0$ or $>1$” gated product, enabling distributed representation and laying the groundwork for sparsity regularization.

## Optimization and Updates

Training aims to make the predicted label distribution $p(y\mid x)$ approximate the annotated distribution $D=\{d_j\}_{j=1}^{l}$, while encouraging latent units to be “off” via a sparsity term. Given a training set $\{(x^{(i)},D^{(i)})\}_{i=1}^{N}$, define the (maximized) objective:

$$
\mathcal{L}(\theta)=
\sum_{i=1}^{N}\sum_{j=1}^{l} d^{(i)}_j \log p(y_j=1\mid x^{(i)};\theta)
+\lambda \sum_{i=1}^{N}\sum_{j=1}^{l} d^{(i)}_j \sum_{r=1}^{R} p(h_r=0\mid x^{(i)},y_j=1),
$$

where $\theta=\{b_j,u_{jr},\omega_r\}$, $\lambda$ is the sparsity weight, and $p(h_r=0\mid x,y_j=1)=\sigma(-a_{jr}(x))=1-\sigma(a_{jr}(x))$, with $\sigma(z)=1/(1+e^{-z})$ and $a_{jr}(x)=f_r(x;\omega_r)+u_{jr}$. The closed-form conditional probability is

$$
p(y_j=1\mid x)=
\frac{e^{b_j}\prod_{r=1}^{R}\big[1+e^{a_{jr}(x)}\big]}
{\sum_{k=1}^{l} e^{b_k}\prod_{r=1}^{R}\big[1+e^{a_{kr}(x)}\big]}.
$$

For a single sample $(x,D)$, define $\ell_j(x)=\log p(y_j=1\mid x)=b_j+\sum_{r}\log(1+e^{a_{jr}(x)})-\log Z(x)$, where $Z(x)=\sum_{k}e^{b_k}\prod_{r}(1+e^{a_{kr}(x)})$. Using $\partial\log(1+e^z)/\partial z=\sigma(z)$ and the softmax property, we obtain

$$
\frac{\partial \ell_j}{\partial b_k}=\mathbf{1}[k=j]-p(y_k=1\mid x),\qquad
\frac{\partial \ell_j}{\partial u_{kr}}=\big(\mathbf{1}[k=j]-p(y_k=1\mid x)\big)\sigma(a_{kr}(x)),
$$

$$
\frac{\partial \ell_j}{\partial \omega_r}=
\Big(\sigma(a_{jr}(x))-\sum_{k}p(y_k=1\mid x)\sigma(a_{kr}(x))\Big)\frac{\partial f_r(x;\omega_r)}{\partial \omega_r}.
$$

The gradients of the sparsity regularizer are

$$
\frac{\partial}{\partial u_{kr}}\sigma(-a_{kr}(x))
=-\sigma(a_{kr}(x))(1-\sigma(a_{kr}(x))),\qquad
\frac{\partial}{\partial \omega_r}\sigma(-a_{jr}(x))
=-\sigma(a_{jr}(x))(1-\sigma(a_{jr}(x)))\frac{\partial f_r(x;\omega_r)}{\partial \omega_r}.
$$

This regularizer does not involve $b_k$, so its derivative w.r.t. the biases is $0$.

Combining both parts and summing over samples weighted by label distributions, the total gradients are:

$$
\frac{\partial \mathcal{L}}{\partial b_k}=\sum_{i}\sum_{j} d^{(i)}_j\big(\mathbf{1}[k=j]-p_k^{(i)}\big),
$$

$$
\frac{\partial \mathcal{L}}{\partial u_{kr}}
=\sum_{i}\sum_{j} d^{(i)}_j\Big[\big(\mathbf{1}[k=j]-p_k^{(i)}\big)\sigma(a^{(i)}_{kr})\Big]
-\lambda\sum_{i} d^{(i)}_k\,\sigma(a^{(i)}_{kr})(1-\sigma(a^{(i)}_{kr})),
$$

$$
\frac{\partial \mathcal{L}}{\partial \omega_r}
=\sum_{i}\sum_{j} d^{(i)}_j\Big[\Big(\sigma(a^{(i)}_{jr})-\sum_{k} p_k^{(i)}\,\sigma(a^{(i)}_{kr})\Big)\frac{\partial f_r(x^{(i)};\omega_r)}{\partial \omega_r}\Big]
-\lambda\sum_{i}\sum_{j} d^{(i)}_j\,\sigma(a^{(i)}_{jr})(1-\sigma(a^{(i)}_{jr}))\frac{\partial f_r(x^{(i)};\omega_r)}{\partial \omega_r}.
$$

In practice, parameters are updated via mini-batch stochastic gradient **ascent** (or gradient **descent** on $-\mathcal{L}$):

$$
b_k \leftarrow b_k+\eta\,\frac{\partial \mathcal{L}}{\partial b_k},\qquad
u_{kr} \leftarrow u_{kr}+\eta\,\frac{\partial \mathcal{L}}{\partial u_{kr}},\qquad
\omega_r \leftarrow \omega_r+\eta\,\frac{\partial \mathcal{L}}{\partial \omega_r},
$$

where $\eta$ is the learning rate. When $f_r(x;\omega_r)=\omega_r^\top x+w_{r0}$, the gradients are $\partial f_r/\partial \omega_r=x$, $\partial f_r/\partial w_{r0}=1$, so the bias can be folded into the weight vector. In the paper’s experiments, linear extractors, a learning rate of about $0.1$, and $30$ training epochs suffice to converge. By stacking a sparsity regularizer on top of the log-likelihood gradient for KL matching, the algorithm jointly achieves “probability matching + sparse hidden-unit activation”, so each latent unit learns to activate on compatible samples and shut off on mismatched ones, forming a stable sparse gating structure.

# Experiments and Result Analysis

Experiments are conducted mainly on two public facial age estimation datasets: MORPH II and ChaLearn Apparent Age. MORPH II contains $55{,}132$ facial images with age range $16$–$77$, with an approximate racial composition of $77\%$ African-American, $19\%$ Caucasian, and $4\%$ Others. After face detection and alignment, BIF (Bio-Inspired Features) are extracted, then reduced to $200$ dimensions using MFA (Marginal Fisher Analysis), followed by zero-mean and unit-variance normalization. The ChaLearn dataset contains $3{,}615$ images ($2{,}479$ for training and $1{,}136$ for validation), also processed with DPM detection and five-point alignment. BIF features of $7{,}152$ dimensions are extracted without dimensionality reduction. Label distributions are constructed as follows: on MORPH, a Gaussian with mean at the true age and standard deviation $1$ is used to generate $D_j$; on ChaLearn, the official mean and standard deviation are plugged into the same Gaussian formula. Evaluation metrics include Mean Absolute Error (MAE) and the cumulative score curve CS@m (i.e., the proportion of samples with error less than $m$ years). The study compares a range of classic LDL methods, including CE-LDL, BFGS-LDL, CPNN-LDL, KPLS, and the traditional regression method OHR, all under identical features and distribution-construction protocols.

Results show that SCE-LDL achieves the best performance on both datasets. On MORPH, MAE is $3.858\pm0.090$, significantly better than CE-LDL ($4.146$), BFGS-LDL ($4.872$), CPNN-LDL ($4.705$), KPLS ($4.442$), and OHR ($7.085$). On ChaLearn, MAE reaches $6.432$, clearly improving over CE-LDL ($6.708$) and KPLS ($6.841$), and far surpassing CPNN-LDL ($12.979$). The CS curves (error thresholds from $0$ to $10$) further indicate that SCE-LDL lies higher in the low-error region, evidencing stronger advantages in high-accuracy predictions. Sensitivity studies analyze the effects of the number of latent units $R$ and the sparsity coefficient $\lambda$: as $R$ varies from $50$ to $175$, MAE changes smoothly, peaking around $150$ yet remaining fairly insensitive; for $\lambda$ in the range $0.001$–$0.01$, performance is stable, with the best around $0.005$. The experiments also show that linear extractors suffice to achieve strong performance; nonlinear forms bring limited gains at higher computational cost. Overall, SCE-LDL outperforms other LDL methods by realizing “selective activation” through latent gating and sparsity: when features match a given age, the latent units activate and multiplicatively boost the probability; mismatched features are naturally suppressed, significantly enhancing robustness. This is especially effective on in-the-wild datasets like ChaLearn, where complex conditions—lighting, expression, etc.—prevail: the gating mechanism filters noisy features so that the predicted distribution becomes more concentrated and stable. In summary, combining energy modeling with sparse gating not only surpasses existing LDL methods in performance but also points to a direction where label distribution learning can further enhance representational power via latent energy structures at the feature level.
