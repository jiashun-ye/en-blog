---
title: >-
  Thorough Analysis | Deep Label Distribution Learning 2.0: Joint Optimization
  of Distribution and Expectation
date: 2025-10-07 16:37:39
categories:
  - Thorough Analysis
tags:
  - Label Distribution Learning
mathjax: true
---

This post closely reads Geng et al.’s “Age Estimation Using Expectation of Label Distribution Learning” (2018) from four angles—method, theory, engineering, and experiments. We first derive the CDF of the Gaussian label distribution and prove the limit equivalence Ranking ≈ 1 − CDF to unify the two lines; then we present a joint learning framework (KL + L1) and the logits gradient to explain the alignment with MAE; next we analyze Thin/TinyAgeNet and hybrid pooling for lightweighting; finally we summarize the experimental setup and conclusions, clarifying the robust gains of “distribution + expectation.”

<!-- more -->

*Paper link: [Age Estimation Using Expectation of Label Distribution Learning](https://openaccess.thecvf.com/content_iccv_2015_workshops/w11/papers/Yang_Deep_Label_Distribution_ICCV_2015_paper.pdf)*

*Prerequisite posts:*

*[Thorough Analysis | From Single Labels to Label Distributions: A New Approach for Facial Age Estimation](https://jiashun-ye.github.io/en-blog/2025/03/16/Thorough-Analysis-From%20Single-Label%20to%20Label%20Distribution%20A%20New%20Approach%20to%20Facial%20Age%20Estimation/)*

*[Thorough Analysis | Deep Label Distribution Learning: Distributional Supervision and Practical Construction](https://jiashun-ye.github.io/en-blog/2025/10/07/Thorough-Analysis-Deep-Label-Distribution-Learning-Distributional-Supervision-and-Practical-Construction/)*

# Joint Optimization of Distribution and Expectation

## Ranking is Essentially Learning the Label Distribution

Let the age random variable be $A\sim\mathcal N(y,\sigma^2)$, whose Gaussian label distribution density is
$$
f_{y,\sigma}(t)=\frac{1}{\sigma\sqrt{2\pi}}\exp\!\Big(-\frac{(t-y)^2}{2\sigma^2}\Big).
$$

The cumulative distribution value at the discrete grid point $l_k$ is
$$
c_k \;=\; F(l_k)\;=\;\int_{-\infty}^{\,l_k} f_{y,\sigma}(t)\,dt.
$$
With the substitution $u=\frac{t-y}{\sigma\sqrt{2}}$ (thus $dt=\sigma\sqrt{2}\,du$), we obtain the standard closed form
$$
c_k \;=\;\int_{-\infty}^{\,\frac{l_k-y}{\sigma\sqrt{2}}}\frac{1}{\sqrt{\pi}}e^{-u^2}\,du
\;=\;\tfrac12\Big[1+\operatorname{erf}\!\Big(\frac{l_k-y}{\sigma\sqrt{2}}\Big)\Big].
$$
Here $\operatorname{erf}(x)$ is the error function, defined by
$$
\operatorname{erf}(x)\;=\;\frac{2}{\sqrt{\pi}}\int_{0}^{x} e^{-t^2}\,dt,
$$
and it satisfies the relation with the standard normal CDF $\Phi(\cdot)$: $\;\Phi(z)=\tfrac12\!\big[1+\operatorname{erf}(z/\sqrt{2})\big]$.

From the above we see that when $l_k<y$ we have $1-c_k>0.5$, when $l_k>y$ we have $1-c_k<0.5$, and at $l_k=y$ we have $1-c_k=0.5$.

Ranking supervision learns, at each threshold $l_k$, the indicator of the event $\{A>l_k\}$. The ideal label can be written as
$$
(p_{\mathrm{rank}})_k \;=\; \mathbf 1\{y>l_k\} \;=\; 1 - F_{\delta_y}(l_k),
$$
where $F_{\delta_y}$ is the CDF of the degenerate distribution concentrated at $y$. Noting that the Gaussian family weakly converges to $\delta_y$ as $\sigma\to0$, for any fixed $l_k\ne y$ we have the limit
$$
\lim_{\sigma\to 0}\big[\,1-F(l_k;y,\sigma)\,\big]
=\begin{cases}
1, & l_k<y,\\[2pt]
0, & l_k>y,
\end{cases}
$$
i.e.,
$$
\lim_{\sigma\to0}(1-c_k)\;=\;\mathbf 1\{y>l_k\}.
$$

Therefore, $1-\mathrm{CDF}$ pointwise approaches the step labels of Ranking in the limit $\sigma\to0$; when $\sigma>0$, the label distribution not only gives the right-tail probability relative to the threshold but also explicitly preserves peak shape and uncertainty through the adjustable $\sigma$, hence being more expressive than Ranking, which provides only left/right cumulative relations.

## Joint Learning Framework

The goal of the joint learning framework is to learn both the label distribution and its expectation so that the training objective aligns with the evaluation MAE. Given the ground-truth age $y$ and a discrete grid $\{l_k\}_{k=1}^{K}$, we first generate the label distribution on the grid using a Gaussian:
$$
p_k \;=\; \frac{\exp\!\big(-\tfrac{(l_k-y)^2}{2\sigma^2}\big)}{\sum_{j=1}^{K}\exp\!\big(-\tfrac{(l_j-y)^2}{2\sigma^2}\big)}.
$$
Let the features be mapped linearly to logits $x\in\mathbb R^{K}$, and let the predicted distribution be given by softmax:
$$
\hat p_k \;=\; \frac{e^{x_k}}{\sum_{j=1}^{K}e^{x_j}},
$$
followed by a parameter-free expectation layer:
$$
\hat y \;=\; \sum_{k=1}^{K}\hat p_k\,l_k.
$$
The total loss combines the KL divergence on the distribution side with the L1 error on the expectation side:
$$
L \;=\; \underbrace{\sum_{k=1}^{K} p_k\,\log\frac{p_k}{\hat p_k}}_{L_{\mathrm{ld}}}
\;+\;
\lambda\,\underbrace{|\hat y - y|}_{L_{\mathrm{er}}}.
$$
This objective unifies “learning the shape of the distribution” ($L_{\mathrm{ld}}$) and “learning the final age value” ($L_{\mathrm{er}}$) in the same network end-to-end, where the expectation layer introduces no learnable parameters.

The gradient of the joint loss with respect to the logits can be written explicitly. Note that the gradient of $L_{\mathrm{ld}}$ with respect to $x$ is $\hat p - p$, and since $\hat y=\sum_k \hat p_k l_k$ and $\tfrac{\partial \hat p_k}{\partial x_i}=\hat p_k(\mathbf 1\{i=k\}-\hat p_i)$, we have
$$
\frac{\partial \hat y}{\partial x_i}
=\sum_{k=1}^{K} l_k\,\frac{\partial \hat p_k}{\partial x_i}
=\hat p_i\,(l_i-\hat y),
$$
and with $L_{\mathrm{er}}=|\hat y-y|$ we have $\tfrac{\partial L_{\mathrm{er}}}{\partial \hat y}=\operatorname{sign}(\hat y-y)$, yielding the total gradient
$$
\frac{\partial L}{\partial x_i}
\;=\;
\hat p_i - p_i
\;+\;
\lambda\,\operatorname{sign}(\hat y-y)\,\hat p_i\,(l_i-\hat y),
\qquad i=1,\dots,K.
$$
We see that the distribution term and the expectation term are naturally coupled at the logits: the first term pushes $\hat p$ toward the label distribution $p$, while the second term redistributes probability mass along the age axis according to the deviation between $\hat y$ and $y$. As $\lambda\!\to\!0$ this degenerates to pure distribution learning as in DLDL; if only the expectation term is kept, it is equivalent to parameter-free expectation regression over softmax probabilities. During training, use the above to backpropagate through the network as usual; during inference, directly read $\hat y$ as the predicted age (optionally averaging predictions from the original image and its horizontal flip).

## Lightweight Network Design

The motivation for lightweight network design is to maintain—or even improve—the effectiveness of joint training of distribution learning and expectation regression with smaller model capacity and compute. In practice, we start from VGG16 and slim it structurally: first remove all fully connected layers to avoid massive parameters from high-dimensional vector–to–class space mappings; next, replace the “conv—flatten—FC” route with hybrid pooling, obtaining a compact representation by combining global average pooling and global max pooling, then attach a linear mapping and softmax to output the distribution and connect the expectation layer. The convolutional part uniformly reduces channels per stage and inserts batch normalization after each convolution to stabilize training and alleviate optimization difficulties caused by reduced capacity. These changes together allow ThinAgeNet and TinyAgeNet to significantly reduce parameter count and memory bandwidth consumption without sacrificing representational completeness, while improving inference throughput.

The motivation for hybrid pooling is to avoid over-smoothing of strong responses by single global average pooling, while global max pooling retains activation peaks of salient regions. Their combination provides a more robust global description for fine-grained tasks like age estimation and can directly replace expensive fully connected layers without clearly increasing overfitting risk. The accompanying channel reduction and batch normalization strike a more reasonable balance between effective capacity and optimization difficulty, yielding a more stable convergence path for the joint loss.

In terms of scale and efficiency, this design reduces the parameter count from the traditional VGG16 level to a few million (ThinAgeNet ≈ 3.7M, TinyAgeNet ≈ 0.9M), while delivering multi-fold inference speedups (≈ 2.6× and 5.5×). The joint learning framework on these backbones still achieves error comparable to or better than large models. Therefore, the lightweight backbone is not an engineering add-on independent of the method, but a structural choice adapted to the end-to-end goal of “distribution + expectation”: by reducing redundant parameters and improving the effectiveness and stability of feature aggregation, it provides sufficient—but not excessive—representational power for both distribution fitting and expectation regression.

# Experiments and Results Analysis

Experimental setup: The authors evaluate on three benchmark datasets: ChaLearn15 and ChaLearn16 are in-the-wild apparent age datasets, each image annotated with a mean and standard deviation; Morph is a larger real age dataset. ChaLearn15 follows the official train/val/test split; ChaLearn16 trains on the official train+val and reports on the test set; Morph is randomly split 80%/20% for train/test as commonly done. The primary metric is MAE, and ε-error is also reported for the ChaLearn challenge datasets. Implementation uses the lightweight backbones ThinAgeNet/TinyAgeNet with aligned and standardized face inputs and standard data augmentation during training; the network head linearly maps to logits, uses softmax to output the predicted age distribution, then a parameter-free expectation layer to obtain a scalar age, and is optimized end-to-end with the joint loss of KL (distribution side) + L1 (expectation side). At inference, run once on the original image and once on the horizontal flip and average.

Results analysis: Overall comparisons show that DLDL-v2 pushes the single-model MAE on Morph down to 1.969, the first report below two years; without using external age annotations, it also achieves performance on ChaLearn15/16 that matches or surpasses contemporaneous methods. The lightweight backbones substantially reduce complexity: ThinAgeNet has about 3.7M parameters and TinyAgeNet about 0.9M, roughly 36×/150× smaller than VGG16-based DEX/DLDL, with forward throughput improved by about 2.6×/5.5×. Ablations show both engineering strategies are consistently effective: data augmentation significantly improves MAE on both apparent and Morph datasets, and hybrid pooling further reduces error across all three datasets compared with global average pooling alone. Comparisons with strong baselines also support the paper’s theoretical reading: Ranking and DLDL clearly outperform direct regression and DEX, and DLDL is slightly better than Ranking, consistent with the conclusion that label distributions carry more information than Ranking. With joint optimization of “distribution + expectation,” the overall best results are obtained, indicating that aligning the training objective with the evaluation metric is key. Hyperparameter sweeps show the method is insensitive to $\lambda$ and age step size over a fairly wide range, exhibiting good tunability and deployability.
