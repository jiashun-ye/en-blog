---
title: "Thorough Analysis | End-to-End Label Distribution Learning: Breaking Fixed Functional-Form Constraints"
date: 2025-09-11 18:47:46
categories:
  - Thorough Analysis
tags:
  - Label Distribution Learning
mathjax: true
---

This post presents a thorough reading of the 2013 paper *Facial Age Estimation by Learning from Label Distributions* by Xin Geng et al. The work extends the 2011 label distribution learning (LDL) framework and the IIS-LLD optimization method by proposing the Conditional Probability Neural Network (CPNN), which directly models the conditional distribution in an end-to-end manner and breaks the limitation of fixed functional-form assumptions. This article focuses on the CPNN modeling pipeline and summarizes its experimental results on the FG-NET and MORPH datasets.

<!-- more -->

*Paper link: [Facial Age Estimation by Learning from Label Distributions](https://ieeexplore.ieee.org/document/6475129)*

*Previous post: [Thorough Analysis | From Single-Label to Label Distribution: A New Approach to Facial Age Estimation](https://jiashun-ye.github.io/en-blog/2025/03/16/Accuracy%20Analysis-From%20Single-Label%20to%20Label%20Distribution%20A%20New%20Approach%20to%20Facial%20Age%20Estimation/)*

Based on the maximum-entropy model, IIS-LLD needs to pre-specify an exponential-family form for the conditional probability distribution; this assumption has limitations and can lead to model misspecification. To overcome this, the paper proposes CPNN, which models $p(y\mid x)$ end-to-end with neural networks, avoids any specific functional-form assumptions, and leverages label-distribution supervision to achieve more flexible and accurate age estimation.

# Flexible Conditional Probability Modeling

The goal is to directly learn the conditional distribution $p(y\mid x)$. Let the input feature be $x\in\mathbb{R}^q$, the discrete age set $\mathcal Y=\{y_1,\dots,y_c\}$, and the given training-time label distribution be $D_x(y)$. Unlike the maximum-entropy model $p(y\mid x;\theta)\propto \exp\!\big(\langle\theta,\phi(x,y)\rangle\big)$, a three-layer perceptron directly parameterizes $p(y\mid x)$ to improve nonlinear fitting capacity and avoid model misspecification. This is the CPNN algorithm.

A straightforward design would feed $x\in\mathbb{R}^q$ and output a length-$c$ age distribution in one shot, but that introduces $\mathcal O(H\!\cdot\!c)$ parameters in the last layer (with $H$ the number of hidden units) and is hard to train robustly with limited data. Instead, the age $y$ is also treated as input. Using a shared-parameter function
$$
f_\theta(x,y)=\mathrm{MLP}_\theta\!\big([\;x,\ \mathrm{enc}(y)\;]\big)\in\mathbb{R},
$$
we enumerate all candidate ages $y\in\mathcal Y$ for the same image to obtain $\{f_\theta(x,y)\}_{y\in\mathcal Y}$, and then apply exponential normalization to get
$$
p_\theta(y\mid x)=\frac{\exp\!\big(f_\theta(x,y)\big)}{\sum_{y'\in\mathcal Y}\exp\!\big(f_\theta(x,y')\big)}.
$$
This “$y$-as-input, single scalar output” design preserves the softmax normalization while markedly reducing parameter count and naturally sharing parameters across neighboring ages.

The training objective is the distribution-level cross-entropy
$$
\mathcal L(\theta)=-\sum_{y\in\mathcal Y} D_x(y)\,\log p_\theta(y\mid x),\qquad
p_\theta(y\mid x)=\frac{\exp(f_\theta(x,y))}{\sum_{y'}\exp(f_\theta(x,y'))}.
$$
Its top-layer gradient is $\partial \mathcal L/\partial f_\theta(x,y)=p_\theta(y\mid x)-D_x(y)$. In backpropagation, the output layer is linear, $f_\theta(x,y)=\sum_{m=1}^{M_2}\theta_{31m}\,z^{(2)}_m(x,y)$, where $z^{(2)}_m=G(I_{2m})$ is the $m$-th hidden unit output, $I_{2m}=\sum_{k=0}^{M_1}\theta_{2mk}\,u_k$ is its net input, $u=[1,x,\mathrm{enc}(y)]$ concatenates the bias, features, and age encoding, $G$ is sigmoid with derivative $G'$. With $\theta_{31m}$ the hidden-to-output weights and $\theta_{2mk}$ the input-to-hidden weights, we have
$$
\frac{\partial \mathcal L}{\partial \theta_{31m}}
=\sum_{y\in\mathcal Y}\big(p_\theta(y\mid x)-D_x(y)\big)\,z^{(2)}_m(x,y),\qquad
$$

$$
\frac{\partial \mathcal L}{\partial \theta_{2mk}}
=\sum_{y\in\mathcal Y}\big(p_\theta(y\mid x)-D_x(y)\big)\,\theta_{31m}\,G'(I_{2m}(x,y))\,u_k(x,y),
$$

and in practice standard BP suffices for training.

# Experiments and Results

FG-NET uses LOPO validation with MAE; the images come from 82 subjects with 1002 images in total, ages 0–69. MORPH uses 10-fold cross-validation and reports MAE±std, ages 16–77. The 200-dimensional features are produced by different pipelines: FG-NET uses the Appearance Model to extract joint shape and grayscale features and takes the first 200 model parameters; MORPH uses Bio-Inspired Features (BIF) and then applies Marginal Fisher Analysis (MFA) to reduce to 200 dimensions, enabling a fair comparison at the same dimensionality.

As this article focuses on CPNN, we only report its performance. The network is a three-layer perceptron with 400 hidden units; age $y$ is concatenated with image features as input, softmax produces $p(y\mid x)$, and the model is trained with distribution-level cross-entropy. Label distributions include three ablations: Gaussian, Triangle, and Single (degenerate), to quantify the “soft supervision” from neighboring ages. Compared with IIS-LLD under the same features and protocols, CPNN achieves lower average error but is more data-dependent with slightly higher variance, while IIS-LLD is more stable and faster due to its maximum-entropy shape prior and coordinate-wise incremental updates; in data-reduction experiments, IIS-LLD exhibits a smoother curve. Focusing on CPNN’s ablations, setting Gaussian $\sigma$ and Triangle base $l$ too narrow (approaching Single) weakens neighboring supervision, while too wide dilutes the primary label; a moderate width is best (empirically $\sigma\!\approx\!2$, $l\!\approx\!6$). Here, “more pronounced gains in the elderly sparse range” means that relative to age ranges with abundant young samples, and relative to baselines such as Single and AGES, the relative improvement in elderly ranges (e.g., 60–69) is larger.

{% raw %}

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  table { width: 76%; margin: 14px auto; border-collapse: collapse; font-size: 0.94em; }
  thead th { border-top: 3px solid #000; border-bottom: 2px solid #000; padding: 4px 6px; text-align: center; background: #f6f6f6; }
  tbody td { padding: 2px 6px; text-align: left; border-bottom: 1px solid #ddd; }
  tbody tr:last-child td { border-bottom: 3px solid #000; }
  caption { font-size: 1.12em; margin: 4px 0 2px; }
</style>
</head>
<body>
  <table>
    <caption>CPNN vs. IIS-LLD: Pros and Cons</caption>
    <thead>
      <tr>
        <th>Method</th>
        <th>Pros</th>
        <th>Limitations</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>IIS-LLD</td>
        <td>Maximum-entropy shape prior brings stability; coordinate-wise incremental updates converge faster; smoother curves when training data is reduced</td>
        <td>Pre-specified $p(y|x)$ shape may mismatch the task; parameters are learned per label, under-utilizing cross-label correlations</td>
      </tr>
      <tr>
        <td>CPNN</td>
        <td>No assumed shape for $p(y|x)$, more flexible expression; all labels share parameters, explicitly exploiting neighboring-age correlation; lower average error</td>
        <td>More data-dependent with slightly higher variance; end-to-end training is slower and needs regularization plus moderate distribution width to curb overfitting</td>
      </tr>
    </tbody>
  </table>
</body>
</html>
{% endraw %}


Causally, CPNN directly models $p(y|x)$ and shares parameters across all labels, injecting the “neighboring ages are similar” structure into the network via soft supervision. With a proper distribution width, it achieves lower MAE when data is relatively sufficient or features are strong; the higher flexibility also makes it more sensitive to data scale, with slightly higher variance. IIS-LLD fixes $p(y|x)$ to an exponential-family form under maximum entropy and optimizes a lower bound via coordinate-wise increments, yielding more controllable updates and better computational efficiency; thus, it often suits scenarios with limited data that demand robustness and efficiency. However, this shape constraint limits absorption of cross-label correlations, so in large-data or strong-feature settings, its average error typically trails CPNN. In practice: if you aim for peak accuracy and have the resources, prefer CPNN with appropriate $\sigma$/$l$ and regularization; if you prioritize training/inference efficiency or face limited data, prefer IIS-LLD.
