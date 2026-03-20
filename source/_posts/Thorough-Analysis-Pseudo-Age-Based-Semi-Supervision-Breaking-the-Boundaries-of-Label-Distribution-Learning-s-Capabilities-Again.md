---
title: >-
  Thorough Analysis | Pseudo-Age-Based Semi-Supervision: Breaking the Boundaries
  of Label Distribution Learning's Capabilities Again
date: 2025-10-07 21:39:38
categories:
  - Thorough Analysis
tags:
  - Label Distribution Learning
mathjax: true
---

A close reading of Geng et al.’s “Semi-Supervised Adaptive Label Distribution Learning for Facial Age Estimation” (2017). Building on a recap of LDL/ALDL, this post unifies notation and lays out the minimal closed‑loop pipeline of SALDL (conditional distribution prediction → pseudo‑age KNN → age‑wise σ adaptation). It also makes explicit that pseudo‑age estimation for **all samples** is based on the current model’s predicted distributions, and it summarizes the MORPH experimental protocol and comparative results.

<!-- more -->

*Paper link: [Semi-Supervised Adaptive Label Distribution Learning for Facial Age Estimation](https://ojs.aaai.org/index.php/AAAI/article/view/10822?utm_source=chatgpt.com)*

*Prerequisite posts:*

*[Thorough Analysis | From Single Labels to Label Distributions: A New Approach for Facial Age Estimation](https://jiashun-ye.github.io/en-blog/2025/03/16/Thorough-Analysis-From%20Single-Label%20to%20Label%20Distribution%20A%20New%20Approach%20to%20Facial%20Age%20Estimation/)*

*[Thorough Analysis | From Unified Variance to Age-Specific Variance: Adaptive Label Distributions and Joint Optimization](https://jiashun-ye.github.io/en-blog/2025/09/17/Thorough-Analysis-From-Unified-Variance-to-Age-Specific-Variance-Adaptive-Label-Distributions-and-Joint-Optimization/)*

In age estimation with scarce labels, Label Distribution Learning (LDL) mitigates variance and overfitting via soft weighting over neighboring ages. Adaptive LDL (ALDL) further learns an age‑wise variance σ, but unlabeled samples must be discarded, wasting information. To leverage LDL’s “neighbor effect” in a semi‑supervised manner—so that unlabeled data are constrained at the distribution level and can participate in training—SALDL couples unlabeled‑sample usage with age‑wise adaptive σ through alternating optimization, yielding more robust estimates under low‑label conditions.

## Semi‑Supervised Adaptive Label Distribution Learning (SALDL)

Let the labeled set be $S_l=\{(x_i,\mu_i)\}$, the unlabeled set $S_u=\{x_m\}$, and the age label space $\mathcal{Y}=\{y_1,\dots,y_L\}$. In each iteration, SALDL links “conditional distribution prediction → pseudo‑age estimation → age‑wise $\,\sigma\,$ adaptation” into a closed loop. The optimization of $\,\Theta\,$ and $\{\sigma_y\}$ follows the KL‑minimization and alternating strategy introduced in the ALDL post; here we streamline the pipeline and detail pseudo‑age estimation for unlabeled samples.

**(1) Conditional distribution prediction.** With current parameters $\Theta^{(k)}$, compute predicted label distributions for **all samples** (both $S_l$ and $S_u$):
$$
p^{(k)}(y\mid x)=\frac{\exp\!\big(\theta_y^{(k)\top} x\big)}{\sum_{y'\in\mathcal{Y}}\exp\!\big(\theta_{y'}^{(k)\top} x\big)}.
$$
For samples in $S_l$, these predictions are used to minimize KL against their target distributions to update $\Theta$. For samples in $S_u$, these predictions provide the distributional basis for pseudo‑age estimation (next step).

**(2) Pseudo‑age estimation (unlabeled samples).** For each $x_m\in S_u$, select $K$ nearest neighbors in $S_l$ using a **feature + predicted‑distribution** composite distance:
$$
\lambda(x_m,x_n)=\lVert x_m-x_n\rVert_2^2\;+\;\alpha\,\mathrm{KL}\!\Big(p^{(k)}(\cdot\mid x_m)\,\Big\|\,p^{(k)}(\cdot\mid x_n)\Big),\quad x_n\in S_l.
$$
Here $x_m\in S_u$, $x_n\in S_l$, $p^{(k)}(y\mid x)$ is the iteration‑$k$ predicted distribution, and $\alpha>0$ balances feature distance and distribution discrepancy.

Let the neighbor set be $\mathcal{N}(x_m)$ with true ages $\{\mu_n: x_n\in\mathcal{N}(x_m)\}$. Define the **pseudo‑age** of $x_m$ as
$$
\tilde{\mu}_m^{(k)}=\frac{1}{K}\sum_{x_n\in\mathcal{N}(x_m)}\mu_n,
$$
and construct its target label distribution (a discrete Gaussian using age‑wise variance $\sigma_y^{(k)}$):
$$
d_m^{(k)}(y)\propto \exp\!\left(-\frac{(y-\tilde{\mu}_m^{(k)})^2}{2(\sigma_y^{(k)})^2}\right),\quad y\in\mathcal{Y}.
$$
All the distributional information used in pseudo‑age estimation comes solely from the current model’s predicted label distributions $p^{(k)}(\cdot\mid x)$; unlabeled samples require neither prior ages nor initialized targets.

**(3) Age‑wise $\,\sigma\,$ adaptation and target reconstruction.** Compute the point estimate $\hat{\mu}(x)=\arg\max_{y\in\mathcal{Y}} p^{(k)}(y\mid x)$ and the reference age $\mu_{\text{ref}}(x)$ (for labeled samples use $\mu_i$; for unlabeled samples use $\tilde{\mu}_m^{(k)}$). Define the error $e(x)=\lvert \hat{\mu}(x)-\mu_{\text{ref}}(x)\rvert$ and select samples with $e(x)$ below the current MAE as the **trusted set**, then group them by age $y$. Next, for each $y$, solve a one‑dimensional constrained problem to obtain the updated age‑wise variance $\sigma_y^{(k+1)}$ that minimizes the total KL between the Gaussian target and the trusted samples’ predicted distributions. Finally, using $\{\sigma_y^{(k+1)}\}$, reconstruct target distributions for all samples—combining with true ages $\mu_i$ (for $S_l$) or pseudo‑ages $\tilde{\mu}_m^{(k)}$ (for $S_u$)—and proceed to the next round of KL minimization for $\Theta$.

**Notation.**  

- $x\in\mathbb{R}^d$: numeric feature vector of an image; $\mu$: true age; $\tilde{\mu}$: pseudo‑age; $\hat{\mu}$: point estimate from the argmax of the predicted distribution.  
- $\mathcal{Y}$: discrete age set; $d(y\mid\cdot,\sigma_y)$: discrete Gaussian label distribution with mean at age and age‑wise variance $\sigma_y$; $Z$: normalization constant.  
- $p^{(k)}(y\mid x)$: iteration‑$k$ predicted distribution for sample $x$; $\Theta^{(k)}$: model parameters; $\theta_y^{(k)}$: class‑conditional weight vector.  
- $\lambda(\cdot,\cdot)$: composite distance; $\mathrm{KL}(\cdot\|\cdot)$: Kullback–Leibler divergence; $K$: number of neighbors; $\alpha>0$: trade‑off hyperparameter.  
- $\sigma_y$: variance for age $y$ (age‑wise adaptive); MAE: current round’s mean absolute error threshold for selecting trusted samples.

The algorithmic workflow is illustrated below.

![](SALDL FlowChart.png)

## Experiments and Results

The setup follows common age‑estimation practice: extract BIF features on MORPH and reduce dimensionality via MFA; fix a held‑out test set; on the training side, simulate “label scarcity” by controlling the labeled proportion while keeping the total number of samples roughly constant. Semi‑supervised methods (SLDL, SALDL) use additional unlabeled samples at every labeled scale. Baselines include KPLS, OHRank, LDL, ALDL, and label propagation (LP). The evaluation metric is MAE. Hyperparameters are chosen by cross‑validation; a typical choice is initial variance $\sigma_0=3$, neighbors $K=10$, composite‑distance weight $\alpha=10^{-3}$, and the maximum number of iterations decreases moderately as the labeled set grows. At inference, the point estimate is $y^*=\arg\max_y p(y\mid x;\Theta^*)$.

Results show that under low labeled counts (e.g., $\leq 10^3$), SALDL consistently outperforms all baselines in MAE. The gains come from two sources: (i) pseudo‑age estimation converts unlabeled data into effective supervision by combining feature distance with distributional proximity; (ii) age‑wise variance adaptation minimizes the KL between target and predicted distributions on the trusted set, aligning label distributions to the varying “rates of change” across ages. Comparatively, SLDL clearly outperforms ALDL (which only adapts), demonstrating the benefit of semi‑supervision; SALDL further improves upon SLDL, indicating that alternating semi‑supervision and adaptation has additive effects. When labels are plentiful, method gaps shrink; in the limit of “fully labeled,” SALDL reduces to ALDL. Even when unlabeled and labeled sources differ in gender/ethnicity composition, increasing unlabeled volume continues to reduce error, reflecting robustness of pseudo‑age estimation and adaptation to distribution shift.
