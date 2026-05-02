# FORGETME: Certified Data Removal POC

**Paper:** Guo, Goldstein, Hannun, Van der Maaten  
**Certified Data Removal from Machine Learning Models (ICML 2020)**  
https://proceedings.mlr.press/v119/guo20c/guo20c.pdf

**Demo:** https://forgetme-nine.vercel.app/

---

## Overview

This project is an interactive proof of concept based on the ICML 2020 paper *Certified Data Removal from Machine Learning Models*. The core question is:

Can we remove a training point from a trained machine learning model without retraining, while still guaranteeing that the model behaves as if it never saw that point?

The paper shows that this is possible using:

- A perturbed training objective
- A Newton-based update rule for removal
- A formal certification bound that guarantees correctness

This demo implements the full pipeline with real matrix computations running in the browser.

---

## Core Idea

### 1. Perturbed Training Objective (Section 3.1)

During training, noise is injected into the objective:

L_b(w; D) = L(w; D) + b^T w

where:

- b ~ N(0, σ^2 I)

This perturbation ensures that individual data points have bounded and controllable influence on the final model.

In the demo, this is executed when you click **Train Model**.

---

### 2. Certified Data Removal (Section 3.2, Equation 6)

To remove a set of points S, the model is updated using:

w^- = w* + H^-1 Δ

where:

- H is the Hessian of the loss at w*
- Δ = sum(i in S) grad ℓ_i(w*) + λ|S| w*

This replaces retraining with a single Newton update step.

---

### 3. Certification Guarantee

The paper provides formal guarantees on how much the model can change after removal.

Worst-case bound (Theorem 1):

4 γ C^2 / (λ^2 (n - 1))

Data-dependent bound (Corollary 1):

γ ||X'||_2 * ||H^-1 Δ||_2 * ||X' H^-1 Δ||_2

The system tracks these bounds in real time.

The privacy budget is:

epsilon * sigma / c, where c = sqrt(2 ln(1.5 / delta))

If the budget is exceeded, full retraining is triggered.

---

## Features

### Dataset Explorer

Synthetic Gaussian binary classification data is used instead of raw datasets for performance and clarity.

- Adjustable class sizes
- Feature dimensionality control
- Class separation control
- PCA visualization of data

---

## Experiment 1: Logistic Regression (MNIST-style)

Based on Section 4.1 of the paper.

### Fig 1: Accuracy vs Regularization and Noise

Controls:

- lambda: L2 regularization strength
- sigma: perturbation noise level

Key insight:

- Higher lambda increases stability and allows more removals
- Higher sigma increases privacy budget but reduces accuracy

---

### Fig 2: Gradient Residual Bounds

Compares:

- Worst-case theoretical bound
- Data-dependent bound
- True empirical residual

Key insight:

Worst-case bounds are extremely loose, while data-dependent bounds closely match real behavior.

---

### Fig 3: Removal Difficulty Ranking

Each training point is ranked by:

||H^-1 Δ||

Interpretation:

- High value: hard to remove, strongly influential point
- Low value: easy removal, weak influence

---

## Experiment 2: LSUN Feature Transfer

Based on Section 4.2.

Uses pretrained ResNeXt-101 features with a linear classifier.

Key idea:

Only the final linear classifier requires certified removal. The feature extractor is reused.

Results:

- High accuracy maintained
- Large number of removals supported
- Significant speedup compared to retraining

---

## Experiment 3: SST Sentiment Analysis

Uses RoBERTa embeddings with a linear classifier.

Features:

- Input custom text reviews
- Simulate removal influence on embeddings
- View Newton update magnitude per example

Key insight:

Certified removal applies cleanly to NLP feature spaces as well.

---

## Experiment 4: SVHN with DP Feature Extractor

Based on Section 4.3.

Uses a DP-trained CNN as feature extractor.

Privacy composition:

epsilon_total = epsilon_DP + epsilon_CR

Observations:

- DP-only training significantly reduces accuracy at low epsilon
- Hybrid approach maintains higher accuracy under same budget

---

## Live Removal System (Core Demo)

Implements Algorithm 1 and Algorithm 2.

Workflow:

1. Train model with perturbed loss (Algorithm 1)
2. Select points to remove
3. Apply Newton update:
   w^- = w* + H^-1 Δ
4. Update model instantly
5. Track privacy budget usage

If budget is exceeded, full retraining is triggered.

---

## Audit System

Every removal is logged with:

- Newton update norm ||H^-1 Δ||
- Data-dependent bound contribution
- Worst-case bound comparison
- Cumulative privacy budget usage
- Model accuracy after each removal

This creates a full traceable history of model edits.

---

## Exact vs Approximate Components

### Exact

- Newton update removal
- Hessian-based computations
- Theoretical bounds
- Budget tracking logic

### Approximate

- Synthetic dataset instead of real MNIST
- Precomputed-style behavior for deep feature extractors
- Membership inference visualization (illustrative)

---

## Key Takeaway

This system demonstrates that:

- Data can be removed without retraining
- Removal can be mathematically certified
- Updates reduce to efficient linear algebra operations
- Privacy and accuracy tradeoffs are explicitly controllable

---

## Reference

Guo, C., Goldstein, T., Hannun, A., Van der Maaten, L.  
Certified Data Removal from Machine Learning Models  
ICML 2020

https://arxiv.org/abs/1911.03030
