# Linear Regression

## What it does
Predicts a continuous numerical output by fitting a weighted sum of input
features. The core assumption is that the relationship between inputs and
output is linear — each feature contributes independently and
proportionally to the prediction.

---

## The core idea
Find weights w such that ŷ = Xw is as close to y as possible, where
"close" is measured by Mean Squared Error. Each weight tells you exactly
how much the output changes for a one-unit increase in that feature,
holding everything else constant.

---

## Prediction formula
```
ŷ = Xw + b
```

Using the bias trick (absorbing b into w by prepending a column of 1s):
```
ŷ = X_b @ w
```

---

## Loss function — Mean Squared Error
```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```

The goal of training is to find w that minimizes this.

---

## Method 1 — Normal Equation

Solves for the exact optimal weights directly using calculus.
Derive by taking the derivative of MSE with respect to w, setting
it to zero, and solving:
```
w* = (XᵀX)⁻¹ Xᵀy
```

No iterations. No learning rate. Exact answer in one shot.

**Limitation:** requires XᵀX to be invertible. Fails when features
are perfectly correlated or when you have more features than samples.
Use `np.linalg.pinv` or `np.linalg.lstsq` for robustness.
Complexity is O(p³) for the matrix inverse — slow for high-dimensional data.

---

## Method 2 — Gradient Descent

Iteratively nudges weights in the direction that reduces the loss.

**Gradient of MSE:**
```
∇L(w) = (2/n) × Xᵀ(Xw - y)
```

**Update rule (repeat n_iterations times):**
```
w := w - α × ∇L(w)
```

Where α is the learning rate. Scales to large datasets. Always runs
but gives an approximate answer that improves with more iterations.

---

## Evaluation — R² score
```
R² = 1 - (SS_residual / SS_total)

SS_residual = Σ(yᵢ - ŷᵢ)²
SS_total    = Σ(yᵢ - ȳ)²
```

- R² = 1.0 → perfect predictions
- R² = 0.0 → no better than predicting the mean
- R² < 0.0 → worse than predicting the mean

---

## Assumptions

| Assumption | What breaks if violated |
|---|---|
| Linearity | Model is fundamentally wrong regardless of data size |
| Independence of errors | Common in time series — estimates become unreliable |
| Homoscedasticity (constant error variance) | Confidence intervals become invalid |
| No multicollinearity | Weights become unstable and uninterpretable |
| Normally distributed errors | Statistical tests and p-values become invalid |

---

## Hyperparameters

| Hyperparameter | Method | What it controls | Typical range |
|---|---|---|---|
| learning_rate | GD only | Step size per iteration | 0.0001 – 0.1 |
| n_iterations | GD only | How many update steps | 100 – 10000 |

Normal Equation has no hyperparameters.

---

## Strengths

- Fully interpretable — each weight has a direct, human-readable meaning
- Normal Equation gives an exact closed-form solution
- Fast to train on small to medium datasets
- Well understood statistically — confidence intervals, p-values, hypothesis tests
- Great baseline before trying complex models
- No hyperparameter tuning required (Normal Equation)

---

## Weaknesses

- Assumes linearity — cannot capture curves or interactions natively
- Sensitive to outliers — squared loss heavily penalizes large errors
- Breaks down with multicollinearity
- Normal Equation is O(p³) — slow for high-dimensional data
- No built-in regularization — overfits when features >> samples
- Gradient Descent requires tuning learning rate and iterations

---

## When to use

- Output is a continuous number
- Relationship between features and target is approximately linear
- Interpretability is required
- You need a fast, simple baseline
- Dataset is small to medium sized
- You want statistical inference, not just predictions

## When not to use

- Output is categorical → use logistic regression
- Relationship is clearly nonlinear → use tree-based models or neural nets
- Features are highly correlated → use Ridge or Lasso
- More features than samples → use Ridge or Lasso
- You need to model feature interactions → use tree-based models

---

## Complexity

| | Normal Equation | Gradient Descent |
|---|---|---|
| Training | O(np² + p³) | O(n × p × iterations) |
| Inference | O(np) | O(np) |
| Space | O(np + p²) | O(np) |

n = samples, p = features