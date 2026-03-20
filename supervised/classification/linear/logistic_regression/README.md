# Logistic Regression

## What it does
Predicts the probability that a sample belongs to a class, then thresholds
that probability to make a binary class decision (0 or 1). Despite the
name it is a classification model, not a regression model.

---

## The core idea
Take the same linear combination as linear regression, then squash the
output through a sigmoid function to force it into the range (0, 1).
The result is interpreted as a probability. A threshold (default 0.5)
converts that probability into a class label.

---

## Prediction

Step 1 — linear combination with bias trick (prepend column of 1s to X):
```
z = X_b @ w
```

Step 2 — squash through sigmoid:
```
ŷ = σ(z) = 1 / (1 + e⁻ᶻ)
```

Step 3 — threshold:
```
class = 1 if ŷ >= 0.5 else 0
```

---

## The sigmoid function

Maps any real number to (0, 1):
```
σ(z) = 1 / (1 + e⁻ᶻ)
```

- z = 0   → σ = 0.5
- z → +∞  → σ → 1.0
- z → -∞  → σ → 0.0

Derivative (used in backprop later):
```
σ'(z) = σ(z) × (1 - σ(z))
```

---

## Why not MSE as the loss

Composing MSE with sigmoid creates a non-convex loss surface full of
local minima. Instead logistic regression uses Binary Cross-Entropy
(Log Loss), which is convex — gradient descent always finds the global
minimum.

---

## Loss function — Binary Cross-Entropy
```
L = -(1/n) × Σ [yᵢ × log(ŷᵢ) + (1 - yᵢ) × log(1 - ŷᵢ)]
```

Intuition:
- When y = 1: loss = -log(ŷ). High predicted probability → low loss.
- When y = 0: loss = -log(1 - ŷ). Low predicted probability → low loss.
- Being confidently wrong is penalized heavily.

`1e-15` is added inside the log to prevent log(0) = -∞.

---

## Gradient

Despite sigmoid and log being involved, the gradient works out cleanly:
```
∇L(w) = (1/n) × X_bᵀ @ (ŷ - y)
```

Same structure as linear regression's gradient. Cross-entropy and sigmoid
are designed to complement each other — the messy terms cancel out.

---

## Update rule
```
w := w - α × ∇L(w)
```

Repeated n_iterations times. No closed-form solution exists — the sigmoid
makes the loss non-linear in w so gradient descent is the only approach.

---

## Decision boundary

The boundary where the model outputs exactly 0.5 is where z = 0:
```
X_b @ w = 0
```

This is a linear equation — the decision boundary is always a straight
line (2D), plane (3D), or hyperplane (higher dimensions). Logistic
regression can only separate linearly separable classes.

---

## Weight interpretation — log-odds

The model learns log-odds (logit):
```
log(p / (1 - p)) = X_b @ w
```

Each weight wᵢ means: a one-unit increase in feature i multiplies the
odds of the positive class by e^wᵢ, holding everything else constant.
This makes logistic regression highly interpretable.

---

## Assumptions

| Assumption | What breaks if violated |
|---|---|
| Linear decision boundary | Model cannot fit the data regardless of size |
| Independence of samples | Coefficient estimates become unreliable |
| No multicollinearity | Weights become unstable and uninterpretable |
| No extreme outliers | Decision boundary gets distorted |
| Large enough sample size | Maximum likelihood estimates become unreliable |

---

## Hyperparameters

| Hyperparameter | What it controls | Typical range |
|---|---|---|
| learning_rate | Step size per gradient update | 0.0001 – 0.1 |
| n_iterations | Number of gradient descent steps | 100 – 10000 |

---

## Strengths

- Outputs a calibrated probability, not just a class label
- Convex loss — gradient descent always finds the global minimum
- Highly interpretable via log-odds
- Fast to train and predict
- Strong statistical foundation
- Threshold is adjustable for different precision/recall tradeoffs

## Weaknesses

- Linear decision boundary only — fails on nonlinear problems
- Sensitive to multicollinearity
- Requires feature scaling for gradient descent to behave well
- No closed-form solution — must use iterative optimization
- Underfits complex datasets

---

## When to use

- Binary classification problems
- Interpretability is required
- Classes are approximately linearly separable
- Probability outputs are needed, not just class labels
- You need a fast reliable classification baseline

## When not to use

- Nonlinear decision boundary → use tree-based models or kernel SVM
- Complex feature interactions → use gradient boosting
- Very high dimensional sparse data → use Naive Bayes or SGD classifier

---

## Complexity

| | Value |
|---|---|
| Training | O(n × p × iterations) |
| Inference | O(np) |
| Space | O(p) |

n = samples, p = features