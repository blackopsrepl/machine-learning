# Theory behind the demonstrations

The Python Machine Learning Specialization defines the topics. These notes use
small fixtures and consistent Ruby shapes rather than notebook state or estimator
APIs. [Course map](course-map.md) identifies the source notebooks; the runnable
files are the primary explanation, not black-box implementations of these notes.

## Shapes and numerical conventions

- A dataset is an array of `m` rows, each with `n` features; targets have `m` entries.
- Regression weights have length `n`; the intercept is a separate scalar.
- A dense layer stores `weights[output_neuron][input_feature]`, plus one bias per
  output. That orientation is the transpose of some NumPy course examples.
- Standardization uses population variance, dividing by `m`, matching NumPy's
  default. Constant columns use scale one and therefore transform to zero.
- Linear regression and neural squared-error objectives use **half** squared
  error. `NumericTools.mse` reports ordinary MSE. The factor of two changes the
  gradient/learning-rate convention, not the optimum.
- L2 for ordinary regression and neural networks is `λ Σw²/(2m)`. Matrix
  factorization intentionally uses the course's **summed** observed-rating loss
  and `λ Σw²/2`. Neither penalizes biases. XGBoost uses its own tree objective.
- Random initialization and data shuffling receive explicit seeded generators.

## 1. Linear regression and gradient descent

Prediction is `f(x)=w·x+b`. Minimize empirical risk:

```text
J(w,b) = (1 / 2m) Σ_i (f(x_i) - y_i)²
∂J/∂w_j = (1/m) Σ_i (f(x_i)-y_i) x_ij
∂J/∂b   = (1/m) Σ_i (f(x_i)-y_i)
w ← w - α ∇w J       b ← b - α ∂J/∂b
```

Compute both gradients from the old parameters before changing either. On the
shared line, the initial residuals are `[-1,-3,-5,-7]`, so the initial slope and
bias gradients are `-8.5` and `-4`. With `α=0.05`, the first update is `w=0.425`,
`b=0.2`; the demo prints exactly this arithmetic.

Batch GD averages every training row before an update. SGD updates after each
shuffled observation and therefore follows a noisy, order-dependent trajectory.
A decreasing learning rate reduces late oscillations. Vectorized arithmetic and
loops express the same gradient; Ruby arrays keep the operations inspectable.

Files: [batch/multiple regression](../supervised/linear_regression.rb),
[SGD](../supervised/stochastic_gradient_descent.rb).

## 2. Least squares, scaling and engineered features

Adding a column of ones gives a design matrix `A` that includes the bias.
Stationarity yields `(AᵀA)θ=Aᵀy`. The example solves that system with pivoted
elimination; it does not explicitly invert a matrix. Singular inputs are rejected.
This educational normal-equation route is not a numerical replacement for
production QR/SVD least squares: `AᵀA` worsens conditioning.

Unequal feature scales stretch loss contours and can make a useful learning rate
impossible. Compute `z_j=(x_j-μ_j)/σ_j`, using statistics fitted only on training
rows. Transform validation/test rows with those same statistics; fitting again
leaks information and changes the coordinate system.

Polynomial regression is still linear in its **parameters**. Replacing `x` with
`[x,x²,...]` expands the hypothesis space without changing the linear solver.
Two-variable monomials include interactions, not just independent powers. With
degree two, the map is `[x1,x2,x1²,x1*x2,x2²]`. The sixth-degree map has 27 entries
plus the separately held bias. Scaling is especially important after expansion.

Files: [normal equations](../supervised/normal_equation.rb),
[scaling](../supervised/feature_scaling.rb),
[polynomial regression](../supervised/polynomial_regression.rb),
[polynomial classification](../supervised/regularized_logistic_regression.rb).

## 3. Logistic regression and stable cross-entropy

For a binary target, model `p(y=1|x)=σ(z)`, where `z=w·x+b` and
`σ(z)=1/(1+exp(-z))`. The Bernoulli negative log likelihood is:

```text
L(z,y) = -y log σ(z) - (1-y) log(1-σ(z))
       = softplus(z) - y*z
softplus(z) = max(z,0) + log(1+exp(-|z|))
∂L/∂z = σ(z)-y
```

The stable logits form avoids `log(0)` and exponential overflow. A branch-stable
sigmoid avoids computing `exp(1000)`. The remaining weight/bias gradients have
the same residual shape as linear regression, replacing `f-y` with `p-y`.

The 0.5 classification boundary is `w·x+b=0`. Changing the threshold changes the
prediction decision, not the fitted objective. A polynomial feature map makes a
nonlinear input-space boundary linear in expanded feature space; the circle
example demonstrates the need rather than merely plotting an arbitrary curve.

Files: [logistic regression](../supervised/logistic_regression.rb),
[regularized nonlinear classifier](../supervised/regularized_logistic_regression.rb).

## 4. Regularization, model evaluation and bias/variance

L2 adds `λ||w||²/(2m)` to the training objective and `λw/m` to its gradient.
The intercept is excluded: regularizing it would penalize the overall target
level, not only model complexity. The exact ridge stationary equations add `λ`
to the weight diagonal of `AᵀA`, not its intercept entry.

Training error measures fit to data used for learning. Validation chooses degree,
penalty or architecture; test data estimates performance after that choice.
Measure the **unpenalized data error** when comparing train/CV curves. A penalty
is a training preference, not an extra error on a held-out observation.

- High training and validation error suggest a restrictive hypothesis/high bias.
- Low training error and a substantial validation gap suggest high variance.
- More data does not give a line the ability to represent a quadratic.
- Lowest training error need not select the best model: a high-degree interpolant
  fits alternating noise, while the quadratic generalizes better.
- A larger neural network can memorize corrupted labels. Validation, not size or
  confidence, decides which candidate to keep.

Classification error is incorrect predictions divided by the number of examples.
Accuracy alone can obscure rare-event failure. `precision=TP/(TP+FP)`,
`recall=TP/(TP+FN)`, `F1=2PR/(P+R)`; undefined ratios use zero here. The metric demo
shows equal 0.9 accuracy for an all-negative detector and one that finds the only
positive, but their recall/F1 differ.

Files: [L2](../supervised/regularization.rb), [evaluation](../evaluation/metrics.rb),
[selection](../evaluation/model_selection.rb),
[learning curves](../evaluation/learning_curves.rb),
[neural selection](../evaluation/neural_model_selection.rb).

## 5. Dense layers, sigmoid, ReLU and softmax

A layer computes `z_j=Σ_k W_jk*a_k+b_j`, then applies an activation. Without
nonlinear activations, stacked affine layers collapse into a single affine map.
ReLU is `max(0,z)`; two units `ReLU(x)+ReLU(-x)` already construct absolute value.
Sigmoid produces one binary probability.

For `K` mutually exclusive classes, use logits `z_1...z_K`:

```text
p_j = exp(z_j - max(z)) / Σ_k exp(z_k - max(z))
L(z,c) = logsumexp(z) - z_c
∂L/∂z_j = p_j - 1[j=c]
```

Subtracting a common constant preserves probabilities. Keeping loss in logit
space is the same numerical motivation as the course's `from_logits=True`, but
there is no framework here. Prediction uses argmax; integer class labels are
expanded to a one-hot target for the derivative.

The course's binary digit and coffee examples become small sigmoid networks; its
multiclass digit/blob examples become a ReLU/softmax network. These are different
fixtures and widths, not different learning algorithms.

Files: [dense layer](../neural_networks/dense_layer.rb),
[ReLU](../neural_networks/relu.rb), [softmax](../neural_networks/softmax.rb),
[binary](../neural_networks/binary_classification.rb),
[multiclass](../neural_networks/multiclass_classification.rb).

## 6. Backpropagation, initialization and gradient checking

Cache activations and preactivations during the forward pass. For fused
sigmoid/BCE, softmax/CE, or linear/half-squared-error, the output preactivation
delta is `prediction-target`. For a hidden layer:

```text
δ_l = (W_(l+1)ᵀ δ_(l+1)) elementwise g'(z_l)
∂L/∂W_l = δ_l outer_product a_(l-1)
∂L/∂b_l = δ_l
```

Sigmoid derivative is `a*(1-a)`; ReLU derivative is one above zero, zero below.
At its nondifferentiable zero kink this implementation chooses zero. Batch-average
sample gradients and add weight-only L2 terms. `Network#backward` accepts a cached
forward pass and a derivative with respect to output **logits**, allowing composed
objectives to reuse the same chain rule.

Uniform fan-in/fan-out scaled random weights break symmetry. Zero initialization
would make hidden neurons learn identical features. The two-tower example also
keeps its initial output embeddings away from the zero-norm normalization kink.

Central differences `(J(θ+εe_j)-J(θ-εe_j))/(2ε)` independently approximate every
partial. Finite differences balance truncation error against floating-point
roundoff and should avoid activation kinks. The executable check covers every
weight/bias for binary, categorical and regression outputs with L2; the
recommenders independently check their composed gradients too.

Files: [backpropagation](../neural_networks/backpropagation.rb),
[numerical checks](../neural_networks/gradient_check.rb).

## 7. Adam

Adam maintains exponentially decayed first and second gradient moments:

```text
m_t = β1*m_(t-1) + (1-β1)*g_t
v_t = β2*v_(t-1) + (1-β2)*g_t²
m_hat = m_t/(1-β1^t)       v_hat = v_t/(1-β2^t)
θ_t = θ_(t-1) - α*m_hat/(sqrt(v_hat)+ε)
```

Bias correction compensates for moments initialized at zero. Each trainable model
owns its own optimizer state; reusing moments across unrelated models is invalid.
A custom loop is genuinely from scratch here because both derivatives and updates
are visible, unlike a loop delegating them to `GradientTape` and Keras.

File: [Adam](../neural_networks/adam.rb).

## 8. Decision trees and random forests

Entropy `H=-Σp log2(p)` measures label uncertainty. Splitting reduces it by
`H(parent) - n_left/n * H(left) - n_right/n * H(right)`. Candidate thresholds lie
between distinct sorted numeric feature values; a binary feature therefore uses
0.5. Recurse on the greatest positive gain. Pure labels, maximum depth, a
minimum-child-size restriction or no useful split stop the tree. Leaves retain
class counts for majority predictions and probabilities.

A random forest trains trees on **bootstrap samples** (sampling rows with
replacement) and samples candidate features independently at each split. Average
tree leaf probabilities for prediction. Out-of-bag evaluation uses only trees
that did not see a given row in their bootstrap. It is not the training vote and
is noisy on this deliberately tiny dataset.

Files: [tree](../trees/decision_tree.rb), [forest](../trees/random_forest.rb).

## 9. Second-order boosted trees / XGBoost

For binary logistic boosting, keep an additive score `F(x)` and probability
`σ(F(x))`. At each round the first/second derivatives are `g_i=p_i-y_i`,
`h_i=p_i*(1-p_i)`. A tree approximates the loss through these derivatives:

```text
leaf weight = -G/(H+λ)
G = Σ_leaf g_i       H = Σ_leaf h_i
split gain = 1/2 [G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)] - γ
F_new(x) = F_old(x) + learning_rate * tree(x)
```

Here `λ` regularizes leaf weights and `γ` charges an extra split/leaf. Reject
children with too little Hessian mass. Start from the class-prior log odds, add
small trees, and monitor validation loss for early stopping. Restore the best
prefix, not the final worse round. The example checks restoration explicitly.

This is the mathematical core of the course's XGBoost model, implemented by exact
greedy search on tiny data. It is not a compatibility implementation of the
external library, its histogram quantization, missing-value routing, subsampling,
distributed execution or GPU optimizations.

File: [XGBoost](../trees/xgboost.rb).

## 10. K-means and palette compression

Minimize mean squared distance to the assigned centroid. Assignment chooses the
nearest current center; updating a nonempty center to its assigned mean is the
optimal least-squares representative. Alternating these steps cannot increase the
objective, but can converge to a local optimum. Random initialization selects
actual observations; empty clusters retain their existing center here.

RGB pixels are points in three dimensions. Replace them with centroid indices to
quantize the palette; reconstruct pixels by looking up the selected centroid.
The reported storage comparison assumes 8-bit RGB palette entries and ideal
one-bit indices for two colors, excluding file headers. It is a conceptual
encoding calculation, not a generated PNG compression benchmark.

Files: [K-means](../unsupervised/k_means.rb),
[palette reconstruction](../unsupervised/image_compression.rb).

## 11. Gaussian anomaly detection

Fit each feature's mean and population variance on predominantly normal examples.
The diagonal-covariance model assumes feature independence:

```text
log p(x) = -1/2 Σ_j [log(2πσ_j²) + (x_j-μ_j)²/σ_j²]
anomaly if log p(x) < threshold
```

A variance floor handles constant features; log densities avoid multiplying many
small numbers into underflow. Density alone does not choose the anomaly threshold.
Choose it using labeled validation examples and the desired precision/recall
criterion. The example enumerates score boundaries and maximizes validation F1.
Gaussianity/independence are assumptions to check, not guarantees from the model.

File: [anomaly detection](../unsupervised/anomaly_detection.rb).

## 12. Collaborative filtering

Ratings are a sparse movie-by-user matrix. Let `R_ij` indicate an observed rating.
Subtract each movie's observed mean; jointly learn movie factors `x_i`, user
factors `w_j` and user biases `b_j`:

```text
e_ij = x_i·w_j + b_j - Ynorm_ij
J = 1/2 Σ_observed e_ij² + λ/2 (Σ||x_i||² + Σ||w_j||²)
dx_i = Σ_observed_users e_ij*w_j + λ*x_i
dw_j = Σ_observed_movies e_ij*x_i + λ*w_j
db_j = Σ_observed_movies e_ij
```

A missing rating contributes nothing. It is not a dislike or a zero target.
Restore movie means before reporting ratings; rank only unseen items. Low observed
training error does not establish accuracy on missing cells, and unconstrained
factorization can predict outside the original rating range.

The Python notebook delegates gradients to TensorFlow; this version supplies the
analytic gradients and verifies every parameter independently.

File: [collaborative filtering](../recommenders/collaborative_filtering.rb).

## 13. Neural content-based recommendation

Two separate networks map user features and movie features into a shared embedding
space. Normalize both embeddings and score their dot product. Learn by rating
error; the toy genre features provide a known cosine target. For raw vector `u`,
normalized vector `u_hat`, counterpart `v_hat`, and score `s`:

```text
s = u_hat·v_hat
dL/du = (s-y) * (v_hat - s*u_hat) / ||u||
```

The other tower has the symmetric derivative. Backpropagate both derivatives
through the visible dense-network code. A small norm floor has its own linear
clamped-branch derivative; initialization avoids its kink during numerical checks.
Squared distance between learned normalized movie vectors supplies item-to-item
similarity; self matches must be excluded when selecting a different similar item.

The toy inputs already occupy a small common scale. With real course data, split
aligned user/item/target rows before fitting input/target scalers; do not fit
preprocessing on held-out observations.

File: [two-tower model](../recommenders/content_based_filtering.rb).

## 14. Tabular policy iteration

An MDP specifies states, actions, rewards, transition probabilities and discount
`γ`. For a fixed policy, repeatedly apply its Bellman expectation backup until
values stabilize; greedily improve its actions; repeat until the policy stabilizes.

```text
Q(s,a) = R(s) + γ Σ_s' P(s'|s,a) V(s')
V_policy(s) = Q(s, policy(s))
policy_new(s) = argmax_a Q(s,a)
```

This example follows the course helper's current-state reward convention:
terminal values equal their rewards and terminal transition rows are zero.
Interior stochastic-misstep rows sum to one. Check the final values against
`V(s)=max_a Q(s,a)`. This is dynamic programming with a known transition model,
not a sampled or neural Q-learning algorithm.

File: [state–action values](../reinforcement_learning/state_action_values.rb).

## 15. Deep Q-learning

DQN learns a Q function from transitions rather than using a known transition
matrix. Sample experiences from a bounded replay buffer to reduce adjacent-sample
correlation. Select random actions with probability `ε`; otherwise choose the
largest online Q value. Form detached targets from a separate target network:

```text
y = r                               if genuinely terminal
y = r + γ max_a Q_target(s',a)       otherwise
L = 1/2 (Q_online(s,a)-y)²
θ_online ← Adam(θ_online, gradient of L)
θ_target ← (1-τ)*θ_target + τ*θ_online
```

Only the selected action contributes a delta; the other action targets equal their
current online predictions. Target arrays are values, not differentiable nodes.
A time-limit truncation does not silently become terminal. Soft target updates
happen after replay minibatch optimization, as in the Python course helper.

The course uses mean squared error; this laboratory uses half squared error,
consistent with its visible regression/backprop convention. Network size, discount
and training budget are adapted to the tiny problem, not claimed as LunarLander
hyperparameters. The environment is a one-dimensional continuous corridor,
**not** a physics port. Its two monotone terminal paths have exact discounted
returns, giving a numerical oracle independent of the learned network.

File: [DQN](../reinforcement_learning/deep_q_learning.rb).
