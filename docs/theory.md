# Theory behind the examples

The Python specialization defines the applications and training configurations.
Each link below points at runnable code. **Ruby** denotes explicit mathematics;
**library** denotes the original estimator or training engine through PyCall.

## 1. Linear regression — Ruby

For observation `x`, `f(x)=w·x+b`. Batch gradient descent minimizes:

```text
J = Σ_i (f(x_i)-y_i)² / (2m)
dw_j = Σ_i (f(x_i)-y_i)*x_ij / m
db = Σ_i (f(x_i)-y_i) / m
w ← w-α*dw; b ← b-α*db
```

Compute both gradients using the same old parameters. One and multiple features
use the same equations. The executable fits the actual restaurant-profit data.

[Batch regression](../supervised/linear_regression.rb).

## 2. Scaling, polynomial features and library regressors

Fit `μ_j` and population standard deviation `σ_j` on the training observations;
transform using `(x_j-μ_j)/σ_j`. Reuse those statistics for later observations.

Polynomial regression changes the features, not the linear prediction equation.
The course's quadratic/cosine experiments show why powers need scaling. For two
inputs, all monomials through degree six give 27 features plus a separate bias.

OLS stationarity is `(AᵀA)θ=Aᵀy`, with a constant column in `A`. The Python course
uses sklearn `LinearRegression`; this repository retains that estimator rather
than coding an inverse. Its `SGDRegressor` lab also retains the actual estimator,
including source defaults for penalty and convergence. Those defaults are not
silently replaced by a homemade per-example update schedule.

[Ruby scaling](../supervised/feature_scaling.rb),
[Ruby feature engineering](../supervised/polynomial_regression.rb),
[library SGD](../supervised/stochastic_gradient_descent.rb),
[library OLS](../supervised/normal_equation.rb).

## 3. Logistic regression — Ruby, with a separate library lab

`p(y=1|x)=σ(w·x+b)`, where `σ(z)=1/(1+exp(-z))`. Bernoulli cross-entropy:

```text
L(z,y) = softplus(z)-y*z
softplus(z) = max(z,0)+log(1+exp(-abs(z)))
dL/dz = σ(z)-y
```

This logits formulation avoids `log(0)` and exponential overflow. The batch
weight/bias gradients have the linear-regression form with residual `p-y`.
A .5 decision threshold corresponds to `w·x+b=0`. Monomial features make that
boundary nonlinear in the original two-dimensional measurements. The executable
applications are the original admissions and microchip problems.

[Ruby logistic regression](../supervised/logistic_regression.rb),
[Ruby microchip classifier](../supervised/regularized_logistic_regression.rb),
[actual sklearn optional lab](../supervised/sklearn_logistic_regression.rb).

## 4. Regularization conventions

The handwritten linear/logistic assignments add `λ*Σw²/(2m)` and `λ*w/m` to the
cost and weight gradient. Do not penalize the bias. Keras `L2(λ)` instead adds
`λ*Σw²`; the collaborative-filtering assignment uses `λ*Σw²/2` with an
**unnormalized** observed-rating loss. These coefficients are not interchangeable.
The code retains each source convention.

[Explicit penalties](../supervised/regularization.rb).

## 5. Evaluation and bias/variance — Ruby metrics, library fits

Train models on training observations. Use validation error to select degree,
regularization or capacity. Evaluate the selected model on held-out test data.
Compare unpenalized prediction errors, not training objectives containing L2.
The course's regression evaluation uses half-MSE; classification uses the
fraction of wrong labels. Some Keras models train with ordinary MSE: distinguish
the training loss from the reported evaluation convention.

High train/CV error suggests high bias. A low training error with a large CV gap
suggests high variance. More observations cannot give a linear hypothesis the
ability to represent a quadratic. The source sample-size/degree/lambda grids,
noisy-quadratic generator, six-class generator and diagnostic CSVs are retained.

For rare events, `precision=TP/(TP+FP)`, `recall=TP/(TP+FN)` and
`F1=2PR/(P+R)`; undefined divisions return zero. Accuracy can conceal a detector
that predicts no anomalies.

[Metrics](../evaluation/metrics.rb), [selection](../evaluation/model_selection.rb),
[learning curves](../evaluation/learning_curves.rb),
[diagnostic CSVs](../evaluation/diagnosing_bias_variance.rb),
[optional evaluation](../evaluation/optional_model_evaluation.rb),
[neural advice](../evaluation/neural_model_selection.rb).

## 6. Dense layers, activations and original networks

A layer computes `z_j=Σ_k W_jk*a_k+b_j`, followed by an activation. Without a
nonlinearity, stacked affine layers collapse to one affine map. Sigmoid gives a
binary probability. ReLU, `max(0,z)`, introduces hinges; the CLI version retains
the original adjustable three-unit experiment.

For mutually exclusive classes:

```text
p_j = exp(z_j-max(z)) / Σ_k exp(z_k-max(z))
L(z,c) = logsumexp(z)-z_c
dL/dz_j = p_j-1[j=c]
```

A common logit shift preserves probabilities. Keras training uses linear output
logits and `from_logits: true` for multiclass CE; prediction uses argmax.

The original digit models remain 400→25→15→1 and 400→25→15→10. The original
coffee experiment remains 2→3→1 with its data generator, normalization and tiled
training data. Ruby dense inference is compared to actual trained Keras weights.

[Dense inference](../neural_networks/dense_layer.rb),
[ReLU](../neural_networks/relu.rb), [softmax](../neural_networks/softmax.rb),
[binary digits](../neural_networks/binary_classification.rb),
[multiclass digits](../neural_networks/multiclass_classification.rb),
[coffee](../neural_networks/coffee_roasting.rb),
[optional four-class model](../neural_networks/multiclass_blobs.rb).

## 7. Backpropagation and numerical derivatives — Ruby extensions

These expose mathematics underlying the course's training engines; they do not
replace its TensorFlow assignments. Cache activations and logits, then apply:

```text
δ_l = (W_(l+1)ᵀ*δ_(l+1)) .* g'(z_l)
dL/dW_l = δ_l outer_product a_(l-1)
dL/db_l = δ_l
```

Fused sigmoid/BCE and softmax/CE start with output-logit derivative `prediction-target`.
Average per-observation derivatives, then add the appropriate weight-only L2.
Nonidentical random initialization breaks hidden-neuron symmetry.

Central differences `(J(θ+εe_j)-J(θ-εe_j))/(2ε)` independently approximate a
partial derivative. Avoid ReLU kinks and excessively small ε. The default check
samples parameters throughout both original digit architectures on real images;
the checker also accepts an explicit index list or the full parameter set.

[Backpropagation](../neural_networks/backpropagation.rb),
[gradient checks](../neural_networks/gradient_check.rb).

## 8. Adam — explicit mathematics and actual library training

```text
m_t = β1*m_(t-1)+(1-β1)*g_t
v_t = β2*v_(t-1)+(1-β2)*g_t²
m_hat = m_t/(1-β1^t); v_hat = v_t/(1-β2^t)
θ_t = θ_(t-1)-α*m_hat/(sqrt(v_hat)+ε)
```

Bias correction compensates for zero-initialized moments. Optimizer state belongs
to a particular parameter vector, not every model in a process. The Ruby
extension traces moments on the original digit-network gradient. Course training
uses the actual Keras Adam implementation and each notebook's learning rate;
the extension is not a claim of bit-for-bit Keras optimizer compatibility.

[Adam extension](../neural_networks/adam.rb).

## 9. Trees and forests

Entropy is `H=-Σp*log2(p)`. Information gain subtracts weighted child entropy from
parent entropy. The handwritten mushroom assignment evaluates the actual ten
binary-feature observations and recurses on informative splits.

Random forests combine bootstrapped trees and randomized candidate features.
The heart-data lab uses sklearn `DecisionTreeClassifier` and
`RandomForestClassifier`, not the handwritten mushroom learner. Preserve the
categorical one-hot encoding, held-out split and parameter sweeps.

[Ruby tree](../trees/decision_tree.rb), [library forest](../trees/random_forest.rb).

## 10. XGBoost — actual library

For additive logistic score `F(x)`, let `p=σ(F)`, `g=p-y` and `h=p*(1-p)`.
The local second-order tree objective gives leaf weight `-G/(H+λ)` and split gain:

```text
1/2 * [G_L²/(H_L+λ)+G_R²/(H_R+λ)-G²/(H+λ)]-γ
```

Shrinkage adds only a learning-rate fraction of each new tree. Early stopping
monitors held-out loss rather than the training fit. The runnable heart-data
example invokes the real XGBoost engine with the original training setup; these
equations explain that engine, not a substitute restricted boosting implementation.

[XGBoost](../trees/xgboost.rb).

## 11. K-means and image compression — Ruby

Alternate nearest-centroid assignments and per-cluster means to minimize squared
distance. Each nonempty-cluster mean minimizes its assigned-point objective;
initialization affects the local optimum. Empty clusters retain their old center.
Run the original 10 updates even if assignments stabilize earlier.

The first application uses the actual 300 points and three source initial centers.
The second treats the original bird's RGB pixels as 3D observations and learns a
16-color palette. Reconstruction looks up each pixel's assigned centroid. Ideal
palette/index storage is not the same as compressed PNG file size.

[K-means](../unsupervised/k_means.rb), [bird reconstruction](../unsupervised/image_compression.rb).

## 12. Gaussian anomaly detection — Ruby

Fit feature means and population variances on predominantly normal examples.
Assuming independent feature densities:

```text
log p(x) = -1/2 * Σ_j [log(2πσ_j²)+(x_j-μ_j)²/σ_j²]
anomaly if p(x)<epsilon
```

Log densities avoid underflow. The original 1000-step probability-threshold
search maximizes validation F1 on each actual server dataset. Labels choose the
threshold; they do not fit the Gaussian. Independence and Gaussianity remain
model assumptions. A small variance floor handles constant inputs.

[Anomaly detection](../unsupervised/anomaly_detection.rb).

## 13. Collaborative filtering — explicit cost, TensorFlow training

`R_ij` marks observed ratings. Subtract each movie's observed mean and learn movie
factors `x_i`, user factors `w_j` and user biases `b_j`:

```text
e_ij = x_i·w_j+b_j-Ynorm_ij
J = 1/2 * Σ_observed e_ij² + λ/2 * (Σ||x_i||²+Σ||w_j||²)
dx_i = Σ_observed_users e_ij*w_j+λ*x_i
dw_j = Σ_observed_movies e_ij*x_i+λ*w_j
db_j = Σ_observed_movies e_ij
```

A missing rating is not a zero-valued target. Restore means before reporting
ratings and exclude the new user's already-rated movies from recommendations.
The scalar cost is explicit Ruby. The original full matrix trains using
TensorFlow GradientTape/Adam, 100 factors and the notebook's personal ratings.
Unconstrained predictions may exceed the original rating scale.

[Collaborative filtering](../recommenders/collaborative_filtering.rb).

## 14. Content-based recommendation — actual TensorFlow towers

Separate 256→128→32 user/item networks produce embeddings normalized to unit
length. Their dot product predicts a scaled rating; MSE trains both towers.
Squared distance between movie embeddings supports similar-item retrieval,
excluding the same movie itself. Real user/movie metadata supports both new-user
and existing-user recommendations.

The source notebook fits the input StandardScalers before splitting aligned
rows, then fits the target MinMaxScaler on the training targets. This repository
preserves that workflow. Fitting input preprocessing before a held-out split
leaks distribution information; a new experimental protocol should fit those
scalers only on training data rather than silently relabel the source workflow.

[Content-based filtering](../recommenders/content_based_filtering.rb).

## 15. Tabular policy iteration — Ruby

For a known transition model:

```text
Q(s,a) = R(s)+γ*Σ_s' P(s'|s,a)*V(s')
V_policy(s) = Q(s,policy(s))
policy_new(s) = argmax_a Q(s,a)
```

Alternate policy evaluation and greedy improvement until stable. The original
six-state helper uses current-state rewards, terminal values equal to terminal
rewards and zero terminal transition rows. Interior transition rows sum to one,
including stochastic missteps. This is dynamic programming, not sampled DQN.

[Policy iteration](../reinforcement_learning/state_action_values.rb).

## 16. Deep Q-learning — actual LunarLander-v2

Replay samples transitions to reduce adjacent-sample correlation. Epsilon-greedy
behavior explores; a separate target network supplies bootstrapped targets:

```text
y = r + γ*(1-done)*max_a Q_target(s',a)
L = mean((Q_online(s,a)-y)²)
θ_online ← Adam(θ_online, gradient of L)
θ_target ← τ*θ_online+(1-τ)*θ_target
```

Only the selected action's prediction enters MSE. The target-network variables
are not updated by the online optimizer. Both networks remain 8→64→64→4, with
the actual Box2D observations, actions and rewards. Source defaults include
replay 100000, batch 64, gamma .995, Adam .001, updates every four environment
steps, tau .001 and epsilon decay .995/min .01.

Gym .26 separates `terminated` and `truncated`; combining them preserves the
older notebook's `done` mask. This is source behavior, not a claim that masking
time limits is the preferred treatment in a newly designed RL experiment.
The source maximum is 2000 episodes × 1000 steps, with its average-return stop.
Short validation runs exercise learning; they do not prove a solved lander.

[Deep Q-learning](../reinforcement_learning/deep_q_learning.rb).
