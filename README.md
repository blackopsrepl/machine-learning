# machine-learning

Andrew Ng's **Python Machine Learning Specialization**, each algorithm in a
readable, runnable Ruby file. Same format as [heuristics](https://github.com/blackopsrepl/heuristics):
pure Ruby, zero dependencies — no gems, native extensions, Python or frameworks.
Every demo prints the decisions or numerical updates that explain its mechanism.

The course supplies the algorithm list; it does not dictate the implementation
language. TensorFlow, scikit-learn and XGBoost calls are replaced with explicit
learning code, including backpropagation, Adam, forests, boosting and DQN.

## The shared toy problems

**Regression** — four points exactly on `y = 1 + 2x`. Batch GD, SGD and ordinary
least squares must agree. A second feature checks that the same gradient extends
to multiple regression; `x²` exposes the difference between features and models.

**Classification** — separable points, a circle, XOR and three small clusters.
The line's failure on XOR and the circle is intentional: nonlinear features and
hidden units have a reason to exist. Noisy labels expose overfitting during model
selection. The tree examples share the course's tiny binary mushroom table.

**Unsupervised learning** — two point clouds, eight RGB pixels and a small server
monitoring table. Every centroid, palette index and anomaly score fits on screen.

**Recommendation** — four movies, three users and explicit missing ratings.
Genre vectors provide an independent content-based example. Missing values never
become zero-valued training labels.

**Reinforcement learning** — the course's six-state world and a continuous-state
corridor with unequal terminal rewards. A larger reward farther away can lose to
a smaller nearby reward after discounting. The DQN must learn that choice.

## Run

Requires Ruby **3.1 or newer**. GNU Make supplies the menu; every file also runs
directly without Make. No downloads, environment setup or training assets.

```sh
make                          # this menu
make all                      # every algorithm
make supervised               # one family
make neural_networks
make evaluation
make trees
make unsupervised
make recommenders
make reinforcement_learning
make backpropagation          # one algorithm
ruby neural_networks/backpropagation.rb
make check                    # Ruby warnings/syntax + every executable assertion
```

## The matrix

| File | Idea | Outcome on the toy instance |
|---|---|---|
| [`supervised/linear_regression.rb`](supervised/linear_regression.rb) | batch gradients, one or multiple features | recovers y=1+2x and the two-feature coefficients |
| [`supervised/stochastic_gradient_descent.rb`](supervised/stochastic_gradient_descent.rb) | shuffle; update after each example | recovers the same line with sequential updates |
| [`supervised/normal_equation.rb`](supervised/normal_equation.rb) | solve the least-squares stationary equations | exact coefficients; rejects a singular system |
| [`supervised/feature_scaling.rb`](supervised/feature_scaling.rb) | fit mean/std on training rows | the same learning rate diverges raw and converges scaled |
| [`supervised/polynomial_regression.rb`](supervised/polynomial_regression.rb) | change the basis, not the solver | quadratic fits x²; a straight line cannot |
| [`supervised/logistic_regression.rb`](supervised/logistic_regression.rb) | stable BCE and explicit gradients | separates the binary points; extreme logits stay finite |
| [`supervised/regularization.rb`](supervised/regularization.rb) | penalize weights, exclude the bias | shrinks the slope while preserving a constant intercept |
| [`supervised/regularized_logistic_regression.rb`](supervised/regularized_logistic_regression.rb) | map two inputs to polynomial monomials | quadratic boundary classifies the circle; a line misses it |
| [`neural_networks/dense_layer.rb`](neural_networks/dense_layer.rb) | dot products, biases, activation | each neuron prints its affine calculation |
| [`neural_networks/relu.rb`](neural_networks/relu.rb) | nonlinear hinges | two ReLU units construct absolute value |
| [`neural_networks/softmax.rb`](neural_networks/softmax.rb) | subtract max; compute logsumexp loss | adding 1,000 to logits changes no probabilities |
| [`neural_networks/adam.rb`](neural_networks/adam.rb) | first/second moments and bias correction | minimizes (θ−3)² with explicit updates |
| [`neural_networks/backpropagation.rb`](neural_networks/backpropagation.rb) | cache activations; propagate deltas | manual gradients train an XOR network |
| [`neural_networks/binary_classification.rb`](neural_networks/binary_classification.rb) | sigmoid network versus linear model | network solves XOR; linear baseline scores 0.5 |
| [`neural_networks/multiclass_classification.rb`](neural_networks/multiclass_classification.rb) | ReLU, logits and categorical loss | learns all three labeled clusters |
| [`neural_networks/gradient_check.rb`](neural_networks/gradient_check.rb) | central differences for every parameter | BCE, categorical CE, regression and L2 agree numerically |
| [`evaluation/metrics.rb`](evaluation/metrics.rb) | accuracy, precision, recall and F1 | equal accuracy conceals very different anomaly recall |
| [`evaluation/model_selection.rb`](evaluation/model_selection.rb) | degree/λ selection using validation | selects degree 2, not noisy degree-5 interpolation |
| [`evaluation/learning_curves.rb`](evaluation/learning_curves.rb) | refit as training size grows | more data cannot repair a wrong linear hypothesis |
| [`evaluation/neural_model_selection.rb`](evaluation/neural_model_selection.rb) | simple, complex and L2 networks | best training fit is not best validation error |
| [`trees/decision_tree.rb`](trees/decision_tree.rb) | entropy and information gain | prints candidate splits; classifies all toy mushrooms |
| [`trees/random_forest.rb`](trees/random_forest.rb) | bootstrap rows; random features per node | probability voting plus genuinely out-of-bag evaluation |
| [`trees/xgboost.rb`](trees/xgboost.rb) | gradient/Hessian trees and shrinkage | fits nonlinear labels; early stopping restores the best prefix |
| [`unsupervised/k_means.rb`](unsupervised/k_means.rb) | alternate assignment and mean updates | separates two clouds with nonincreasing distortion |
| [`unsupervised/image_compression.rb`](unsupervised/image_compression.rb) | cluster RGB values into a palette | eight inline pixels reconstruct through two colors |
| [`unsupervised/anomaly_detection.rb`](unsupervised/anomaly_detection.rb) | Gaussian log density; validation threshold | finds both outliers with validation F1=1 |
| [`recommenders/collaborative_filtering.rb`](recommenders/collaborative_filtering.rb) | masked factorization with user biases | fits observed ratings; ranks missing cells without training on them |
| [`recommenders/content_based_filtering.rb`](recommenders/content_based_filtering.rb) | learn normalized user/item towers | genre-aligned ranking and meaningful item distances |
| [`reinforcement_learning/state_action_values.rb`](reinforcement_learning/state_action_values.rb) | evaluate/improve a six-state policy | Bellman optimality holds with and without stochastic missteps |
| [`reinforcement_learning/deep_q_learning.rb`](reinforcement_learning/deep_q_learning.rb) | replay, target network and explicit DQN gradients | learned returns match the exact corridor oracle at all five demo starts |

The failure cases carry the explanation: raw scaling diverges, a line misses
curvature, a linear classifier misses XOR, noisy interpolation loses to the right
basis, and accuracy alone hides a rare-event detector that finds nothing.

## Layout

```text
lib/style.rb             the same ANSI vocabulary as heuristics
lib/problem.rb           shared tiny fixtures, returned as fresh arrays
lib/numeric.rb           dot products, statistics, scaling, small linear solves
supervised/              regression, classification and feature construction
neural_networks/         layers, activations, Adam and manual backpropagation
evaluation/              metrics, held-out selection and bias/variance examples
trees/                   entropy trees, bootstrap forest and second-order boosting
unsupervised/            clustering, palette reconstruction and anomaly detection
recommenders/            masked factorization and normalized two-tower training
reinforcement_learning/  policy iteration and replay-based DQN
docs/theory.md           equations, derivations and numerical conventions
docs/course-map.md       every current Python notebook mapped to these examples
```

Algorithm modules/classes can be required without running their terminal demos.
Learning code is not buried in `lib/`: composed examples require the algorithm
file they reuse. The two-tower model and DQN reuse the visible backpropagation
implementation rather than containing alternate hidden neural trainers.

## Determinism and numerical checks

Randomized demos own a `Random.new(42)` and identify the seed. `make check` runs
real demonstrations with numerical assertions, not mocks. The neural, factorization
and two-tower examples check analytic gradients with finite differences. DQN
compares actual rollout returns against an independent exact-path oracle.

All demonstrations are bounded teaching workloads. Printed training accuracy is
not evidence of generalization: the evaluation examples keep train, validation
and test observations separate. Theory and individual headers state the objective,
normalization and stopping conventions used by each implementation.

## Course scope and deliberate substitutions

This follows the Python specialization, **not** the older Octave course. No
Octave-only SVM, PCA or spam-classifier exercises are added. Python/Jupyter
onboarding and NumPy indexing are language tooling, not additional algorithms.

- Images and large tabular datasets become tiny inline fixtures; the mushroom
  example retains the assignment's ten-row binary teaching table. No pretrained
  models or downloaded course assets are required.
- Neural widths shrink while the forward/backward mathematics stays intact.
- `trees/xgboost.rb` implements exact-greedy, second-order regularized boosting,
  not the external XGBoost engine's histogram, sparse, distributed or GPU systems.
- DQN uses the explicit continuous corridor instead of `LunarLander-v2`/Box2D.
  Replay, epsilon-greedy exploration, selected-action targets, terminal masking,
  Adam and soft target updates are implemented; Lunar Lander physics is not.
- Plots and notebook widgets become inspectable terminal arithmetic. This is an
  educational laboratory, not a replacement for production ML libraries.

## Sources

[Machine Learning Specialization — DeepLearning.AI / Stanford Online](https://www.coursera.org/specializations/machine-learning-introduction)
by Andrew Ng and the course team. [Course map](docs/course-map.md) identifies the
local Python notebook sources and distinguishes reimplementations from extracted
handwritten mathematics. The explanations and Ruby implementations here are
written for this laboratory; this repository does not redistribute notebooks,
lecture PDFs or large course datasets.
