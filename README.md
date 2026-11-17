# machine-learning

Andrew Ng's **Python Machine Learning Specialization**, in the terminal-first
Ruby format of [heuristics](https://github.com/blackopsrepl/heuristics).
The course supplies the problems, dataset shapes, architectures and training
settings; the terminal replaces notebook plots and widgets.

Everything is Ruby. What the course writes out in NumPy is handwritten here.
What it takes from scikit-learn, Keras, XGBoost and Gym/Box2D is ported from
those libraries' own algorithms and defaults, including their seeded random
streams, so course seeds give the course's samples, splits and terrain. No
Python, no gems, no downloads.

## Run

Tested with Ruby 3.4. GNU Make provides the menu.

```sh
make                             # menu
make linear_regression           # one example
make supervised                  # one course family
make handwritten                 # what the course writes out itself
make ported                      # what the course takes from a library
make check                       # checks, syntax, then every example
```

Each file also runs directly:

```sh
ruby unsupervised/k_means.rb
ruby reinforcement_learning/lunar_lander.rb
ruby reinforcement_learning/deep_q_learning.rb
ruby reinforcement_learning/deep_q_learning.rb --evaluate output/lunar_lander.json --episodes 5
ruby neural_networks/relu.rb --w1 1 --b1 -1 --w2 2 --b2 -4
```

Every example uses the course's full budgets: the polynomial labs run up to
1,000,000 iterations, the neural evaluation sweeps their full epochs, and deep
Q-learning up to 2000 episodes of 1000 steps. `make check` and `make all` take
hours; Lunar Lander training dominates. Trained Q-network weights go to the
ignored `output/lunar_lander.json`.

## Data

[`data/`](data/README.md) ships one directory per course dataset, under the
course's file names, formats, dtypes and shapes. `heart.csv` is the original
(ODbL). The other course files are not redistributable, so
[`data/generate.rb`](data/generate.rb) produces deterministic stand-ins in the
same shapes. To run on the course's own files:

```sh
ruby data/import_course.rb /path/to/MachineLearningSpecialist /path/to/course-data
ML_DATA=/path/to/course-data make check
```

## Implementations

| File | Kind | Application |
|---|---|---|
| [`supervised/feature_scaling.rb`](supervised/feature_scaling.rb) | handwritten | four-feature housing z-score normalization |
| [`supervised/linear_regression.rb`](supervised/linear_regression.rb) | handwritten | restaurant profits; batch gradient descent |
| [`supervised/logistic_regression.rb`](supervised/logistic_regression.rb) | handwritten | exam scores → admission |
| [`supervised/normal_equation.rb`](supervised/normal_equation.rb) | ported | scikit-learn `LinearRegression` on housing |
| [`supervised/polynomial_regression.rb`](supervised/polynomial_regression.rb) | handwritten | polynomial and cosine feature engineering |
| [`supervised/regularization.rb`](supervised/regularization.rb) | handwritten | L2 for linear and logistic regression |
| [`supervised/regularized_logistic_regression.rb`](supervised/regularized_logistic_regression.rb) | handwritten | microchip tests; degree-six features |
| [`supervised/sklearn_logistic_regression.rb`](supervised/sklearn_logistic_regression.rb) | ported | scikit-learn `LogisticRegression` lab |
| [`supervised/stochastic_gradient_descent.rb`](supervised/stochastic_gradient_descent.rb) | ported | scikit-learn `SGDRegressor` on housing |
| [`neural_networks/sequential.rb`](neural_networks/sequential.rb) | ported | Keras `Sequential`: dense layers, losses, backprop, `fit` |
| [`neural_networks/adam.rb`](neural_networks/adam.rb) | ported | Keras `Adam` |
| [`neural_networks/gradient_check.rb`](neural_networks/gradient_check.rb) | check | finite differences against the training engine |
| [`neural_networks/binary_classification.rb`](neural_networks/binary_classification.rb) | ported | digits 0/1; 400→25→15→1 |
| [`neural_networks/multiclass_classification.rb`](neural_networks/multiclass_classification.rb) | ported | digits 0–9; 400→25→15→10 logits |
| [`neural_networks/coffee_roasting.rb`](neural_networks/coffee_roasting.rb) | ported | roast data, normalization, 2→3→1 |
| [`neural_networks/multiclass_blobs.rb`](neural_networks/multiclass_blobs.rb) | ported | `make_blobs` four classes; 2→2→4 |
| [`neural_networks/dense_layer.rb`](neural_networks/dense_layer.rb) | handwritten | NumPy `my_dense` / `my_sequential` inference |
| [`neural_networks/relu.rb`](neural_networks/relu.rb) | handwritten | three-hinge ReLU model; flags replace sliders |
| [`neural_networks/softmax.rb`](neural_networks/softmax.rb) | handwritten | stable softmax and logit loss |
| [`evaluation/metrics.rb`](evaluation/metrics.rb) | handwritten | half-MSE and classification error |
| [`evaluation/model_selection.rb`](evaluation/model_selection.rb) | ported | degree and λ chosen on validation data |
| [`evaluation/learning_curves.rb`](evaluation/learning_curves.rb) | ported | degree 16, 50 → 750 examples |
| [`evaluation/diagnosing_bias_variance.rb`](evaluation/diagnosing_bias_variance.rb) | ported | the four diagnostic CSVs |
| [`evaluation/neural_model_selection.rb`](evaluation/neural_model_selection.rb) | ported | six-class networks and L2 sweep |
| [`evaluation/optional_model_evaluation.rb`](evaluation/optional_model_evaluation.rb) | ported | polynomial and neural candidates on both CSVs |
| [`trees/decision_tree.rb`](trees/decision_tree.rb) | handwritten | mushrooms; entropy and information gain |
| [`trees/cart.rb`](trees/cart.rb) | ported | scikit-learn `DecisionTreeClassifier` |
| [`trees/random_forest.rb`](trees/random_forest.rb) | ported | tree and forest sweeps on heart.csv |
| [`trees/xgboost.rb`](trees/xgboost.rb) | ported | XGBoost with early stopping on heart.csv |
| [`unsupervised/k_means.rb`](unsupervised/k_means.rb) | handwritten | 300 points, three clusters |
| [`unsupervised/image_compression.rb`](unsupervised/image_compression.rb) | handwritten | bird image to 16 colors |
| [`unsupervised/anomaly_detection.rb`](unsupervised/anomaly_detection.rb) | handwritten | Gaussian densities, F1-chosen threshold |
| [`recommenders/collaborative_filtering.rb`](recommenders/collaborative_filtering.rb) | ported | 4778×443 ratings, 100 features, Adam |
| [`recommenders/content_based_filtering.rb`](recommenders/content_based_filtering.rb) | ported | 256→128→32 user and movie towers |
| [`reinforcement_learning/state_action_values.rb`](reinforcement_learning/state_action_values.rb) | handwritten | six-state Bellman values and policy |
| [`reinforcement_learning/lunar_lander.rb`](reinforcement_learning/lunar_lander.rb) | ported | Gym LunarLander-v2 on a Box2D port |
| [`reinforcement_learning/deep_q_learning.rb`](reinforcement_learning/deep_q_learning.rb) | ported | DQN: replay, target network, ε-greedy |

Shared code in [`lib/`](lib): NumPy's `RandomState` (MT19937) and `Generator`
(PCG64) streams and CPython's `random` ([`numpy_random.rb`](lib/numpy_random.rb)),
scikit-learn's `make_blobs` and `train_test_split`
([`sklearn_data.rb`](lib/sklearn_data.rb)), the heart.csv one-hot encoding
([`heart_data.rb`](lib/heart_data.rb)), the Box2D port ([`box2d.rb`](lib/box2d.rb)),
readers for the course's text, CSV, `.npy` and PNG files, and the terminal style.

## Model and training settings

| Application | Course configuration |
|---|---|
| Restaurant profits | 97 observations; batch GD, α .01, 1500 iterations |
| Admissions / microchips | explicit logistic gradients; degree-six map and L2 for microchips |
| Binary digits | 1000 × 400 inputs; 400→25→15→1 sigmoid; BCE; Adam .001; 20 epochs |
| Multiclass digits | 5000 × 400 inputs; 400→25→15→10, ReLU hidden layers, linear logits; sparse CE; Adam .001; 40 epochs |
| Coffee roasting | `default_rng(2)` data; normalization; 1000-fold tiling; 2→3→1 sigmoid; Adam .01; 10 epochs |
| Optional multiclass | 100 points, four centers; 2→2→4; Adam .01; 200 epochs |
| Collaborative movies | 4778 × 443 ratings plus the notebook's ratings column; mean-normalized masked loss; 100 features; Adam .1; λ 1; 200 steps |
| Content-based movies | scaling then 80/20 split; 14/16 inputs; 256→128→32 normalized towers; Adam .01; 30 epochs |
| Heart classifiers | one-hot features, 80/20 split seed 55; tree and forest sweeps; XGBoost 500 rounds, lr .1, early stopping 50 |
| K-means / image | 300 points, three fixed initial centers; 128×128 image, 16 colors; 10 iterations each |
| Anomalies | two server datasets; Gaussian fit and validation-F1 threshold search |
| Evaluation | noisy-quadratic generator, six-class blobs, four diagnostic CSVs, both optional CSVs; course degree, λ and model-size grids |
| Lunar Lander | 8→64→64→4 online and target nets; replay 100000; batch 64; γ .995; Adam .001; update every 4 steps; τ .001; ε decay .995 to .01; up to 2000 × 1000 |

## Validation

`make checks` compares the Ruby random streams with values printed by NumPy
1.26 and CPython 3.11 and checks every dataset's file names, dtypes and shapes.
`make check` adds a syntax pass and runs every example; examples assert their
own invariants (decreasing costs, gradient checks, early-stopping bookkeeping,
the Gym controller landing).

The ports were compared with the libraries during development. `SGDRegressor`
stops at the same epoch with the same coefficients. `DecisionTreeClassifier`,
`RandomForestClassifier` and XGBoost give the same accuracies, leaf counts, best
iteration and probabilities on heart.csv. The Keras engine tracks TensorFlow
float64 training to about 1e-7. Gym's heuristic controller scores within about
one point of Gym on the same Lunar Lander seeds.

See [`docs/course-map.md`](docs/course-map.md) for notebook provenance and
[`docs/theory.md`](docs/theory.md) for the equations.

## Scope and sources

This is the Python specialization, not the older Octave course: no SVM, PCA or
spam classifier. Introductory notebook sections share the named algorithm
kernels; notebook graphics and widgets are not copied.

[Machine Learning Specialization — DeepLearning.AI / Stanford Online](https://www.coursera.org/specializations/machine-learning-introduction),
Andrew Ng and the course team. The code and concise theory are written for this
repository.
