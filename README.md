# machine-learning

Andrew Ng's **Python Machine Learning Specialization**, in the terminal-first
Ruby format of [heuristics](https://github.com/blackopsrepl/heuristics).
The course supplies the problems, data, architectures and training settings;
the terminal replaces notebook plots and widgets.

Handwritten course mathematics stays in Ruby. Course examples that use
TensorFlow, scikit-learn or XGBoost call those actual libraries through PyCall.
They are not replaced by reduced Ruby estimators.

## Run

Tested with Ruby 3.4 and an isolated Python 3.11 runtime. GNU Make provides the
menu. The library track additionally requires uv, Bundler, and the C/C++/SWIG
build tools used by PyCall and Box2D. No system Python packages are installed.

```sh
make                             # menu
make data-check                  # verify the original assets are present
make scratch                     # handwritten course examples
make setup-reference             # isolated Python libraries and Ruby bridge
make binary_classification       # real digit data, original binary architecture
make content_based_filtering     # original movie and user feature tables
make deep_q_learning             # real LunarLander-v2, full course budget
make check                       # execute every entry point, with stated bounds
```

Each file also runs directly, for example:

```sh
ruby unsupervised/k_means.rb
ruby reinforcement_learning/deep_q_learning.rb --episodes 2 --steps 1000
ruby neural_networks/relu.rb --w1 1 --b1 -1 --w2 2 --b2 -4
```

`make supervised`, `make neural_networks`, `make evaluation`, `make trees`,
`make unsupervised`, `make recommenders` and `make reinforcement_learning`
run a family. `make libraries`, `make mixed` and `make extensions` select an
implementation track. `make all` uses the complete source training budgets;
Lunar Lander training can take a long time.

## Original data

Assets are read from the existing course directory, not replaced with generated
stand-ins and not copied into Git:

```text
/srv/res/computing/DataEng/MachineLearningSpecialist
```

To use the same course tree from a different location:

```sh
export ML_COURSE_ROOT=/path/to/MachineLearningSpecialist
make data-check
```

[`data/sources.json`](data/sources.json) records the source subdirectories.
[`lib/course_data.rb`](lib/course_data.rb) reads numeric text and NumPy arrays;
[`lib/png_image.rb`](lib/png_image.rb) reads the original bird PNG. The binary
assignment's first-1000-row selection follows its loader, not a new sampling rule.
The small mushroom table and generated coffee/blob data are themselves original
course examples.

The library dependencies are pinned in
[`requirements-reference.txt`](requirements-reference.txt) and
[`Gemfile.lock`](Gemfile.lock). Setup writes only `.reference/` and `.bundle/`.
Compatible Gym/Box2D versions retain `LunarLander-v2`; changing the simulator
version is not a transparent dependency upgrade.

## Implementations

- **Ruby:** explicit course mathematics; no TensorFlow, sklearn or XGBoost runtime.
- **Library:** the actual course estimator/optimizer/runtime through PyCall.
- **Mixed:** explicit Ruby inference compared with library-trained digit weights.
- **Ruby extension:** additional inspectable Adam/backpropagation/derivative code.
  These use real digit inputs and course-sized networks; they do not stand in
  for the original TensorFlow training assignments.

| File | Track | Application |
|---|---|---|
| [`supervised/feature_scaling.rb`](supervised/feature_scaling.rb) | Ruby | original four-feature housing normalization |
| [`supervised/linear_regression.rb`](supervised/linear_regression.rb) | Ruby | restaurant profits; handwritten batch GD |
| [`supervised/logistic_regression.rb`](supervised/logistic_regression.rb) | Ruby | exam-score admissions; handwritten gradients |
| [`supervised/normal_equation.rb`](supervised/normal_equation.rb) | Library | housing; actual sklearn LinearRegression |
| [`supervised/polynomial_regression.rb`](supervised/polynomial_regression.rb) | Ruby | original polynomial/cosine feature experiments |
| [`supervised/regularization.rb`](supervised/regularization.rb) | Ruby | linear/logistic L2 on original observations |
| [`supervised/regularized_logistic_regression.rb`](supervised/regularized_logistic_regression.rb) | Ruby | microchip tests; degree-six features |
| [`supervised/sklearn_logistic_regression.rb`](supervised/sklearn_logistic_regression.rb) | Library | original optional sklearn classifier |
| [`supervised/stochastic_gradient_descent.rb`](supervised/stochastic_gradient_descent.rb) | Library | housing; actual sklearn SGDRegressor |
| [`neural_networks/adam.rb`](neural_networks/adam.rb) | Ruby extension | handwritten optimizer; course-sized digit gradients |
| [`neural_networks/backpropagation.rb`](neural_networks/backpropagation.rb) | Ruby extension | handwritten derivative extension; original digit dimensions |
| [`neural_networks/binary_classification.rb`](neural_networks/binary_classification.rb) | Library | 1000 digit images; 400-25-15-1 Keras model |
| [`neural_networks/coffee_roasting.rb`](neural_networks/coffee_roasting.rb) | Library | original roasting generator, model and NumPy-style inference |
| [`neural_networks/dense_layer.rb`](neural_networks/dense_layer.rb) | Mixed | handwritten inference on original digit pixels/weights |
| [`neural_networks/gradient_check.rb`](neural_networks/gradient_check.rb) | Ruby extension | finite differences on original image inputs |
| [`neural_networks/multiclass_blobs.rb`](neural_networks/multiclass_blobs.rb) | Library | original four-class optional network |
| [`neural_networks/multiclass_classification.rb`](neural_networks/multiclass_classification.rb) | Library | 5000 digit images; 400-25-15-10 Keras model |
| [`neural_networks/relu.rb`](neural_networks/relu.rb) | Ruby | original adjustable three-hinge model |
| [`neural_networks/softmax.rb`](neural_networks/softmax.rb) | Ruby | handwritten stable softmax and logit loss |
| [`evaluation/diagnosing_bias_variance.rb`](evaluation/diagnosing_bias_variance.rb) | Library | all four original diagnostic CSVs |
| [`evaluation/learning_curves.rb`](evaluation/learning_curves.rb) | Library | degree 16; original m=50 through 750 experiment |
| [`evaluation/metrics.rb`](evaluation/metrics.rb) | Ruby | original MSE and categorical-error exercises |
| [`evaluation/model_selection.rb`](evaluation/model_selection.rb) | Library | original generators, splits, degree and Ridge grids |
| [`evaluation/neural_model_selection.rb`](evaluation/neural_model_selection.rb) | Library | original six-class advice models and lambda sweep |
| [`evaluation/optional_model_evaluation.rb`](evaluation/optional_model_evaluation.rb) | Library | both original optional CSVs and exact candidate networks |
| [`trees/decision_tree.rb`](trees/decision_tree.rb) | Ruby | original mushroom entropy/information-gain assignment |
| [`trees/random_forest.rb`](trees/random_forest.rb) | Library | heart.csv; actual sklearn tree/forest sweeps |
| [`trees/xgboost.rb`](trees/xgboost.rb) | Library | heart.csv; actual XGBoost and early stopping |
| [`unsupervised/anomaly_detection.rb`](unsupervised/anomaly_detection.rb) | Ruby | both original Gaussian server datasets |
| [`unsupervised/image_compression.rb`](unsupervised/image_compression.rb) | Ruby | original bird PNG; sixteen-color reconstruction |
| [`unsupervised/k_means.rb`](unsupervised/k_means.rb) | Ruby | original 300-point, three-centroid assignment |
| [`recommenders/collaborative_filtering.rb`](recommenders/collaborative_filtering.rb) | Library | actual movie ratings; 100 latent features |
| [`recommenders/content_based_filtering.rb`](recommenders/content_based_filtering.rb) | Library | actual movie/user features; 256-128-32 towers |
| [`reinforcement_learning/deep_q_learning.rb`](reinforcement_learning/deep_q_learning.rb) | Library | actual LunarLander-v2; original 8-64-64-4 DQN |
| [`reinforcement_learning/state_action_values.rb`](reinforcement_learning/state_action_values.rb) | Ruby | original six-state stochastic policy iteration |

## Model and training contracts

| Application | Original configuration retained |
|---|---|
| Restaurant profits | 97 observations; batch GD, alpha .01, 1500 iterations |
| Admissions / microchips | actual exam/test measurements; explicit logistic gradients; microchip degree-six map and L2 |
| Binary digits | 1000 × 400 inputs; 400→25→15→1 sigmoid; BCE; Adam .001; 20 epochs |
| Multiclass digits | 5000 × 400 inputs; 400→25→15→10; ReLU hidden layers and linear logits; sparse CE; Adam .001; 40 epochs |
| Coffee roasting | original PCG64 generator and labels; normalization; 1000-fold tiling; 2→3→1 sigmoid; Adam .01; 10 epochs |
| Optional multiclass | original 100 points / four centers; 2→2→4; Adam .01; 200 epochs |
| Collaborative movies | 4778 × 443 original ratings, plus the notebook's personal-rating column; masked/normalized loss; 100 features; Adam .1; lambda 1; 200 updates |
| Content-based movies | 58187 aligned user/movie/rating pairs; source scaling/split order; 14/16 network inputs; separate 256→128→32 normalized towers; Adam .01; 30 epochs |
| Heart classifiers | original heart.csv, one-hot features, 80/20 split seed 55; sklearn tree/forest parameter sweeps and actual XGBoost early stopping |
| K-means / image | original 300 points / 3 fixed initial centers; bird_small.png / 16 colors; 10 iterations each |
| Anomalies | both original server datasets; Gaussian estimates and validation-F1 threshold selection |
| Advice / evaluation | original quadratic generator and six-class blobs, four diagnostic CSVs, optional regression/classification CSVs, source degree/lambda/model-size grids |
| Lunar Lander | actual Box2D environment; 8→64→64→4 online and target nets; replay 100000; batch 64; gamma .995; Adam .001; update every 4 steps; tau .001; epsilon decay .995/min .01; up to 2000 episodes × 1000 steps |

Source settings are defaults. `ML_ITERATIONS` explicitly bounds the polynomial
feature-engineering run; `ML_EPOCHS` bounds the two long neural-evaluation runners.
DQN exposes `--episodes` and `--steps`. These controls do not change datasets,
feature dimensions, architectures or objectives.

DQN saves newly trained weights to ignored `output/lunar_lander.keras` and accepts
`--evaluate MODEL` for inference. The supplied course model is also usable:

```sh
model=$(ruby -Ilib -rcourse_data -e 'puts CourseData.path(:dqn,"lunar_lander_model.h5")')
ruby reinforcement_learning/deep_q_learning.rb --evaluate "$model" --episodes 2
```

## Validation

`make check` verifies data availability and Ruby syntax, then runs every example.
Only the long training checks are shortened: polynomial experiments use 1000
iterations, neural advice/optional evaluation use 2 epochs, and DQN runs 2 real
episodes of up to 1000 steps. All other examples use their source budgets.
This is execution coverage, not a claim that two episodes solve Lunar Lander.

The handwritten derivative extension checks selected weight/bias derivatives
across the actual binary and multiclass digit architectures. TensorFlow training
uses TensorFlow gradients, not that extension. Random seeds are explicit;
framework/version differences can change fitted weights and scores.

See [`docs/course-map.md`](docs/course-map.md) for notebook provenance and
[`docs/theory.md`](docs/theory.md) for the equations and implementation conventions.
The content recommender preserves the source's input-scaling-before-split order;
that is a source limitation, not recommended held-out evaluation practice.

## Scope and sources

This is the Python specialization, not the older Octave course. No Octave-only
SVM, PCA or spam-classifier material is included. Introductory notebook sections
share the named algorithm kernels; notebook graphics and widgets are not copied.

[Machine Learning Specialization — DeepLearning.AI / Stanford Online](https://www.coursera.org/specializations/machine-learning-introduction),
Andrew Ng and the course team. The code and concise theory are written for this
repository. Course notebooks, lecture PDFs, datasets and supplied models remain
in the external source tree.
