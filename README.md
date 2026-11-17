# machine-learning

Machine learning algorithms, one Ruby file each. Every example trains on a
full-sized dataset and prints what happens as it goes: costs falling, splits
chosen, clusters moving, a lander learning to touch down.

You need Ruby 3.4 and make.

## Run

```bash
make                    # the menu
make linear_regression  # one example
make trees              # one family
make handwritten        # algorithms the course writes out by hand
make ported             # algorithms the course takes from a library
make check              # every example, start to finish
ruby trees/xgboost.rb   # or run a file directly
```

The examples use the course's own settings, so some of them take a while. The
six-class network sweep runs for about 45 minutes, and deep Q-learning trains
for up to 2000 Lunar Lander episodes. A full `make check` takes hours.

## What's here

| File | What it does |
|------|--------------|
| `supervised/linear_regression.rb` | predicts restaurant profit from city population with batch gradient descent |
| `supervised/feature_scaling.rb` | z-score normalization of the housing features |
| `supervised/polynomial_regression.rb` | fits x², x³ and cos(x/2) targets with engineered features |
| `supervised/logistic_regression.rb` | admits students from two exam scores |
| `supervised/regularized_logistic_regression.rb` | sorts microchips with degree-six features and L2 |
| `supervised/regularization.rb` | L2 penalties for linear and logistic regression |
| `supervised/stochastic_gradient_descent.rb` | scikit-learn's `SGDRegressor` on the housing data |
| `supervised/normal_equation.rb` | scikit-learn's `LinearRegression` |
| `supervised/sklearn_logistic_regression.rb` | scikit-learn's `LogisticRegression` |
| `neural_networks/sequential.rb` | a Keras-style `Sequential` model with dense layers, losses, backprop and `fit` |
| `neural_networks/adam.rb` | Keras's Adam optimizer |
| `neural_networks/gradient_check.rb` | checks backprop against finite differences |
| `neural_networks/dense_layer.rb` | the course's hand-written forward pass |
| `neural_networks/binary_classification.rb` | tells handwritten 0s from 1s, 400→25→15→1 |
| `neural_networks/multiclass_classification.rb` | reads all ten digits, 400→25→15→10 |
| `neural_networks/coffee_roasting.rb` | good roast or bad roast from temperature and time |
| `neural_networks/multiclass_blobs.rb` | four clusters of points, 2→2→4 |
| `neural_networks/relu.rb` | builds a piecewise-linear target out of ReLU hinges |
| `neural_networks/softmax.rb` | stable softmax and cross-entropy |
| `evaluation/metrics.rb` | half-MSE and classification error |
| `evaluation/model_selection.rb` | picks polynomial degree and λ on validation data |
| `evaluation/learning_curves.rb` | a degree-16 model trained on 50 up to 750 examples |
| `evaluation/diagnosing_bias_variance.rb` | bias and variance on the four lab datasets |
| `evaluation/neural_model_selection.rb` | network size and L2 strength on a six-class problem |
| `evaluation/optional_model_evaluation.rb` | polynomial and neural candidates, picked on validation data |
| `trees/decision_tree.rb` | edible or poisonous mushrooms by information gain |
| `trees/cart.rb` | scikit-learn's `DecisionTreeClassifier` |
| `trees/random_forest.rb` | tree and forest parameter sweeps on heart disease data |
| `trees/xgboost.rb` | gradient-boosted trees with early stopping |
| `unsupervised/k_means.rb` | 300 points, three clusters |
| `unsupervised/image_compression.rb` | squeezes a photo down to 16 colors |
| `unsupervised/anomaly_detection.rb` | flags failing servers with Gaussian densities |
| `recommenders/collaborative_filtering.rb` | predicts movie ratings from learned user and movie features |
| `recommenders/content_based_filtering.rb` | user and movie towers, 256→128→32 |
| `reinforcement_learning/state_action_values.rb` | Bellman values on a six-state corridor |
| `reinforcement_learning/lunar_lander.rb` | Gym's LunarLander-v2 with the Box2D physics it needs |
| `reinforcement_learning/deep_q_learning.rb` | trains a Q-network to land it |

When the course calls scikit-learn, TensorFlow, XGBoost or Gym, the Ruby file
runs the same algorithm with the same defaults. The random number generators
match NumPy's and Python's, so a course seed gives the course's samples,
splits and terrain. `make checks` tests them against values printed by NumPy
and Python.

[`docs/theory.md`](docs/theory.md) has the math behind each file, and
[`docs/course-map.md`](docs/course-map.md) lists which file covers each course
notebook.

## Data

Each dataset sits in [`data/`](data/README.md) under its course file name and
shape. `heart.csv` is the original Kaggle dataset (ODbL). The rest of the
course files can't be redistributed, so `data/generate.rb` builds stand-ins of
the same shape from fixed seeds. If you have the course files, point the
examples at them:

```bash
ruby data/import_course.rb /path/to/course /path/to/course-data
ML_DATA=/path/to/course-data make check
```

## Layout

```text
lib/                      terminal style, data readers, NumPy and Python random streams, Box2D
data/                     the datasets and the script that generates them
supervised/               regression and classification
neural_networks/          dense networks and their training
evaluation/               model selection, bias and variance
trees/                    decision trees, forests, boosting
unsupervised/             k-means, image compression, anomaly detection
recommenders/             collaborative and content-based filtering
reinforcement_learning/   policy iteration, Lunar Lander, DQN
checks/                   random-stream and dataset checks
```

Trained Lunar Lander weights go to `output/lunar_lander.json`, which git ignores.

Based on Andrew Ng's Machine Learning Specialization.
