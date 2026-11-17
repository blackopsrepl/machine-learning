# Python course → runnable Ruby examples

Scope: Andrew Ng / DeepLearning.AI / Stanford Online's Python Machine Learning
Specialization. The course defines the applications, dataset shapes, model
dimensions, preprocessing and training settings; only presentation changes to
terminal output. Algorithms the course writes out in NumPy are handwritten
Ruby. Algorithms it takes from scikit-learn, Keras, XGBoost or Gym/Box2D are
ported to Ruby from those libraries' own algorithms and defaults. Nothing runs
Python.

`D/` means the `DeepLearningAI-ML_Specialization/` course tree; `P/` means
`Machine-Learning-Specialization-Coursera/`. They overlap; repeated notebook names
are not separate algorithms. The source-location appendix lists the audited
non-archive notebook paths. Introductory sections share named algorithm kernels;
this is not a cell-for-cell notebook/UI port.

Datasets live in [`data/`](../data/README.md) under the course's file names and
shapes. The binary-digit loader keeps the course's first-1000-row selection.

## Coverage matrix

### `C1_W1_Lab01_Python_Jupyter_Soln.ipynb`

Language/notebook onboarding; direct Ruby/Make entry points.

[README.md](../README.md).

### `C1_W1_Lab02_Model_Representation_Soln.ipynb`

Prediction, half-MSE and batch gradients; course restaurant-profit assignment.

[supervised/linear_regression.rb](../supervised/linear_regression.rb).

### `C1_W1_Lab03_Cost_function_Soln.ipynb`

Prediction, half-MSE and batch gradients; course restaurant-profit assignment.

[supervised/linear_regression.rb](../supervised/linear_regression.rb).

### `C1_W1_Lab03_Model_Representation_Soln.ipynb`

Prediction, half-MSE and batch gradients; course restaurant-profit assignment.

[supervised/linear_regression.rb](../supervised/linear_regression.rb).

### `C1_W1_Lab04_Cost_function_Soln.ipynb`

Prediction, half-MSE and batch gradients; course restaurant-profit assignment.

[supervised/linear_regression.rb](../supervised/linear_regression.rb).

### `C1_W1_Lab04_Gradient_Descent_Soln.ipynb`

Prediction, half-MSE and batch gradients; course restaurant-profit assignment.

[supervised/linear_regression.rb](../supervised/linear_regression.rb).

### `C1_W1_Lab05_Gradient_Descent_Soln.ipynb`

Prediction, half-MSE and batch gradients; course restaurant-profit assignment.

[supervised/linear_regression.rb](../supervised/linear_regression.rb).

### `C1_W2_Lab01_Python_Numpy_Vectorization_Soln.ipynb`

Array and dot-product foundations; Ruby loops make no NumPy performance claim.

[lib/numeric.rb](../lib/numeric.rb), [neural_networks/dense_layer.rb](../neural_networks/dense_layer.rb).

### `C1_W2_Lab02_Multiple_Variable_Soln.ipynb`

Explicit multivariate gradients and the course housing application.

[supervised/feature_scaling.rb](../supervised/feature_scaling.rb), [supervised/linear_regression.rb](../supervised/linear_regression.rb).

### `C1_W2_Lab03_Feature_Scaling_and_Learning_Rate_Soln.ipynb`

Course four-feature housing data and explicit population-standardization calculations.

[supervised/feature_scaling.rb](../supervised/feature_scaling.rb).

### `C1_W2_Lab04_FeatEng_PolyReg_Soln.ipynb`

Course quadratic/cosine feature experiments, scaling, rates and iteration budgets.

[supervised/polynomial_regression.rb](../supervised/polynomial_regression.rb).

### `C1_W2_Lab05_Sklearn_GD_Soln.ipynb`

scikit-learn StandardScaler/SGDRegressor, ported, on the course housing table.

[supervised/stochastic_gradient_descent.rb](../supervised/stochastic_gradient_descent.rb).

### `C1_W2_Lab06_Sklearn_Normal_Soln.ipynb`

scikit-learn LinearRegression, ported, on the course housing table.

[supervised/normal_equation.rb](../supervised/normal_equation.rb).

### `C1_W2_Linear_Regression.ipynb`

Prediction, half-MSE and batch gradients; course restaurant-profit assignment.

[supervised/linear_regression.rb](../supervised/linear_regression.rb).

### `C1_W3_Lab01_Classification_Soln.ipynb`

Explicit sigmoid, logit boundary, BCE and gradients; course admissions application.

[supervised/logistic_regression.rb](../supervised/logistic_regression.rb).

### `C1_W3_Lab02_Sigmoid_function_Soln.ipynb`

Explicit sigmoid, logit boundary, BCE and gradients; course admissions application.

[supervised/logistic_regression.rb](../supervised/logistic_regression.rb).

### `C1_W3_Lab03_Decision_Boundary_Soln.ipynb`

Explicit sigmoid, logit boundary, BCE and gradients; course admissions application.

[supervised/logistic_regression.rb](../supervised/logistic_regression.rb).

### `C1_W3_Lab04_LogisticLoss_Soln.ipynb`

Explicit sigmoid, logit boundary, BCE and gradients; course admissions application.

[supervised/logistic_regression.rb](../supervised/logistic_regression.rb).

### `C1_W3_Lab05_Cost_Function_Soln.ipynb`

Explicit sigmoid, logit boundary, BCE and gradients; course admissions application.

[supervised/logistic_regression.rb](../supervised/logistic_regression.rb).

### `C1_W3_Lab06_Gradient_Descent_Soln.ipynb`

Explicit sigmoid, logit boundary, BCE and gradients; course admissions application.

[supervised/logistic_regression.rb](../supervised/logistic_regression.rb).

### `C1_W3_Lab07_Scikit_Learn_Soln.ipynb`

scikit-learn LogisticRegression, ported, on the six optional-lab observations.

[supervised/sklearn_logistic_regression.rb](../supervised/sklearn_logistic_regression.rb).

### `C1_W3_Lab08_Overfitting_Soln.ipynb`

Polynomial capacity and held-out selection using the course diagnostic generator and sklearn pipeline.

[evaluation/model_selection.rb](../evaluation/model_selection.rb).

### `C1_W3_Lab09_Regularization_Soln.ipynb`

Course admission/microchip inputs, degree-six mapping, weight-only L2 and explicit gradients.

[supervised/regularization.rb](../supervised/regularization.rb), [supervised/logistic_regression.rb](../supervised/logistic_regression.rb), [supervised/regularized_logistic_regression.rb](../supervised/regularized_logistic_regression.rb).

### `C1_W3_Logistic_Regression.ipynb`

Course admission/microchip inputs, degree-six mapping, weight-only L2 and explicit gradients.

[supervised/regularization.rb](../supervised/regularization.rb), [supervised/logistic_regression.rb](../supervised/logistic_regression.rb), [supervised/regularized_logistic_regression.rb](../supervised/regularized_logistic_regression.rb).

### `C2W3_Lab_01_Model_Evaluation_and_Selection.ipynb`

Both course CSVs, degree 1..10, all three course regression/classification architectures, source splits/scaling/optimizers/epochs.

[evaluation/optional_model_evaluation.rb](../evaluation/optional_model_evaluation.rb), [evaluation/metrics.rb](../evaluation/metrics.rb).

### `C2W3_Lab_02_Diagnosing_Bias_and_Variance.ipynb`

All four course diagnostic CSVs and source polynomial/Ridge comparisons.

[evaluation/diagnosing_bias_variance.rb](../evaluation/diagnosing_bias_variance.rb).

### `C2_W1_Assignment.ipynb`

Digit images, 400→25→15→1 Keras training (ported) and handwritten inference.

[neural_networks/binary_classification.rb](../neural_networks/binary_classification.rb), [neural_networks/dense_layer.rb](../neural_networks/dense_layer.rb).

### `C2_W1_Lab01_Neurons_and_Layers.ipynb`

Explicit neuron/layer calculations and the ported Keras layers.

[neural_networks/dense_layer.rb](../neural_networks/dense_layer.rb), [neural_networks/coffee_roasting.rb](../neural_networks/coffee_roasting.rb).

### `C2_W1_Lab02_CoffeeRoasting_TF.ipynb`

Course generated observations, normalization, 2→3→1 training and explicit dense inference.

[neural_networks/coffee_roasting.rb](../neural_networks/coffee_roasting.rb), [neural_networks/dense_layer.rb](../neural_networks/dense_layer.rb).

### `C2_W1_Lab03_CoffeeRoasting_Numpy.ipynb`

Course generated observations, normalization, 2→3→1 training and explicit dense inference.

[neural_networks/coffee_roasting.rb](../neural_networks/coffee_roasting.rb), [neural_networks/dense_layer.rb](../neural_networks/dense_layer.rb).

### `C2_W2_Assignment.ipynb`

Ten-digit problem and 400→25→15→10 Keras logit model (ported).

[neural_networks/multiclass_classification.rb](../neural_networks/multiclass_classification.rb), [neural_networks/softmax.rb](../neural_networks/softmax.rb).

### `C2_W2_Multiclass_TF.ipynb`

Four-center/100-point make_blobs experiment and 2→2→4 Keras model (ported).

[neural_networks/multiclass_blobs.rb](../neural_networks/multiclass_blobs.rb).

### `C2_W2_Relu.ipynb`

Course fixed/adjustable hinge model and piecewise target; CLI parameters replace sliders.

[neural_networks/relu.rb](../neural_networks/relu.rb).

### `C2_W2_SoftMax.ipynb`

Explicit stable softmax and fused categorical logit loss, plus digit training.

[neural_networks/softmax.rb](../neural_networks/softmax.rb), [neural_networks/multiclass_classification.rb](../neural_networks/multiclass_classification.rb).

### `C2_W3_Assignment.ipynb`

Course noisy quadratic generator, degree/lambda grids, sample sizes and six-class capacity/regularization experiment.

[evaluation/model_selection.rb](../evaluation/model_selection.rb), [evaluation/learning_curves.rb](../evaluation/learning_curves.rb), [evaluation/neural_model_selection.rb](../evaluation/neural_model_selection.rb).

### `C2_W4_Decision_Tree_with_Markdown.ipynb`

Course ten-row mushroom assignment; entropy, partitions and information gain are explicit Ruby.

[trees/decision_tree.rb](../trees/decision_tree.rb).

### `C2_W4_Lab_01_Decision_Trees.ipynb`

Course ten-row mushroom assignment; entropy, partitions and information gain are explicit Ruby.

[trees/decision_tree.rb](../trees/decision_tree.rb).

### `C2_W4_Lab_02_Tree_Ensemble.ipynb`

heart.csv preprocessing/split, ported scikit-learn CART/forest sweeps and ported XGBoost early stopping.

[trees/random_forest.rb](../trees/random_forest.rb), [trees/xgboost.rb](../trees/xgboost.rb).

### `C3_W1_Anomaly_Detection.ipynb`

Both course server datasets; handwritten Gaussian fitting and 1000-step validation-F1 threshold selection.

[unsupervised/anomaly_detection.rb](../unsupervised/anomaly_detection.rb).

### `C3_W1_KMeans_Assignment.ipynb`

300-point assignment, initial centers, ten iterations and 16-color bird reconstruction.

[unsupervised/k_means.rb](../unsupervised/k_means.rb), [unsupervised/image_compression.rb](../unsupervised/image_compression.rb).

### `C3_W2_Collaborative_RecSys_Assignment.ipynb`

Ratings/mask/metadata and personal column; explicit objective, written-out gradients and 100-factor Adam training.

[recommenders/collaborative_filtering.rb](../recommenders/collaborative_filtering.rb).

### `C3_W2_RecSysNN_Assignment.ipynb`

Course features/scalers/split; 256→128→32 towers, 30 epochs, new/existing-user and learned-item similarity workflows.

[recommenders/content_based_filtering.rb](../recommenders/content_based_filtering.rb).

### `C3_W3_A1_Assignment.ipynb`

LunarLander-v2 ported with its Box2D physics; course observation/action spaces, networks, replay, objective and training settings.

[reinforcement_learning/deep_q_learning.rb](../reinforcement_learning/deep_q_learning.rb).

### `State-action value function example.ipynb`

Course six-state reward/transition conventions and policy iteration.

[reinforcement_learning/state_action_values.rb](../reinforcement_learning/state_action_values.rb).

## Execution boundaries

- `make check` runs every entry point at the course's settings, including the
  1,000,000-iteration polynomial runs and up to 2000 Lunar Lander episodes.
- Course input-scaling-before-split in the content recommender is preserved and
  documented, not silently corrected into a different workflow.
- Gym 0.26 separates terminated/truncated flags; the DQN combines them to retain
  the course's `done` replay semantics.
- The Lunar Lander port omits Box2D's continuous collision; everything else the
  environment uses is ported.

## Source-location appendix

**`C1_W1_Lab01_Python_Jupyter_Soln.ipynb`**

- `D/1-Regression-Classification/week1/C1_W1_Lab01_Python_Jupyter_Soln.ipynb`
- `P/C1 - Supervised Machine Learning - Regression and Classification/week1/Optional Labs/C1_W1_Lab01_Python_Jupyter_Soln.ipynb`

**`C1_W1_Lab02_Model_Representation_Soln.ipynb`**

- `D/1-Regression-Classification/week1/C1_W1_Lab02_Model_Representation_Soln.ipynb`

**`C1_W1_Lab03_Cost_function_Soln.ipynb`**

- `D/1-Regression-Classification/week1/C1_W1_Lab03_Cost_function_Soln.ipynb`

**`C1_W1_Lab03_Model_Representation_Soln.ipynb`**

- `P/C1 - Supervised Machine Learning - Regression and Classification/week1/Optional Labs/C1_W1_Lab03_Model_Representation_Soln.ipynb`

**`C1_W1_Lab04_Cost_function_Soln.ipynb`**

- `P/C1 - Supervised Machine Learning - Regression and Classification/week1/Optional Labs/C1_W1_Lab04_Cost_function_Soln.ipynb`

**`C1_W1_Lab04_Gradient_Descent_Soln.ipynb`**

- `D/1-Regression-Classification/week1/C1_W1_Lab04_Gradient_Descent_Soln.ipynb`

**`C1_W1_Lab05_Gradient_Descent_Soln.ipynb`**

- `P/C1 - Supervised Machine Learning - Regression and Classification/week1/Optional Labs/C1_W1_Lab05_Gradient_Descent_Soln.ipynb`

**`C1_W2_Lab01_Python_Numpy_Vectorization_Soln.ipynb`**

- `D/1-Regression-Classification/week2/C1_W2_Lab01_Python_Numpy_Vectorization_Soln.ipynb`
- `P/C1 - Supervised Machine Learning - Regression and Classification/week2/Optional Labs/C1_W2_Lab01_Python_Numpy_Vectorization_Soln.ipynb`

**`C1_W2_Lab02_Multiple_Variable_Soln.ipynb`**

- `D/1-Regression-Classification/week2/C1_W2_Lab02_Multiple_Variable_Soln.ipynb`
- `P/C1 - Supervised Machine Learning - Regression and Classification/week2/Optional Labs/C1_W2_Lab02_Multiple_Variable_Soln.ipynb`

**`C1_W2_Lab03_Feature_Scaling_and_Learning_Rate_Soln.ipynb`**

- `D/1-Regression-Classification/week2/C1_W2_Lab03_Feature_Scaling_and_Learning_Rate_Soln.ipynb`
- `P/C1 - Supervised Machine Learning - Regression and Classification/week2/Optional Labs/C1_W2_Lab03_Feature_Scaling_and_Learning_Rate_Soln.ipynb`

**`C1_W2_Lab04_FeatEng_PolyReg_Soln.ipynb`**

- `D/1-Regression-Classification/week2/C1_W2_Lab04_FeatEng_PolyReg_Soln.ipynb`
- `P/C1 - Supervised Machine Learning - Regression and Classification/week2/Optional Labs/C1_W2_Lab04_FeatEng_PolyReg_Soln.ipynb`

**`C1_W2_Lab05_Sklearn_GD_Soln.ipynb`**

- `D/1-Regression-Classification/week2/C1_W2_Lab05_Sklearn_GD_Soln.ipynb`
- `P/C1 - Supervised Machine Learning - Regression and Classification/week2/Optional Labs/C1_W2_Lab05_Sklearn_GD_Soln.ipynb`

**`C1_W2_Lab06_Sklearn_Normal_Soln.ipynb`**

- `D/1-Regression-Classification/week2/C1_W2_Lab06_Sklearn_Normal_Soln.ipynb`
- `P/C1 - Supervised Machine Learning - Regression and Classification/week2/Optional Labs/C1_W2_Lab06_Sklearn_Normal_Soln.ipynb`

**`C1_W2_Linear_Regression.ipynb`**

- `P/C1 - Supervised Machine Learning - Regression and Classification/week2/C1W2A1/C1_W2_Linear_Regression.ipynb`

**`C1_W3_Lab01_Classification_Soln.ipynb`**

- `D/1-Regression-Classification/week3/C1_W3_Lab01_Classification_Soln.ipynb`
- `P/C1 - Supervised Machine Learning - Regression and Classification/week3/Optional Labs/C1_W3_Lab01_Classification_Soln.ipynb`

**`C1_W3_Lab02_Sigmoid_function_Soln.ipynb`**

- `D/1-Regression-Classification/week3/C1_W3_Lab02_Sigmoid_function_Soln.ipynb`
- `P/C1 - Supervised Machine Learning - Regression and Classification/week3/Optional Labs/C1_W3_Lab02_Sigmoid_function_Soln.ipynb`

**`C1_W3_Lab03_Decision_Boundary_Soln.ipynb`**

- `D/1-Regression-Classification/week3/C1_W3_Lab03_Decision_Boundary_Soln.ipynb`
- `P/C1 - Supervised Machine Learning - Regression and Classification/week3/Optional Labs/C1_W3_Lab03_Decision_Boundary_Soln.ipynb`

**`C1_W3_Lab04_LogisticLoss_Soln.ipynb`**

- `D/1-Regression-Classification/week3/C1_W3_Lab04_LogisticLoss_Soln.ipynb`
- `P/C1 - Supervised Machine Learning - Regression and Classification/week3/Optional Labs/C1_W3_Lab04_LogisticLoss_Soln.ipynb`

**`C1_W3_Lab05_Cost_Function_Soln.ipynb`**

- `D/1-Regression-Classification/week3/C1_W3_Lab05_Cost_Function_Soln.ipynb`
- `P/C1 - Supervised Machine Learning - Regression and Classification/week3/Optional Labs/C1_W3_Lab05_Cost_Function_Soln.ipynb`

**`C1_W3_Lab06_Gradient_Descent_Soln.ipynb`**

- `D/1-Regression-Classification/week3/C1_W3_Lab06_Gradient_Descent_Soln.ipynb`
- `P/C1 - Supervised Machine Learning - Regression and Classification/week3/Optional Labs/C1_W3_Lab06_Gradient_Descent_Soln.ipynb`

**`C1_W3_Lab07_Scikit_Learn_Soln.ipynb`**

- `D/1-Regression-Classification/week3/C1_W3_Lab07_Scikit_Learn_Soln.ipynb`
- `P/C1 - Supervised Machine Learning - Regression and Classification/week3/Optional Labs/C1_W3_Lab07_Scikit_Learn_Soln.ipynb`

**`C1_W3_Lab08_Overfitting_Soln.ipynb`**

- `D/1-Regression-Classification/week3/C1_W3_Lab08_Overfitting_Soln.ipynb`
- `P/C1 - Supervised Machine Learning - Regression and Classification/week3/Optional Labs/C1_W3_Lab08_Overfitting_Soln.ipynb`

**`C1_W3_Lab09_Regularization_Soln.ipynb`**

- `D/1-Regression-Classification/week3/C1_W3_Lab09_Regularization_Soln.ipynb`
- `P/C1 - Supervised Machine Learning - Regression and Classification/week3/Optional Labs/C1_W3_Lab09_Regularization_Soln.ipynb`

**`C1_W3_Logistic_Regression.ipynb`**

- `D/1-Regression-Classification/LogisticRegression/C1_W3_Logistic_Regression.ipynb`
- `P/C1 - Supervised Machine Learning - Regression and Classification/week3/C1W3A1/C1_W3_Logistic_Regression.ipynb`

**`C2W3_Lab_01_Model_Evaluation_and_Selection.ipynb`**

- `D/2-Advanced-Algorithms/week3/C2W3_Lab_01_Model_Evaluation_and_Selection.ipynb`

**`C2W3_Lab_02_Diagnosing_Bias_and_Variance.ipynb`**

- `D/2-Advanced-Algorithms/week3/C2W3_Lab_02_Diagnosing_Bias_and_Variance.ipynb`

**`C2_W1_Assignment.ipynb`**

- `D/2-Advanced-Algorithms/MulticlassNeuralNetwork/Files/home/jovyan/work/C2_W1_Assignment.ipynb`
- `P/C2 - Advanced Learning Algorithms/week1/C2W1A1/C2_W1_Assignment.ipynb`

**`C2_W1_Lab01_Neurons_and_Layers.ipynb`**

- `D/2-Advanced-Algorithms/week1/C2_W1_Lab01_Neurons_and_Layers.ipynb`
- `P/C2 - Advanced Learning Algorithms/week1/optional-labs/C2_W1_Lab01_Neurons_and_Layers.ipynb`

**`C2_W1_Lab02_CoffeeRoasting_TF.ipynb`**

- `D/2-Advanced-Algorithms/week1/C2_W1_Lab02_CoffeeRoasting_TF.ipynb`
- `P/C2 - Advanced Learning Algorithms/week1/optional-labs/C2_W1_Lab02_CoffeeRoasting_TF.ipynb`

**`C2_W1_Lab03_CoffeeRoasting_Numpy.ipynb`**

- `D/2-Advanced-Algorithms/week1/C2_W1_Lab03_CoffeeRoasting_Numpy.ipynb`
- `P/C2 - Advanced Learning Algorithms/week1/optional-labs/C2_W1_Lab03_CoffeeRoasting_Numpy.ipynb`

**`C2_W2_Assignment.ipynb`**

- `D/2-Advanced-Algorithms/BinaryNeuralNetwork/C2_W2_Assignment.ipynb`
- `P/C2 - Advanced Learning Algorithms/week2/C2W2A1/C2_W2_Assignment.ipynb`

**`C2_W2_Multiclass_TF.ipynb`**

- `D/2-Advanced-Algorithms/week2/C2_W2_Multiclass_TF.ipynb`
- `P/C2 - Advanced Learning Algorithms/week2/optional-labs/C2_W2_Multiclass_TF.ipynb`

**`C2_W2_Relu.ipynb`**

- `D/2-Advanced-Algorithms/week2/C2_W2_Relu.ipynb`
- `P/C2 - Advanced Learning Algorithms/week2/optional-labs/C2_W2_Relu.ipynb`

**`C2_W2_SoftMax.ipynb`**

- `D/2-Advanced-Algorithms/week2/C2_W2_SoftMax.ipynb`
- `P/C2 - Advanced Learning Algorithms/week2/optional-labs/C2_W2_SoftMax.ipynb`

**`C2_W3_Assignment.ipynb`**

- `D/2-Advanced-Algorithms/MachineLearningAdvice/Files/home/jovyan/work/C2_W3_Assignment.ipynb`
- `P/C2 - Advanced Learning Algorithms/week3/C2W3A1/C2_W3_Assignment.ipynb`

**`C2_W4_Decision_Tree_with_Markdown.ipynb`**

- `D/2-Advanced-Algorithms/DecisionTrees/C2_W4_Decision_Tree_with_Markdown.ipynb`
- `P/C2 - Advanced Learning Algorithms/week4/C2W4A1/C2_W4_Decision_Tree_with_Markdown.ipynb`

**`C2_W4_Lab_01_Decision_Trees.ipynb`**

- `D/2-Advanced-Algorithms/week4/C2_W4_Lab_01_Decision_Trees.ipynb`

**`C2_W4_Lab_02_Tree_Ensemble.ipynb`**

- `D/2-Advanced-Algorithms/week4/C2_W4_Lab_02_Tree_Ensemble.ipynb`

**`C3_W1_Anomaly_Detection.ipynb`**

- `D/3-Unsupervised Learning, Recommenders, Reinforcement Learning/week1/C3W1A/C3W1A2/C3_W1_Anomaly_Detection.ipynb`
- `P/C3 - Unsupervised Learning, Recommenders, Reinforcement Learning/week1/C3W1A/C3W1A2/C3_W1_Anomaly_Detection.ipynb`

**`C3_W1_KMeans_Assignment.ipynb`**

- `D/3-Unsupervised Learning, Recommenders, Reinforcement Learning/week1/C3W1A/C3W1A1/C3_W1_KMeans_Assignment.ipynb`
- `P/C3 - Unsupervised Learning, Recommenders, Reinforcement Learning/week1/C3W1A/C3W1A1/C3_W1_KMeans_Assignment.ipynb`

**`C3_W2_Collaborative_RecSys_Assignment.ipynb`**

- `D/3-Unsupervised Learning, Recommenders, Reinforcement Learning/week2/C3W2/C3W2A1/C3_W2_Collaborative_RecSys_Assignment.ipynb`
- `P/C3 - Unsupervised Learning, Recommenders, Reinforcement Learning/week2/C3W2/C3W2A1/C3_W2_Collaborative_RecSys_Assignment.ipynb`

**`C3_W2_RecSysNN_Assignment.ipynb`**

- `D/3-Unsupervised Learning, Recommenders, Reinforcement Learning/week2/C3W2/C3W2A2/C3_W2_RecSysNN_Assignment.ipynb`
- `P/C3 - Unsupervised Learning, Recommenders, Reinforcement Learning/week2/C3W2/C3W2A2/C3_W2_RecSysNN_Assignment.ipynb`

**`C3_W3_A1_Assignment.ipynb`**

- `D/3-Unsupervised Learning, Recommenders, Reinforcement Learning/week3/C3W3A1/C3_W3_A1_Assignment.ipynb`
- `P/C3 - Unsupervised Learning, Recommenders, Reinforcement Learning/week3/C3W3A1/C3_W3_A1_Assignment.ipynb`

**`State-action value function example.ipynb`**

- `D/3-Unsupervised Learning, Recommenders, Reinforcement Learning/week3/optional-labs/State-action value function example.ipynb`
- `P/C3 - Unsupervised Learning, Recommenders, Reinforcement Learning/week3/optional-labs/State-action value function example.ipynb`
