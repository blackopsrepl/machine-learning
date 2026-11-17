# Python course → runnable Ruby examples

Scope: Andrew Ng / DeepLearning.AI / Stanford Online's
[Python Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction).
The older Octave course is not the catalogue. Algorithm families are preserved;
fixtures, network widths and terminal presentation are deliberately smaller.

The source directory contains overlapping checkouts/copies. The audited inventory
contains **77 non-archive notebook paths with 44 distinct filenames**. Alternate
week-1 numbering does not create additional algorithms. Archives/beta/scaffolds
are not new requirements. Every non-archive notebook filename is mapped below;
the source-location appendix retains all 77 paths so duplicate names cannot hide
missing assignments or optional labs.

Local source root:
`/srv/res/computing/DataEng/MachineLearningSpecialist/`

- `D/` means `DeepLearningAI-ML_Specialization/`.
- `P/` means `Machine-Learning-Specialization-Coursera/`.

Source inventories in that directory describe static inspection, not a claim that
the legacy notebooks execute unchanged. Ruby numerical checks exercise this
repository's actual implementations. The Ruby code and theory explanations are
new implementations, not mechanically translated or redistributed notebooks.
The tree fixture retains the course's tiny mushroom teaching table.

## Coverage matrix

### `C1_W1_Lab01_Python_Jupyter_Soln.ipynb`

Language/notebook onboarding, not a separate learning algorithm; replaced by direct Ruby/Make entry points.

[README.md](../README.md).

### `C1_W1_Lab02_Model_Representation_Soln.ipynb`

Same prediction, half-MSE and explicit gradient updates; one and multiple features share the trainer.

[supervised/linear_regression.rb](../supervised/linear_regression.rb).

### `C1_W1_Lab03_Cost_function_Soln.ipynb`

Same prediction, half-MSE and explicit gradient updates; one and multiple features share the trainer.

[supervised/linear_regression.rb](../supervised/linear_regression.rb).

### `C1_W1_Lab03_Model_Representation_Soln.ipynb`

Same prediction, half-MSE and explicit gradient updates; one and multiple features share the trainer.

[supervised/linear_regression.rb](../supervised/linear_regression.rb).

### `C1_W1_Lab04_Cost_function_Soln.ipynb`

Same prediction, half-MSE and explicit gradient updates; one and multiple features share the trainer.

[supervised/linear_regression.rb](../supervised/linear_regression.rb).

### `C1_W1_Lab04_Gradient_Descent_Soln.ipynb`

Same prediction, half-MSE and explicit gradient updates; one and multiple features share the trainer.

[supervised/linear_regression.rb](../supervised/linear_regression.rb).

### `C1_W1_Lab05_Gradient_Descent_Soln.ipynb`

Same prediction, half-MSE and explicit gradient updates; one and multiple features share the trainer.

[supervised/linear_regression.rb](../supervised/linear_regression.rb).

### `C1_W2_Lab01_Python_Numpy_Vectorization_Soln.ipynb`

Array/dot-product foundations; no NumPy performance claim is transferred to Ruby arrays.

[lib/numeric.rb](../lib/numeric.rb), [neural_networks/dense_layer.rb](../neural_networks/dense_layer.rb).

### `C1_W2_Lab02_Multiple_Variable_Soln.ipynb`

Same prediction, half-MSE and explicit gradient updates; one and multiple features share the trainer.

[supervised/linear_regression.rb](../supervised/linear_regression.rb).

### `C1_W2_Lab03_Feature_Scaling_and_Learning_Rate_Soln.ipynb`

Training-only population mean/std and a real divergence/convergence contrast.

[supervised/feature_scaling.rb](../supervised/feature_scaling.rb).

### `C1_W2_Lab04_FeatEng_PolyReg_Soln.ipynb`

Explicit polynomial feature construction.

[supervised/polynomial_regression.rb](../supervised/polynomial_regression.rb).

### `C1_W2_Lab05_Sklearn_GD_Soln.ipynb`

Reimplements the stochastic-update algorithm, not SGDRegressor API/default equivalence.

[supervised/stochastic_gradient_descent.rb](../supervised/stochastic_gradient_descent.rb).

### `C1_W2_Lab06_Sklearn_Normal_Soln.ipynb`

Same OLS objective; small full-rank normal-equation solve rather than sklearn numerical internals.

[supervised/normal_equation.rb](../supervised/normal_equation.rb).

### `C1_W2_Linear_Regression.ipynb`

Same prediction, half-MSE and explicit gradient updates; one and multiple features share the trainer.

[supervised/linear_regression.rb](../supervised/linear_regression.rb).

### `C1_W3_Lab01_Classification_Soln.ipynb`

Sigmoid/logit boundary, stable Bernoulli loss and explicit gradients replace widgets/estimator calls.

[supervised/logistic_regression.rb](../supervised/logistic_regression.rb).

### `C1_W3_Lab02_Sigmoid_function_Soln.ipynb`

Sigmoid/logit boundary, stable Bernoulli loss and explicit gradients replace widgets/estimator calls.

[supervised/logistic_regression.rb](../supervised/logistic_regression.rb).

### `C1_W3_Lab03_Decision_Boundary_Soln.ipynb`

Sigmoid/logit boundary, stable Bernoulli loss and explicit gradients replace widgets/estimator calls.

[supervised/logistic_regression.rb](../supervised/logistic_regression.rb).

### `C1_W3_Lab04_LogisticLoss_Soln.ipynb`

Sigmoid/logit boundary, stable Bernoulli loss and explicit gradients replace widgets/estimator calls.

[supervised/logistic_regression.rb](../supervised/logistic_regression.rb).

### `C1_W3_Lab05_Cost_Function_Soln.ipynb`

Sigmoid/logit boundary, stable Bernoulli loss and explicit gradients replace widgets/estimator calls.

[supervised/logistic_regression.rb](../supervised/logistic_regression.rb).

### `C1_W3_Lab06_Gradient_Descent_Soln.ipynb`

Sigmoid/logit boundary, stable Bernoulli loss and explicit gradients replace widgets/estimator calls.

[supervised/logistic_regression.rb](../supervised/logistic_regression.rb).

### `C1_W3_Lab07_Scikit_Learn_Soln.ipynb`

Sigmoid/logit boundary, stable Bernoulli loss and explicit gradients replace widgets/estimator calls.

[supervised/logistic_regression.rb](../supervised/logistic_regression.rb).

### `C1_W3_Lab08_Overfitting_Soln.ipynb`

The underfit/overfit story becomes deterministic train/validation errors, not an estimator-backed widget.

[evaluation/model_selection.rb](../evaluation/model_selection.rb).

### `C1_W3_Lab09_Regularization_Soln.ipynb`

Both linear/logistic penalties, weight gradients and unpenalized bias.

[supervised/regularization.rb](../supervised/regularization.rb), [supervised/regularized_logistic_regression.rb](../supervised/regularized_logistic_regression.rb).

### `C1_W3_Logistic_Regression.ipynb`

Both unregularized and polynomial/L2 logistic classification; explicit sixth-degree map is available.

[supervised/logistic_regression.rb](../supervised/logistic_regression.rb), [supervised/regularized_logistic_regression.rb](../supervised/regularized_logistic_regression.rb).

### `C2W3_Lab_01_Model_Evaluation_and_Selection.ipynb`

Train/CV/test discipline; degree, regularization, sample size and neural capacity comparisons.

[evaluation/metrics.rb](../evaluation/metrics.rb), [evaluation/model_selection.rb](../evaluation/model_selection.rb), [evaluation/learning_curves.rb](../evaluation/learning_curves.rb), [evaluation/neural_model_selection.rb](../evaluation/neural_model_selection.rb).

### `C2W3_Lab_02_Diagnosing_Bias_and_Variance.ipynb`

Train/CV/test discipline; degree, regularization, sample size and neural capacity comparisons.

[evaluation/metrics.rb](../evaluation/metrics.rb), [evaluation/model_selection.rb](../evaluation/model_selection.rb), [evaluation/learning_curves.rb](../evaluation/learning_curves.rb), [evaluation/neural_model_selection.rb](../evaluation/neural_model_selection.rb).

### `C2_W1_Assignment.ipynb`

Binary sigmoid MLP inference and training use shared toy classification rather than coffee/digit assets.

[neural_networks/dense_layer.rb](../neural_networks/dense_layer.rb), [neural_networks/binary_classification.rb](../neural_networks/binary_classification.rb), [neural_networks/backpropagation.rb](../neural_networks/backpropagation.rb).

### `C2_W1_Lab01_Neurons_and_Layers.ipynb`

Linear/sigmoid neuron mathematics; no TensorFlow layer objects.

[neural_networks/dense_layer.rb](../neural_networks/dense_layer.rb), [supervised/linear_regression.rb](../supervised/linear_regression.rb), [supervised/logistic_regression.rb](../supervised/logistic_regression.rb).

### `C2_W1_Lab02_CoffeeRoasting_TF.ipynb`

Binary sigmoid MLP inference and training use shared toy classification rather than coffee/digit assets.

[neural_networks/dense_layer.rb](../neural_networks/dense_layer.rb), [neural_networks/binary_classification.rb](../neural_networks/binary_classification.rb), [neural_networks/backpropagation.rb](../neural_networks/backpropagation.rb).

### `C2_W1_Lab03_CoffeeRoasting_Numpy.ipynb`

Binary sigmoid MLP inference and training use shared toy classification rather than coffee/digit assets.

[neural_networks/dense_layer.rb](../neural_networks/dense_layer.rb), [neural_networks/binary_classification.rb](../neural_networks/binary_classification.rb), [neural_networks/backpropagation.rb](../neural_networks/backpropagation.rb).

### `C2_W2_Assignment.ipynb`

ReLU multiclass network, categorical loss and optimizer; smaller model/fixture, same algorithm family.

[neural_networks/multiclass_classification.rb](../neural_networks/multiclass_classification.rb), [neural_networks/backpropagation.rb](../neural_networks/backpropagation.rb), [neural_networks/adam.rb](../neural_networks/adam.rb).

### `C2_W2_Multiclass_TF.ipynb`

ReLU multiclass network, categorical loss and optimizer; smaller model/fixture, same algorithm family.

[neural_networks/multiclass_classification.rb](../neural_networks/multiclass_classification.rb), [neural_networks/backpropagation.rb](../neural_networks/backpropagation.rb), [neural_networks/adam.rb](../neural_networks/adam.rb).

### `C2_W2_Relu.ipynb`

Nonlinear hinge construction is visible as arithmetic rather than a slider.

[neural_networks/relu.rb](../neural_networks/relu.rb).

### `C2_W2_SoftMax.ipynb`

Stable softmax and logit-space cross-entropy.

[neural_networks/softmax.rb](../neural_networks/softmax.rb).

### `C2_W3_Assignment.ipynb`

Train/CV/test discipline; degree, regularization, sample size and neural capacity comparisons.

[evaluation/metrics.rb](../evaluation/metrics.rb), [evaluation/model_selection.rb](../evaluation/model_selection.rb), [evaluation/learning_curves.rb](../evaluation/learning_curves.rb), [evaluation/neural_model_selection.rb](../evaluation/neural_model_selection.rb).

### `C2_W4_Decision_Tree_with_Markdown.ipynb`

Entropy/gain, recursive construction, explicit stopping and a real prediction function.

[trees/decision_tree.rb](../trees/decision_tree.rb).

### `C2_W4_Lab_01_Decision_Trees.ipynb`

Entropy/gain, recursive construction, explicit stopping and a real prediction function.

[trees/decision_tree.rb](../trees/decision_tree.rb).

### `C2_W4_Lab_02_Tree_Ensemble.ipynb`

Reimplements tree, bagged/random-feature forest and second-order boosting. Educational exact-greedy XGBoost core, not external engine/API parity.

[trees/decision_tree.rb](../trees/decision_tree.rb), [trees/random_forest.rb](../trees/random_forest.rb), [trees/xgboost.rb](../trees/xgboost.rb).

### `C3_W1_Anomaly_Detection.ipynb`

Gaussian fitting and validation F1 threshold selection, moved to log space.

[unsupervised/anomaly_detection.rb](../unsupervised/anomaly_detection.rb), [evaluation/metrics.rb](../evaluation/metrics.rb).

### `C3_W1_KMeans_Assignment.ipynb`

Assignment/update/random initialization and RGB palette reconstruction.

[unsupervised/k_means.rb](../unsupervised/k_means.rb), [unsupervised/image_compression.rb](../unsupervised/image_compression.rb).

### `C3_W2_Collaborative_RecSys_Assignment.ipynb`

Mean-normalized observed-only matrix factorization, user biases, L2 and manual gradients replace TensorFlow autodiff.

[recommenders/collaborative_filtering.rb](../recommenders/collaborative_filtering.rb).

### `C3_W2_RecSysNN_Assignment.ipynb`

Two towers, embedding normalization, dot-product rating prediction and squared item distance; all derivatives explicit.

[recommenders/content_based_filtering.rb](../recommenders/content_based_filtering.rb).

### `C3_W3_A1_Assignment.ipynb`

DQN replay/target/epsilon-greedy/Adam algorithm; explicit continuous corridor replaces LunarLander-v2 physics.

[reinforcement_learning/deep_q_learning.rb](../reinforcement_learning/deep_q_learning.rb).

### `State-action value function example.ipynb`

Six-state transition model, policy evaluation/improvement and Q values, including helper code.

[reinforcement_learning/state_action_values.rb](../reinforcement_learning/state_action_values.rb).

## Supporting mechanisms

The Python course invokes Adam through TensorFlow and differentiates composite
models automatically. `neural_networks/adam.rb`, `backpropagation.rb` and
`gradient_check.rb` make those necessary mechanisms explicit. They support the
same course models; they do not import extra Octave algorithms.

General numerical helpers live in `lib/numeric.rb`, not external gems. A normal
equation solve is intentionally restricted to small nonsingular systems. The
XGBoost example is a second-order regularized boosting implementation, not an
API/feature-complete clone of the external library. DQN uses a new toy environment,
not LunarLander physics. See [theory](theory.md) for objective/loss conventions.

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
