# Andrew Ng's Python Machine Learning Specialization as runnable Ruby files.
RUBY ?= ruby
GROUPS := supervised neural_networks evaluation trees unsupervised recommenders reinforcement_learning
DEMOS := $(foreach group,$(GROUPS),$(sort $(wildcard $(group)/*.rb)))
NAMES := $(notdir $(basename $(DEMOS)))
# The course writes these algorithms out itself (NumPy).
HANDWRITTEN := linear_regression feature_scaling polynomial_regression logistic_regression regularized_logistic_regression regularization dense_layer relu softmax metrics decision_tree k_means image_compression anomaly_detection state_action_values
# The course calls a library for these (scikit-learn, TensorFlow, XGBoost, Gym/Box2D).
PORTED := stochastic_gradient_descent normal_equation sklearn_logistic_regression binary_classification multiclass_classification coffee_roasting multiclass_blobs model_selection learning_curves diagnosing_bias_variance neural_model_selection optional_model_evaluation cart random_forest xgboost collaborative_filtering content_based_filtering lunar_lander deep_q_learning
# The Keras machinery the ported networks run on.
ENGINES := sequential adam gradient_check

DESC_linear_regression := restaurant profits; batch gradient descent
DESC_feature_scaling := housing features; z-score normalization
DESC_polynomial_regression := engineered features; x², x³, cos(x/2) fits
DESC_logistic_regression := exam scores → admission; gradient descent
DESC_regularized_logistic_regression := microchip tests; degree-six features, L2
DESC_regularization := L2 for linear and logistic regression
DESC_stochastic_gradient_descent := scikit-learn SGDRegressor on housing
DESC_normal_equation := scikit-learn LinearRegression (least squares)
DESC_sklearn_logistic_regression := scikit-learn LogisticRegression (C=1)
DESC_sequential := Keras Sequential: dense layers, losses, backprop, fit
DESC_adam := Keras Adam update rule
DESC_gradient_check := finite differences against backpropagation
DESC_dense_layer := NumPy my_dense / my_sequential forward pass
DESC_binary_classification := digits 0/1; 400→25→15→1 sigmoid network
DESC_multiclass_classification := digits 0–9; 400→25→15→10 ReLU, logits
DESC_coffee_roasting := roast data, Normalization, 2→3→1 network
DESC_multiclass_blobs := make_blobs four classes; 2→2→4 network
DESC_relu := three-hinge ReLU model (slider lab)
DESC_softmax := stable softmax and cross-entropy
DESC_metrics := half-MSE and classification error
DESC_model_selection := degree and λ chosen on validation data
DESC_learning_curves := degree 16 with 50→750 examples
DESC_diagnosing_bias_variance := bias/variance diagnosis on the lab CSVs
DESC_neural_model_selection := six-class networks and L2 sweep
DESC_optional_model_evaluation := polynomial and neural candidates, two CSVs
DESC_decision_tree := mushrooms; entropy and information gain
DESC_cart := scikit-learn DecisionTreeClassifier on heart.csv
DESC_random_forest := tree and forest sweeps on heart.csv
DESC_xgboost := exact greedy XGBoost with early stopping
DESC_k_means := 300 points, three clusters
DESC_image_compression := bird image to 16 colors
DESC_anomaly_detection := Gaussian densities, F1-chosen threshold
DESC_collaborative_filtering := 4778×443 ratings; 100 latent features
DESC_content_based_filtering := user/movie towers, normalized dot product
DESC_state_action_values := six-state Bellman values and policy
DESC_lunar_lander := Gym LunarLander-v2 on a Box2D port
DESC_deep_q_learning := DQN: replay, target network, ε-greedy

.DEFAULT_GOAL := help
.PHONY: all help list check checks handwritten ported engines data $(GROUPS) $(NAMES)
help: list
list:
	@printf '\n── machine learning · Python ML Specialization in Ruby ─────────\n'
	@$(foreach group,$(GROUPS),printf '\n  $(group)\n'; $(foreach file,$(filter $(group)/%,$(DEMOS)),printf '    make %-32s # %s\n' '$(basename $(notdir $(file)))' '$(DESC_$(basename $(notdir $(file))))';)) true
	@printf '\n  make handwritten     algorithms the course writes out in NumPy\n  make ported          algorithms the course takes from a library\n  make engines         Keras-style training machinery\n  make all             every example at the course settings\n  make checks          dataset schema and NumPy/CPython random streams\n  make check           syntax, checks, then every example\n  make data            regenerate the shipped datasets\n\n'
all: $(NAMES)
handwritten: $(HANDWRITTEN)
ported: $(PORTED)
engines: $(ENGINES)
$(GROUPS):
	@$(foreach file,$(filter $@/%,$(DEMOS)),$(RUBY) $(file) &&) true

define demo_target
$(basename $(notdir $(1))):
	@$$(RUBY) $(1)
endef
$(foreach file,$(DEMOS),$(eval $(call demo_target,$(file))))

checks:
	@$(RUBY) checks/numpy_random.rb
	@$(RUBY) checks/course_data.rb

check: checks
	@for file in lib/*.rb checks/*.rb data/*.rb $(DEMOS); do $(RUBY) -wc "$$file" >/dev/null || exit 1; done
	@for file in $(DEMOS); do $(RUBY) "$$file" || exit $$?; done

data:
	@$(RUBY) data/generate.rb
