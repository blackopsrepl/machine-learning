# The Python course, presented as individually runnable Ruby examples.
RUBY ?= ruby
GROUPS := supervised neural_networks evaluation trees unsupervised recommenders reinforcement_learning
DEMOS := $(foreach group,$(GROUPS),$(sort $(wildcard $(group)/*.rb)))
NAMES := $(notdir $(basename $(DEMOS)))
SCRATCH := linear_regression feature_scaling polynomial_regression logistic_regression regularized_logistic_regression regularization decision_tree k_means image_compression anomaly_detection state_action_values softmax relu metrics
LIBRARIES := stochastic_gradient_descent normal_equation sklearn_logistic_regression binary_classification multiclass_classification coffee_roasting multiclass_blobs model_selection learning_curves neural_model_selection diagnosing_bias_variance optional_model_evaluation random_forest xgboost collaborative_filtering content_based_filtering deep_q_learning
MIXED := dense_layer
EXTENSIONS := adam backpropagation gradient_check

DESC_linear_regression := restaurant profits; handwritten batch GD
DESC_feature_scaling := original four-feature housing normalization
DESC_polynomial_regression := original polynomial/cosine feature experiments
DESC_logistic_regression := exam-score admissions; handwritten gradients
DESC_regularized_logistic_regression := microchip tests; degree-six features
DESC_regularization := linear/logistic L2 on original observations
DESC_stochastic_gradient_descent := housing; actual sklearn SGDRegressor
DESC_normal_equation := housing; actual sklearn LinearRegression
DESC_sklearn_logistic_regression := original optional sklearn classifier
DESC_dense_layer := handwritten inference on original digit pixels/weights
DESC_binary_classification := 1000 digit images; 400-25-15-1 Keras model
DESC_multiclass_classification := 5000 digit images; 400-25-15-10 Keras model
DESC_coffee_roasting := original roasting generator, model and NumPy-style inference
DESC_multiclass_blobs := original four-class optional network
DESC_relu := original adjustable three-hinge model
DESC_softmax := handwritten stable softmax and logit loss
DESC_adam := handwritten optimizer; course-sized digit gradients
DESC_backpropagation := handwritten derivative extension; original digit dimensions
DESC_gradient_check := finite differences on original image inputs
DESC_metrics := original MSE and categorical-error exercises
DESC_model_selection := original generators, splits, degree and Ridge grids
DESC_learning_curves := degree 16; original m=50 through 750 experiment
DESC_neural_model_selection := original six-class advice models and lambda sweep
DESC_diagnosing_bias_variance := all four original diagnostic CSVs
DESC_optional_model_evaluation := both original optional CSVs and exact candidate networks
DESC_decision_tree := original mushroom entropy/information-gain assignment
DESC_random_forest := heart.csv; actual sklearn tree/forest sweeps
DESC_xgboost := heart.csv; actual XGBoost and early stopping
DESC_k_means := original 300-point, three-centroid assignment
DESC_image_compression := original bird PNG; sixteen-color reconstruction
DESC_anomaly_detection := both original Gaussian server datasets
DESC_collaborative_filtering := actual movie ratings; 100 latent features
DESC_content_based_filtering := actual movie/user features; 256-128-32 towers
DESC_state_action_values := original six-state stochastic policy iteration
DESC_deep_q_learning := actual LunarLander-v2; original 8-64-64-4 DQN

.DEFAULT_GOAL := help
.PHONY: all help list check data-check setup-reference scratch libraries mixed extensions $(GROUPS) $(NAMES)
help: list
list:
	@printf '\n── machine learning · original Python course ─────────\n'
	@$(foreach group,$(GROUPS),printf '\n  $(group)\n'; $(foreach file,$(filter $(group)/%,$(DEMOS)),printf '    make %-30s # %s\n' '$(basename $(notdir $(file)))' '$(DESC_$(basename $(notdir $(file))))';)) true
	@printf '\n  make data-check       locate the original course assets\n  make scratch          handwritten course algorithms\n  make libraries        actual reference libraries\n  make mixed            Ruby inference with library-trained weights\n  make extensions       explicit optimizer/backprop math\n  make all              complete original training budgets (Lander can be long)\n  make check            all data/models; bounded long-training checks\n\n'
all: $(NAMES)
scratch: $(SCRATCH)
libraries: $(LIBRARIES)
mixed: $(MIXED)
extensions: $(EXTENSIONS)
$(GROUPS):
	@$(foreach file,$(filter $@/%,$(DEMOS)),$(RUBY) $(file) &&) true

define demo_target
$(basename $(notdir $(1))):
	@$$(RUBY) $(1)
endef
$(foreach file,$(DEMOS),$(eval $(call demo_target,$(file))))

data-check:
	@$(RUBY) checks/course_data.rb

check: data-check
	@for file in lib/*.rb checks/*.rb $(DEMOS); do $(RUBY) -wc "$$file" >/dev/null || exit 1; done
	@printf '\nValidation: original data and model dimensions; polynomial=1000 iterations, advice/optional=2 epochs, LunarLander=2 episodes × up to 1000 steps. Other examples use full source budgets.\n'
	@for file in $(DEMOS); do \
	  case "$$file" in \
	    */polynomial_regression.rb) ML_ITERATIONS=1000 $(RUBY) "$$file" ;; \
	    */neural_model_selection.rb|*/optional_model_evaluation.rb) ML_EPOCHS=2 $(RUBY) "$$file" ;; \
	    */deep_q_learning.rb) $(RUBY) "$$file" --episodes 2 --steps 1000 ;; \
	    *) $(RUBY) "$$file" ;; \
	  esac || exit $$?; \
	done

setup-reference:
	@test -x .reference/python/bin/python || uv venv --python 3.11 .reference/python
	@uv pip install --python .reference/python/bin/python -r requirements-reference.txt
	@bundle config set --local path .reference/gems
	@bundle install
