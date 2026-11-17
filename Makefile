# Machine Learning Laboratory — pure Ruby, zero dependencies.
RUBY ?= ruby
GROUPS := supervised neural_networks evaluation trees unsupervised recommenders reinforcement_learning
DEMOS := $(foreach group,$(GROUPS),$(sort $(wildcard $(group)/*.rb)))
NAMES := $(notdir $(basename $(DEMOS)))

DESC_linear_regression := batch gradients, one or multiple features
DESC_stochastic_gradient_descent := shuffle; update after each example
DESC_normal_equation := solve the least-squares stationary equations
DESC_feature_scaling := fit mean/std on training rows
DESC_polynomial_regression := change the basis, not the solver
DESC_logistic_regression := stable BCE and explicit gradients
DESC_regularization := penalize weights, exclude the bias
DESC_regularized_logistic_regression := map two inputs to polynomial monomials
DESC_dense_layer := dot products, biases, activation
DESC_relu := nonlinear hinges
DESC_softmax := subtract max; compute logsumexp loss
DESC_adam := first/second moments and bias correction
DESC_backpropagation := cache activations; propagate deltas
DESC_binary_classification := sigmoid network versus linear model
DESC_multiclass_classification := ReLU, logits and categorical loss
DESC_gradient_check := central differences for every parameter
DESC_metrics := accuracy, precision, recall and F1
DESC_model_selection := degree/λ selection using validation
DESC_learning_curves := refit as training size grows
DESC_neural_model_selection := simple, complex and L2 networks
DESC_decision_tree := entropy and information gain
DESC_random_forest := bootstrap rows; random features per node
DESC_xgboost := gradient/Hessian trees and shrinkage
DESC_k_means := alternate assignment and mean updates
DESC_image_compression := cluster RGB values into a palette
DESC_anomaly_detection := Gaussian log density; validation threshold
DESC_collaborative_filtering := masked factorization with user biases
DESC_content_based_filtering := learn normalized user/item towers
DESC_state_action_values := evaluate/improve a six-state policy
DESC_deep_q_learning := replay, target network and explicit DQN gradients

.DEFAULT_GOAL := help
.PHONY: all help list check $(GROUPS) $(NAMES)
help: list
list:
	@printf '\n── machine learning · one file per algorithm ─────────\n'
	@$(foreach group,$(GROUPS),printf '\n  $(group)\n'; $(foreach file,$(filter $(group)/%,$(DEMOS)),printf '    make %-30s # %s\n' '$(basename $(notdir $(file)))' '$(DESC_$(basename $(notdir $(file))))';)) true
	@printf '\n  make all       run every demo\n  make check     syntax checks and every executable assertion\n\n'
all: $(NAMES)
$(GROUPS):
	@$(foreach file,$(filter $@/%,$(DEMOS)),$(RUBY) $(file) &&) true

define demo_target
$(basename $(notdir $(1))):
	@$$(RUBY) $(1)
endef
$(foreach file,$(DEMOS),$(eval $(call demo_target,$(file))))

check:
	@for file in lib/*.rb $(DEMOS); do $(RUBY) -wc "$$file" >/dev/null || exit 1; done
	@$(MAKE) --no-print-directory all

.PHONY: setup-reference
setup-reference:
	@test -x .reference/python/bin/python || uv venv --python 3.11 .reference/python
	@uv pip install --python .reference/python/bin/python -r requirements-reference.txt
	@bundle config set --local path .reference/gems
	@bundle install
