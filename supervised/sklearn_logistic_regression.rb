# frozen_string_literal: true

# Original C1 W3 scikit-learn lab, including its six reference observations.
require_relative "../lib/style"
require_relative "../lib/reference"
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Scikit-Learn Logistic Regression", "original optional-lab classification data")
  x = Reference.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
  y = Reference.array([0, 0, 0, 1, 1, 1])
  model = PyCall.import_module("sklearn.linear_model").LogisticRegression.new
  model.fit(x, y)
  step("predictions #{model.predict(x)}; accuracy=#{model.score(x, y)}")
  ok("the original optional estimator example is retained separately from handwritten logistic regression")
end
