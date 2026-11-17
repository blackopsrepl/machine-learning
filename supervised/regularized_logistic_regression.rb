# frozen_string_literal: true

# Regularized nonlinear logistic regression — map two features to monomials.
# Degree d includes every x1^(i-j)*x2^j for i=1..d; bias stays separate.
# A circle cannot be separated by one line, but x1²+x2² makes it linear in φ(x).
# Run me: ruby supervised/regularized_logistic_regression.rb
require_relative "logistic_regression"
module RegularizedLogisticRegression
  module_function

  def map_features(rows, degree: 6)
    rows.map { |a, b| (1..degree).flat_map { |power| (0..power).map { |j| a**(power - j) * b**j } } }
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Regularized Logistic Regression", "polynomial feature map · nonlinear boundary")
  x = [-1.5, -0.5, 0.5, 1.5].product([-1.5, -0.5, 0.5, 1.5])
  y = x.map { |a, b| a * a + b * b < 1 ? 1 : 0 }
  linear_w, linear_b = LogisticRegression.fit(x, y)
  linear = x.map { |r| LogisticRegression.probability(r, linear_w, linear_b) >= 0.5 ? 1 : 0 }
  expanded = RegularizedLogisticRegression.map_features(x, degree: 2)
  scaler = NumericTools.fit_scaler(expanded)
  features = NumericTools.transform(expanded, scaler)
  w, b = LogisticRegression.fit(features, y, l2: 0.1, iterations: 2000) { |i, weights, bias, cost| step("it #{i}: penalized BCE=#{cost.round(6)} ||w||²=#{weights.sum { |v| v * v }.round(4)} b=#{bias.round(4)}") if i <= 2 || (i % 500).zero? }
  predicted = features.map { |r| LogisticRegression.probability(r, w, b) >= 0.5 ? 1 : 0 }
  step("linear accuracy #{NumericTools.accuracy(y, linear)}; quadratic accuracy #{NumericTools.accuracy(y, predicted)}")
  x.each_index { |i| note("#{vector(x[i])} → features #{vector(expanded[i])} class=#{predicted[i]} target=#{y[i]}") }
  raise "nonlinear boundary missing" unless predicted == y && NumericTools.accuracy(y, linear) < 1
  raise "sixth-degree mapping width differs" unless RegularizedLogisticRegression.map_features([[1.0, 2.0]])[0].size == 27

  ok("the same logistic objective learns a circular boundary through explicit features")
end
