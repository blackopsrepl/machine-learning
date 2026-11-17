# frozen_string_literal: true

# Original microchip-quality assignment: all 118 measurements, monomials through
# degree six, lambda .01, alpha .01 and 10000 handwritten gradient updates.
# The bias remains separate and unpenalized.
require_relative "logistic_regression"
module RegularizedLogisticRegression
  module_function

  def map_features(rows, degree: 6)
    rows.map { |a, b| (1..degree).flat_map { |power| (0..power).map { |j| a**(power - j) * b**j } } }
  end
end
if __FILE__ == $PROGRAM_NAME
  require_relative "../lib/course_data"
  include Style
  banner("Regularized Logistic Regression", "course microchip tests · degree six · handwritten")
  raw, y = CourseData.supervised(:logistic, "data/ex2data2.txt")
  x = RegularizedLogisticRegression.map_features(raw, degree: 6)
  rng = Random.new(1)
  initial = Array.new(x.first.size) { rng.rand - 0.5 }
  w, b = LogisticRegression.fit(x, y, rate: 0.01, iterations: 10_000, l2: 0.01, initial_weights: initial, initial_bias: 1.0) do |i, _weights, _bias, cost|
    step("iteration #{i}: regularized cost=#{cost}") if i <= 3 || (i % 2000).zero?
  end
  predicted = x.map { |row| LogisticRegression.probability(row, w, b) >= 0.5 ? 1 : 0 }
  accuracy = NumericTools.accuracy(y, predicted)
  step("#{raw.size} microchips; #{w.size} mapped features plus bias; training accuracy #{accuracy}")
  raise "Microchip classifier failed" unless accuracy >= 0.75

  ok("sixth-degree model fitted to the original test measurements")
end
