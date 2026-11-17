# frozen_string_literal: true

# L2 regularization — J_data + λ||w||²/(2m); gradient adds λw/m, never λb/m.
# Compare training fit with shrinkage. Smaller training error is not by itself
# better generalization; held-out model selection is a separate example.
# Run me: ruby supervised/regularization.rb
require_relative "normal_equation"
require_relative "logistic_regression"
if __FILE__ == $PROGRAM_NAME
  require_relative "../lib/course_data"
  include Style
  banner("Regularization", "original housing and microchip data · handwritten L2 objectives")
  rows = CourseData.table(:houses, "data/houses.txt").drop(1)
  x = rows.map { |r| r.first(4) }
  y = rows.map(&:last)
  x = NumericTools.transform(x, NumericTools.fit_scaler(x))
  [0.0, 1.0, 10.0].each do |lambda|
    w, b = LinearRegression.fit(x, y, rate: 0.1, iterations: 1000, l2: lambda)
    step("housing lambda=#{lambda}: data cost=#{LinearRegression.loss(x, y, w, b)}; weights=#{vector(w)}")
  end
  chips, labels = CourseData.supervised(:logistic, "data/ex2data2.txt")
  require_relative "regularized_logistic_regression"
  expanded = RegularizedLogisticRegression.map_features(chips)
  [0.0, 1.0, 10.0].each do |lambda|
    w, b = LogisticRegression.fit(expanded, labels, iterations: 1000, rate: 0.01, l2: lambda)
    step("microchip lambda=#{lambda}: objective=#{LogisticRegression.loss(expanded, labels, w, b, l2: lambda)}")
  end
  ok("linear and logistic penalties operate on the original feature matrices")
end
