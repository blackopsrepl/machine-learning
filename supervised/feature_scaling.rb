# frozen_string_literal: true

# Feature scaling — z_j=(x_j-mean_j)/population_std_j.
# Fit statistics on training rows only; transform held-out rows unchanged.
# Standardization improves conditioning, not the information in the features.
# Run me: ruby supervised/feature_scaling.rb
require_relative "linear_regression"
if __FILE__ == $PROGRAM_NAME
  require_relative "../lib/course_data"
  include Style
  banner("Feature Scaling", "course housing features · handwritten z-score normalization")
  rows = CourseData.table(:houses, "data/houses.txt", skip: 1)
  x = rows.map { |row| row.first(4) }
  y = rows.map(&:last)
  scaler = NumericTools.fit_scaler(x)
  scaled = NumericTools.transform(x, scaler)
  step("#{x.size} houses; means #{vector(scaler[0])}; standard deviations #{vector(scaler[1])}")
  [9.9e-7, 9e-7, 1e-7].each do |rate|
    w, b = LinearRegression.fit(x, y, rate: rate, iterations: 10)
    step("raw learning rate #{rate}: cost #{LinearRegression.loss(x, y, w, b)}")
  end
  w, b = LinearRegression.fit(scaled, y, rate: 0.1, iterations: 1000)
  step("normalized weights #{vector(w)} bias #{b}; cost #{LinearRegression.loss(scaled, y, w, b)}")
  raise "Housing normalization failed" unless w.all?(&:finite?) && LinearRegression.loss(scaled, y, w, b) < 1000

  ok("the course housing columns share one fitted training coordinate system")
end
