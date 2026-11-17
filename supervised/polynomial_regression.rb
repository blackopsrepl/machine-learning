# frozen_string_literal: true

# Original feature-engineering lab: x=0..19, quadratic targets and cos(x/2).
# Handwritten gradient descent retains the source feature sets, normalization,
# learning rates and iteration budgets, including the degree-13 cosine example.
require_relative "linear_regression"
module PolynomialRegression
  module_function

  def features(values, degree)
    raise ArgumentError, "positive degree required" unless degree.positive?

    values.map { |x| (1..degree).map { |power| x.to_f**power } }
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Polynomial Regression", "original feature-engineering lab targets and iteration budgets")
  values = (0...20).map(&:to_f)
  cases = [
    ["raw x, target 1+x²", values.map { |x| [x] }, values.map { |x| 1 + x * x }, 1000, 1e-2],
    ["engineered x²", values.map { |x| [x * x] }, values.map { |x| 1 + x * x }, 10_000, 1e-5],
    ["raw x,x²,x³", PolynomialRegression.features(values, 3), values.map { |x| x * x }, 10_000, 1e-7],
    ["scaled x,x²,x³", nil, values.map { |x| x * x }, 100_000, 0.1],
    ["scaled degree 13, target cos(x/2)", nil, values.map { |x| Math.cos(x / 2) }, 1_000_000, 0.1]
  ]
  cases.each_with_index do |(label, x, y, budget, rate), index|
    unless x
      features = PolynomialRegression.features(values, index == 3 ? 3 : 13)
      x = NumericTools.transform(features, NumericTools.fit_scaler(features))
    end
    iterations = ENV.fetch("ML_ITERATIONS", budget).to_i
    step("#{label}: #{iterations} iterations, alpha=#{rate}")
    w, b = LinearRegression.fit(x, y, rate: rate, iterations: iterations, trace_every: [iterations / 5, 1].max) { |i, _w, _b, cost, _dw, _db| note("iteration #{i}: cost=#{cost}") }
    raise "Polynomial fit became nonfinite" unless w.all?(&:finite?) && b.finite?

    step("final cost=#{LinearRegression.loss(x, y, w, b)}; bias=#{b}; weights=#{vector(w)}")
  end
  ok("the original polynomial and cosine exercises retain their data and budgets")
end
