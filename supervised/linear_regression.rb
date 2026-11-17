# frozen_string_literal: true

# Linear regression — y_hat = w·x + b, J = Σ(y_hat-y)²/(2m).
# Batch gradients average residual*x and residual; update all parameters together.
# The exact line y=1+2x makes convergence and the intercept visible by eye.
# Run me: ruby supervised/linear_regression.rb
require_relative "../lib/style"
require_relative "../lib/problem"
require_relative "../lib/numeric"

module LinearRegression
  module_function
  def predict(row, weights, bias) = NumericTools.dot(row, weights) + bias
  def loss(x, y, weights, bias, l2: 0.0)
    NumericTools.mse(y, x.map { |r| predict(r, weights, bias) }) / 2 + l2 * weights.sum { |w| w*w } / (2*x.size)
  end

  def gradients(x, y, weights, bias, l2: 0.0)
    residuals = x.zip(y).map { |r, target| predict(r, weights, bias) - target }
    dw = weights.each_index.map { |j| (x.each_index.sum { |i| residuals[i]*x[i][j] } + l2*weights[j]).to_f / x.size }
    [dw, NumericTools.mean(residuals)]
  end

  def fit(x, y, rate: 0.05, iterations: 1000, l2: 0.0)
    weights, bias = Array.new(x.first.size, 0.0), 0.0
    iterations.times do |i|
      dw, db = gradients(x, y, weights, bias, l2: l2)
      weights = weights.zip(dw).map { |w, g| w-rate*g }
      bias -= rate*db
      yield i+1, weights.dup, bias, loss(x, y, weights, bias, l2: l2), dw, db if block_given?
    end
    [weights, bias]
  end
end

if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Linear Regression", "batch gradient descent · exact line")
  x, y = Problem.line, Problem.line_targets
  w, b = LinearRegression.fit(x, y) do |i, weights, bias, cost, dw, db|
    step(format("it %-4d w=% .4f b=% .4f J=%.6f  gradients=(%.4f, %.4f)", i, weights[0], bias, cost, dw[0], db)) if i <= 3 || (i % 250).zero?
  end
  x.zip(y).each { |r,t| note(format("x=%.0f predicted=%.4f target=%.0f", r[0], LinearRegression.predict(r,w,b), t)) }
  raise "line not learned" unless LinearRegression.loss(x,y,w,b) < 1e-8
  # Same gradient works with multiple features; no alternate estimator is hidden.
  multi = [[0.0,0.0], [1.0,0.0], [0.0,1.0], [1.0,1.0]]
  mw, mb = LinearRegression.fit(multi, [1.0,3.0,4.0,6.0], iterations: 3000)
  raise "multiple regression failed" unless LinearRegression.loss(multi,[1,3,4,6],mw,mb) < 1e-8
  step("two features: weights #{vector(mw)}, bias #{format('%.4f',mb)}")
  ok("one and multiple features recover their known coefficients")
end
