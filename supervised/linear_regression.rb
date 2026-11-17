# frozen_string_literal: true

# Linear regression — y_hat=w·x+b, half mean squared error and simultaneous
# batch-gradient updates. The demonstration uses the course's 97 restaurant
# population/profit observations, alpha .01 and 1500 iterations.
require_relative "../lib/style"
require_relative "../lib/numeric"

module LinearRegression
  module_function

  def predict(row, weights, bias) = NumericTools.dot(row, weights) + bias

  def loss(x, y, weights, bias, l2: 0.0)
    NumericTools.mse(y, x.map { |r| predict(r, weights, bias) }) / 2 + l2 * weights.sum { |w| w * w } / (2 * x.size)
  end

  def gradients(x, y, weights, bias, l2: 0.0)
    residuals = x.zip(y).map { |r, target| predict(r, weights, bias) - target }
    dw = weights.each_index.map { |j| (x.each_index.sum { |i| residuals[i] * x[i][j] } + l2 * weights[j]).to_f / x.size }
    [dw, NumericTools.mean(residuals)]
  end

  def fit(x, y, rate: 0.05, iterations: 1000, l2: 0.0, trace_every: 1)
    weights, bias = Array.new(x.first.size, 0.0), 0.0
    iterations.times do |i|
      dw, db = gradients(x, y, weights, bias, l2: l2)
      weights = weights.zip(dw).map { |w, g| w - rate * g }
      bias -= rate * db
      yield i + 1, weights.dup, bias, loss(x, y, weights, bias, l2: l2), dw, db if block_given? && ((i % trace_every).zero? || i == iterations - 1)
    end
    [weights, bias]
  end
end

if __FILE__ == $PROGRAM_NAME
  require_relative "../lib/course_data"
  include Style
  banner("Linear Regression", "course restaurant profits · handwritten batch GD")
  x, y = CourseData.supervised(:linear, "data/ex1data1.txt")
  costs = [LinearRegression.loss(x, y, [0.0], 0.0)]
  w, b = LinearRegression.fit(x, y, rate: 0.01, iterations: 1500) do |i, weights, bias, cost, _dw, _db|
    costs << cost
    step("iteration #{i}: cost=#{cost} w=#{weights.inspect} b=#{bias}") if i <= 3 || (i % 300).zero?
  end
  [3.5, 7.0].each { |population| step("population #{population * 10_000}: predicted profit $#{(LinearRegression.predict([population], w, b) * 10_000).round(2)}") }
  raise "gradient descent increased the cost" unless costs.each_cons(2).all? { |a, c| c <= a + 1e-12 }

  ok("trained on all #{x.size} observations")
end
