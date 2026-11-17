# frozen_string_literal: true

# Binary logistic regression — p=σ(w·x+b), dJ/dz=p-y.
# Stable loss softplus(z)-y*z avoids log(0); bias is excluded from L2.
# A threshold acts on probability, not on the training loss.
# Run me: ruby supervised/logistic_regression.rb
require_relative "../lib/style"
require_relative "../lib/numeric"
module LogisticRegression
  module_function

  def sigmoid(z)
    z >= 0 ? 1.0 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z))
  end

  def softplus(z) = [z, 0.0].max + Math.log(1 + Math.exp(-z.abs))
  def probability(row, w, b) = sigmoid(NumericTools.dot(row, w) + b)

  def loss(x, y, w, b, l2: 0.0)
    total = x.zip(y).sum do |row, target|
      z = NumericTools.dot(row, w) + b
      softplus(z) - target * z
    end
    (total + l2 * w.sum { |v| v * v } / 2).to_f / x.size
  end

  def gradients(x, y, w, b, l2: 0.0)
    errors = x.zip(y).map { |row, target| probability(row, w, b) - target }
    dw = w.each_index.map { |j| (x.each_index.sum { |i| errors[i] * x[i][j] } + l2 * w[j]).to_f / x.size }
    [dw, NumericTools.mean(errors)]
  end

  def fit(x, y, rate: 0.2, iterations: 800, l2: 0.0, initial_weights: nil, initial_bias: 0.0)
    w, b = initial_weights ? initial_weights.dup : Array.new(x[0].size, 0.0), initial_bias
    iterations.times do |i|
      dw, db = gradients(x, y, w, b, l2: l2)
      w = w.zip(dw).map { |v, g| v - rate * g }
      b -= rate * db
      yield i + 1, w.dup, b, loss(x, y, w, b, l2: l2) if block_given?
    end
    [w, b]
  end
end
if __FILE__ == $PROGRAM_NAME
  require_relative "../lib/course_data"
  include Style
  banner("Logistic Regression", "course exam scores → admission · handwritten")
  x, y = CourseData.supervised(:logistic, "data/ex2data1.txt")
  rng = Random.new(1)
  initial = Array.new(2) { 0.01 * (rng.rand - 0.5) }
  w, b = LogisticRegression.fit(x, y, rate: 0.001, iterations: 10_000, initial_weights: initial, initial_bias: -8.0) do |i, weights, bias, cost|
    step("iteration #{i}: cost=#{cost} weights=#{weights.inspect} bias=#{bias}") if i <= 3 || (i % 2000).zero?
  end
  predicted = x.map { |row| LogisticRegression.probability(row, w, b) >= 0.5 ? 1 : 0 }
  accuracy = NumericTools.accuracy(y, predicted)
  step("training accuracy #{accuracy}; boundary #{w[0]}*exam1 + #{w[1]}*exam2 + #{b} = 0")
  raise "Admissions model failed" unless accuracy >= 0.85

  ok("trained on all #{x.size} original admissions records")
end
