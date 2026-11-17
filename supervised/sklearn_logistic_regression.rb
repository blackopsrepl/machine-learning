# frozen_string_literal: true

# Course C1 W3 lab: scikit-learn's LogisticRegression with its defaults —
# minimize ½||w||² + C·Σ log-loss with C=1 and an unpenalized intercept.
# Newton's method on that strictly convex objective reaches the same unique
# optimum that scikit-learn's L-BFGS solver approximates.
# Run me: ruby supervised/sklearn_logistic_regression.rb
require_relative "logistic_regression"
require_relative "normal_equation"
module PenalizedLogisticRegression
  module_function

  def objective(x, y, w, b, c)
    0.5 * w.sum { |v| v * v } + c * x.zip(y).sum do |row, target|
      z = NumericTools.dot(row, w) + b
      LogisticRegression.softplus(z) - target * z
    end
  end

  def fit(x, y, c: 1.0, tolerance: 1e-12, max_iterations: 100)
    n = x.first.size
    theta = Array.new(n + 1, 0.0)
    max_iterations.times do |iteration|
      w, b = theta.first(n), theta.last
      gradient = Array.new(n + 1, 0.0)
      hessian = Array.new(n + 1) { Array.new(n + 1, 0.0) }
      x.zip(y).each do |row, target|
        p = LogisticRegression.probability(row, w, b)
        features = row + [1.0]
        features.each_with_index do |fi, i|
          gradient[i] += c * (p - target) * fi
          features.each_with_index { |fj, j| hessian[i][j] += c * p * (1 - p) * fi * fj }
        end
      end
      n.times do |i|
        gradient[i] += w[i]
        hessian[i][i] += 1.0
      end
      step = OrdinaryLeastSquares.solve(hessian, gradient)
      theta = theta.zip(step).map { |value, delta| value - delta }
      norm = Math.sqrt(gradient.sum { |g| g * g })
      yield iteration + 1, theta.first(n), theta.last, norm if block_given?
      break if norm < tolerance
    end
    [theta.first(n), theta.last]
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Scikit-Learn Logistic Regression", "course optional-lab observations · C=1 objective in Ruby")
  x = [[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]]
  y = [0, 0, 0, 1, 1, 1]
  w, b = PenalizedLogisticRegression.fit(x, y) do |iteration, weights, bias, norm|
    step("Newton step #{iteration}: weights #{vector(weights)} intercept #{bias}; gradient norm #{norm}")
  end
  predictions = x.map { |row| LogisticRegression.probability(row, w, b) >= 0.5 ? 1 : 0 }
  accuracy = NumericTools.accuracy(y, predictions)
  step("predictions #{predictions.inspect}; accuracy=#{accuracy}; objective=#{PenalizedLogisticRegression.objective(x, y, w, b, 1.0)}")
  raise "the course's six observations are not all classified" unless accuracy == 1.0

  ok("the estimator's penalized objective is solved explicitly")
end
