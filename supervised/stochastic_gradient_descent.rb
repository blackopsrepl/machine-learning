# frozen_string_literal: true

# Stochastic gradient descent — one residual updates the model at a time.
# Unlike batch GD, each shuffled example sees the weights left by the previous.
# A decaying step size reduces late oscillation. No SGDRegressor is involved.
# Run me: ruby supervised/stochastic_gradient_descent.rb
require_relative "linear_regression"
module StochasticGradientDescent
  module_function
  def fit(x, y, epochs: 300, rate: 0.05, rng: Random.new(42))
    w, b, updates = Array.new(x.first.size,0.0), 0.0, 0
    epochs.times do |epoch|
      x.each_index.to_a.shuffle(random: rng).each do |i|
        error = LinearRegression.predict(x[i],w,b)-y[i]
        alpha = rate/(1+0.001*updates)
        w = w.each_index.map { |j| w[j]-alpha*error*x[i][j] }
        b -= alpha*error
        updates += 1
        yield epoch+1, i, error, w.dup, b if block_given?
      end
    end
    [w,b]
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Stochastic Gradient Descent", "one example per update · Random.new(42)")
  w,b = StochasticGradientDescent.fit(Problem.line,Problem.line_targets) do |epoch,i,error,weights,bias|
    step(format("epoch %d row %d residual=% .4f w=%.4f b=%.4f",epoch,i,error,weights[0],bias)) if epoch <= 2
  end
  cost = LinearRegression.loss(Problem.line,Problem.line_targets,w,b)
  step(format("final w=%.5f b=%.5f J=%.8f",w[0],b,cost))
  raise "SGD failed" unless cost < 1e-5
  ok("sequential updates learn the same line as batch GD")
end
