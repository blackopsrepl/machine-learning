# frozen_string_literal: true

# Course C1 W2 lab: scikit-learn's SGDRegressor on standardized housing data.
# This is the estimator's algorithm, ported line by line: squared loss,
# L2 penalty alpha=1e-4 applied as weight decay, inverse-scaling rate
# eta=eta0/t^0.25 with eta0=0.01, one update per shuffled sample, and the
# "no improvement of tol*n in summed epoch loss for 5 epochs" stopping rule.
# Shuffling uses scikit-learn's own xorshift stream from random_state=42.
# Run me: ruby supervised/stochastic_gradient_descent.rb
require_relative "../lib/style"
require_relative "../lib/numeric"
require_relative "../lib/numpy_random"
module SGDRegressor
  MAX_INT = 2_147_483_647

  module_function

  # scikit-learn's our_rand_r: 32-bit xorshift, results modulo 2^31.
  def rand_r(state)
    state = 1 if state.zero?
    state ^= (state << 13) & 0xffff_ffff
    state ^= state >> 17
    state ^= (state << 5) & 0xffff_ffff
    [state, state % (MAX_INT + 1)]
  end

  # SequentialDataset.shuffle: each epoch restarts the same seed on the
  # already permuted index, so successive epochs see different orders.
  def shuffle!(order, seed)
    (0...order.size - 1).each do |i|
      seed, value = rand_r(seed)
      j = i + value % (order.size - i)
      order[i], order[j] = order[j], order[i]
    end
    order
  end

  def fit(x, y, alpha: 1e-4, eta0: 0.01, power_t: 0.25, max_iter: 1000, tol: 1e-3, n_iter_no_change: 5, random_state: 42)
    seed = NumpyRandom::RandomState.new(random_state).interval(MAX_INT - 1)
    weights, intercept, t = Array.new(x.first.size, 0.0), 0.0, 1.0
    order = (0...x.size).to_a
    best_loss, no_improvement, epochs = Float::INFINITY, 0, 0
    max_iter.times do |epoch|
      epochs = epoch + 1
      shuffle!(order, seed)
      summed_loss = 0.0
      order.each do |i|
        prediction = NumericTools.dot(weights, x[i]) + intercept
        eta = eta0 / t**power_t
        summed_loss += 0.5 * (prediction - y[i])**2
        update = -eta * (prediction - y[i]).clamp(-1e12, 1e12)
        decay = [0.0, 1.0 - eta * alpha].max
        weights.each_index { |j| weights[j] = weights[j] * decay + update * x[i][j] }
        intercept += update
        t += 1
      end
      yield epochs, summed_loss / x.size, weights.dup, intercept if block_given?
      no_improvement = summed_loss > best_loss - tol * x.size ? no_improvement + 1 : 0
      best_loss = summed_loss if summed_loss < best_loss
      break if no_improvement >= n_iter_no_change
    end
    { weights: weights, intercept: intercept, epochs: epochs, updates: t }
  end

  def predict(model, row) = NumericTools.dot(model[:weights], row) + model[:intercept]
end
if __FILE__ == $PROGRAM_NAME
  require_relative "../lib/course_data"
  include Style
  banner("SGDRegressor", "course housing data · scikit-learn's algorithm in Ruby")
  x, y = CourseData.supervised(:houses, "data/houses.txt", skip: 1)
  scaler = NumericTools.fit_scaler(x)
  normalized = NumericTools.transform(x, scaler)
  step("#{x.size} houses; peak-to-peak raw #{vector(x.transpose.map { |c| c.max - c.min })}; normalized #{vector(normalized.transpose.map { |c| c.max - c.min })}")
  model = SGDRegressor.fit(normalized, y) do |epoch, mean_loss, _w, _b|
    step("epoch #{epoch}: mean half squared error=#{mean_loss}") if epoch <= 3 || (epoch % 10).zero?
  end
  step("iterations=#{model[:epochs]}; updates=#{model[:updates]}; weights #{vector(model[:weights])}; intercept #{model[:intercept]}")
  predicted = normalized.map { |row| SGDRegressor.predict(model, row) }
  step("first predictions #{vector(predicted.first(4))}; targets #{vector(y.first(4))}")
  raise "SGD fit is not finite" unless predicted.all?(&:finite?)

  ok("trained on the housing records with scikit-learn's update, decay and stopping rules")
end
