# frozen_string_literal: true

# Course C2 W1 coffee-roasting labs: 200 (temperature, duration) roasts from
# NumPy's default_rng(2), good when 175<t<260, 12<d<15 and d≤21−3t/85.
# Keras Normalization: (x−mean)/max(√variance, 1e-7) with population variance.
# The TF lab tiles the data 1000×, trains 2→3→1 sigmoid units with BCE and
# Adam(0.01) for 10 epochs, then predicts with the weights the notebook supplies.
# Run me: ruby neural_networks/coffee_roasting.rb
require_relative "sequential"
require_relative "../lib/numpy_random"
module CoffeeRoasting
  # Weights the TF notebook loads after training, so results do not depend on the run.
  W1 = [[-8.94, 0.29, 12.89], [-0.17, -7.34, 10.79]].freeze
  B1 = [-9.87, -9.28, 1.01].freeze
  W2 = [[-31.38], [-27.86], [-32.79]].freeze
  B2 = [15.54].freeze

  module_function

  def data
    rng = NumpyRandom::Generator.new(2)
    x = Array.new(200) { [rng.random, rng.random] }
    x.each do |row|
      row[1] = row[1] * 4 + 11.5
      row[0] = row[0] * (285 - 150) + 150
    end
    y = x.map do |t, d|
      limit = -3.0 / (260 - 175) * t + 21
      t > 175 && t < 260 && d > 12 && d < 15 && d <= limit ? 1 : 0
    end
    [x, y]
  end

  def normalization(x)
    means = x.transpose.map { |column| column.sum / column.size }
    scales = x.transpose.each_with_index.map do |column, j|
      [Math.sqrt(column.sum { |v| (v - means[j])**2 } / column.size), 1e-7].max
    end
    ->(rows) { rows.map { |row| row.each_with_index.map { |v, j| (v - means[j]) / scales[j] } } }
  end

  # Keras stores kernels as inputs×units; Sequential stores units×inputs.
  def keras_weights(w, b) = [w.transpose.flatten, b.dup]
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Coffee Roasting", "course data generator · normalization · 2→3→1 sigmoid network")
  x, y = CoffeeRoasting.data
  step("#{x.size} roasts, #{y.sum} good; temperature #{x.map(&:first).minmax.map { |v| v.round(2) }}, duration #{x.map(&:last).minmax.map { |v| v.round(2) }}")
  normalize = CoffeeRoasting.normalization(x)
  xn = normalize.call(x)
  step("normalized temperature #{xn.map(&:first).minmax.map { |v| v.round(2) }}, duration #{xn.map(&:last).minmax.map { |v| v.round(2) }}")
  xt, yt = xn * 1000, y * 1000
  model = Sequential.new(2, [[3, :sigmoid], [1, :sigmoid]], rng: Random.new(1234))
  model.compile(loss: :binary_crossentropy, optimizer: Adam.new(learning_rate: 0.01))
  step("trained on #{xt.size} tiled rows; #{model.count_params} parameters")
  model.fit(xt, yt, epochs: 10) { |epoch, loss| step("epoch #{epoch}: loss=#{loss}") }
  trained = model.predict(xn).map(&:first)
  step("trained-model accuracy on the 200 roasts=#{trained.zip(y).count { |p, t| (p >= 0.5 ? 1 : 0) == t } / 200.0}")
  model.weights = [CoffeeRoasting.keras_weights(CoffeeRoasting::W1, CoffeeRoasting::B1), CoffeeRoasting.keras_weights(CoffeeRoasting::W2, CoffeeRoasting::B2)]
  test = normalize.call([[200, 13.9], [200, 17]])
  predictions = model.predict(test).map(&:first)
  step("supplied weights: P(good | 200°, 13.9 min)=#{predictions[0]}; P(good | 200°, 17 min)=#{predictions[1]}")
  raise "supplied weights must accept the first roast and reject the second" unless predictions[0] >= 0.5 && predictions[1] < 0.5

  ok("the course's generated roasts, normalization, training and supplied-weight decisions")
end
