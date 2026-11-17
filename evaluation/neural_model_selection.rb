# frozen_string_literal: true

# Course C2 W3 assignment, part 2: make_blobs(800, six centers, std 0.4,
# random_state=2), split 50/40/10 (random_state=1). Complex 120→40→6,
# simple 6→6 and L2(0.1) 120→40→6 ReLU networks with linear logits, sparse
# categorical cross-entropy from logits, Adam(0.01), 1000 epochs. The
# notebook fits each graded model twice (graded cell, then unit-test cell),
# then sweeps λ over 0…0.3. Error is the misclassified fraction.
# Run me: ruby evaluation/neural_model_selection.rb
require_relative "../neural_networks/sequential"
require_relative "../lib/sklearn_data"
module NeuralModelSelection
  CENTERS = [[-1, 0], [1, 0], [0, 1], [0, -1], [-2, 1], [-2, -1]].freeze

  module_function

  def build(widths, input_size:, lambda_: 0.0, output: :linear, seed: 1234)
    layers = widths.each_with_index.map do |width, i|
      last = i == widths.size - 1
      [width, last ? output : :relu, { l2: last ? 0.0 : lambda_ }]
    end
    Sequential.new(input_size, layers, rng: Random.new(seed))
  end

  def error(model, x, y)
    predicted = model.logits(x).map { |row| row.index(row.max) }
    predicted.zip(y).count { |p, t| p != t }.to_f / y.size
  end

  def train(widths, x, y, lambda_: 0.0, passes: 1, epochs: 1000, seed: 1234)
    model = build(widths, input_size: 2, lambda_: lambda_, seed: seed)
    model.compile(loss: :sparse_categorical_crossentropy, optimizer: Adam.new(learning_rate: 0.01))
    passes.times { |pass| model.fit(x, y, epochs: epochs, rng: Random.new(seed + pass)) }
    model
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Neural Bias, Variance and Regularization", "course make_blobs sample · six classes")
  x, y = SklearnData.make_blobs(n_samples: 800, centers: NeuralModelSelection::CENTERS, cluster_std: 0.4, random_state: 2)
  train, other, train_y, other_y = SklearnData.train_test_split(x, y, test_size: 0.5, random_state: 1)
  cv, test, cv_y, test_y = SklearnData.train_test_split(other, other_y, test_size: 0.2, random_state: 1)
  step("split #{train.size}/#{cv.size}/#{test.size}")
  models = [["complex", [120, 40, 6], 0.0], ["simple", [6, 6], 0.0], ["regularized", [120, 40, 6], 0.1]].map do |label, widths, lambda_|
    model = NeuralModelSelection.train(widths, train, train_y, lambda_: lambda_, passes: 2)
    step("#{label} #{widths.inspect} λ=#{lambda_}: train error=#{NeuralModelSelection.error(model, train, train_y)}; validation=#{NeuralModelSelection.error(model, cv, cv_y)}")
    [label, model]
  end
  models.each { |label, model| step("#{label}: test error=#{NeuralModelSelection.error(model, test, test_y)}") }
  [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3].each do |lambda_|
    model = NeuralModelSelection.train([120, 40, 6], train, train_y, lambda_: lambda_)
    step("λ=#{lambda_}: train error=#{NeuralModelSelection.error(model, train, train_y)}; validation=#{NeuralModelSelection.error(model, cv, cv_y)}")
  end
  ok("network capacity and L2 strength compared on validation error")
end
