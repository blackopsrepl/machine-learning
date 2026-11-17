# frozen_string_literal: true

# Central differences (L(θ+ε)−L(θ−ε))/2ε against the training engine's own
# backpropagation, for each loss the course uses and for L2 kernels.
# Run me: ruby neural_networks/gradient_check.rb
require_relative "sequential"
module GradientCheck
  module_function

  def batch_loss(model, x, y)
    activations, logits = model.forward(x)
    loss, delta = model.loss_and_delta(activations.last, logits.last, y, x.size)
    [loss + model.regularization, activations, logits, delta]
  end

  # Maximum |analytic − numerical| over the selected (tensor, index) pairs.
  def check(model, x, y, picks, epsilon: 1e-6)
    _loss, activations, logits, delta = batch_loss(model, x, y)
    analytic, = model.backward(activations, logits, delta)
    picks.map do |t, i|
      tensor = model.tensors[t]
      original = tensor[i]
      tensor[i] = original + epsilon
      plus = batch_loss(model, x, y).first
      tensor[i] = original - epsilon
      minus = batch_loss(model, x, y).first
      tensor[i] = original
      (analytic[t][i] - (plus - minus) / (2 * epsilon)).abs
    end.max
  end
end
if __FILE__ == $PROGRAM_NAME
  require_relative "../lib/course_data"
  include Style
  banner("Gradient Checking", "finite differences against the engine's backpropagation")
  rng = Random.new(42)
  cases = [
    ["binary digits, BCE", :binary, [[25, :sigmoid], [15, :sigmoid], [1, :sigmoid]], :binary_crossentropy],
    ["ten digits, sparse CE from logits", :multiclass, [[25, :relu], [15, :relu], [10, :linear]], :sparse_categorical_crossentropy],
    ["L2(0.1) ReLU regression, MSE", :binary, [[12, :relu, { l2: 0.1 }], [3, :linear]], :mse]
  ]
  cases.each do |label, kind, layers, loss|
    x, y = CourseData.digits(kind)
    rows = (0...x.size).to_a.sample(8, random: rng)
    targets = loss == :mse ? rows.map { |i| [x[i].sum / 100, x[i][210], y[i]] } : rows.map { |i| y[i] }
    model = Sequential.new(400, layers, rng: Random.new(7)).compile(loss: loss, optimizer: Adam.new)
    picks = model.tensors.each_with_index.flat_map { |tensor, t| (0...tensor.size).to_a.sample(6, random: rng).map { |i| [t, i] } }
    error = GradientCheck.check(model, rows.map { |i| x[i] }, targets, picks)
    step("#{label}: #{picks.size} parameters across #{model.tensors.size} tensors; max |analytic−numerical|=#{error}")
    raise "#{label}: backpropagation disagrees with finite differences" unless error < 1e-7
  end
  ok("every loss and the L2 penalty differentiate correctly through every layer")
end
