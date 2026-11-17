# frozen_string_literal: true

# Course C2 W2 optional lab: make_blobs(100, centers (-5,2),(-2,-2),(1,2),(5,-2),
# std 1, random_state 30); 2→2→4 network, ReLU then linear logits, sparse
# categorical cross-entropy from logits, Adam(0.01), 200 epochs.
# Run me: ruby neural_networks/multiclass_blobs.rb
require_relative "sequential"
require_relative "../lib/sklearn_data"
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Four-Class Neural Network", "course make_blobs sample · 2→2→4")
  x, y = SklearnData.make_blobs(n_samples: 100, centers: [[-5, 2], [-2, -2], [1, 2], [5, -2]], cluster_std: 1.0, random_state: 30)
  step("class representation #{y.first(10).inspect}; shape #{x.size}×#{x.first.size}")
  model = Sequential.new(2, [[2, :relu], [4, :linear]], rng: Random.new(1234))
  model.compile(loss: :sparse_categorical_crossentropy, optimizer: Adam.new(learning_rate: 0.01))
  history = model.fit(x, y, epochs: 200) { |epoch, loss| step("epoch #{epoch}: loss=#{loss}") if epoch <= 3 || (epoch % 50).zero? }
  hidden, output = model.layers
  step("L1 weights #{vector(hidden.kernel)} biases #{vector(hidden.bias)}")
  step("L2 weights #{vector(output.kernel)} biases #{vector(output.bias)}")
  predicted = model.logits(x).map { |row| row.index(row.max) }
  step("training accuracy=#{predicted.zip(y).count { |p, t| p == t } / 100.0}")
  raise "training did not reduce the loss" unless history.last < history.first

  ok("two ReLU units carve the plane for four linear output units")
end
