# frozen_string_literal: true

# Original C2 W2 optional multiclass lab: 100 examples, four fixed centers,
# random_state 30, std 1; 2→2→4 network, ReLU/logits, Adam(.01), 200 epochs.
require_relative "../lib/style"
require_relative "../lib/reference"
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Four-Class Neural Network", "original optional blob dataset · TensorFlow/Keras")
  x, y = PyCall.import_module("sklearn.datasets").make_blobs(n_samples: 100, centers: [[-5, 2], [-2, -2], [1, 2], [5, -2]], cluster_std: 1.0, random_state: 30).to_a
  Reference.seed(1234)
  tf = Reference.tensorflow
  model = tf.keras.Sequential.new([tf.keras.Input(shape: [2]), tf.keras.layers.Dense.new(2, activation: "relu", name: "L1"), tf.keras.layers.Dense.new(4, activation: "linear", name: "L2")])
  model.compile(loss: tf.keras.losses.SparseCategoricalCrossentropy.new(from_logits: true), optimizer: tf.keras.optimizers.Adam.new(learning_rate: 0.01))
  history = model.fit(x, y, epochs: 200, batch_size: 32, verbose: 0)
  step("architecture 2→2→4; loss #{history.history['loss'].to_a.first} → #{history.history['loss'].to_a.last}")
  accuracy = Reference.accuracy(y, Reference.numpy.argmax(model.predict(x, verbose: 0), axis: 1))
  step("original four-class training accuracy=#{accuracy}")
  raise "Four-class optional model failed" unless accuracy >= 0.85

  ok("the optional four-class example remains distinct from ten-digit recognition")
end
