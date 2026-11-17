# frozen_string_literal: true

# Course C2 W1: recognize handwritten zero/one digits from their 400 pixels.
# Preserve 400→25→15→1 sigmoid layers, BCE, Adam(0.001), 20 epochs, batches of 32.
# This is the library-backed training portion; dense_layer.rb exposes Ruby inference.
require_relative "../lib/style"
require_relative "../lib/reference"
module BinaryClassification
  module_function

  def build
    tf = Reference.tensorflow
    tf.keras.Sequential.new([
                              tf.keras.Input(shape: [400]),
                              tf.keras.layers.Dense.new(25, activation: "sigmoid"),
                              tf.keras.layers.Dense.new(15, activation: "sigmoid"),
                              tf.keras.layers.Dense.new(1, activation: "sigmoid")
                            ])
  end

  def train(epochs: 20)
    Reference.seed(1234)
    x, y = Reference.digits(:binary_digits)
    model = build
    tf = Reference.tensorflow
    model.compile(loss: tf.keras.losses.BinaryCrossentropy.new, optimizer: tf.keras.optimizers.Adam.new(learning_rate: 0.001))
    history = model.fit(x, y, epochs: epochs, batch_size: 32, shuffle: true, verbose: 0)
    [model, history, x, y]
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Binary Digit Recognition", "original 1000 × 400 pixels · TensorFlow/Keras")
  model, history, x, y = BinaryClassification.train
  Reference.epoch_trace(history)
  prediction = model.predict(x, verbose: 0)
  labels = Reference.numpy.greater_equal(prediction, 0.5).astype("int32")
  accuracy = Reference.accuracy(y, labels)
  step("architecture 400→25→15→1; parameters=#{model.count_params}; training accuracy=#{accuracy}")
  step("first ten probabilities #{Reference.list(prediction.reshape([-1])).first(10)}")
  raise "Digit recognition did not train" unless accuracy >= 0.9 && model.count_params == 10431

  Dir.mkdir("output") unless Dir.exist?("output")
  model.save("output/binary_digits.keras")
  ok("zero/one recognition trained on the original digit images")
end
