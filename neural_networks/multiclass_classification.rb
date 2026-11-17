# frozen_string_literal: true

# Course C2 W2: ten handwritten digits, 400→25→15→10, ReLU hidden layers.
# Linear output logits, sparse categorical cross-entropy(from_logits=True),
# Adam(0.001), 40 epochs and batches of 32; no class-count reduction.
require_relative "../lib/style"
require_relative "../lib/reference"
module MulticlassClassification
  module_function

  def build
    tf = Reference.tensorflow
    tf.keras.Sequential.new([
                              tf.keras.Input(shape: [400]),
                              tf.keras.layers.Dense.new(25, activation: "relu"),
                              tf.keras.layers.Dense.new(15, activation: "relu"),
                              tf.keras.layers.Dense.new(10, activation: "linear")
                            ])
  end

  def train(epochs: 40)
    Reference.seed(1234)
    x, y = Reference.digits(:multiclass)
    model = build
    tf = Reference.tensorflow
    model.compile(loss: tf.keras.losses.SparseCategoricalCrossentropy.new(from_logits: true), optimizer: tf.keras.optimizers.Adam.new(learning_rate: 0.001))
    history = model.fit(x, y, epochs: epochs, batch_size: 32, shuffle: true, verbose: 0)
    [model, history, x, y]
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Multiclass Digit Recognition", "original 5000 × 400 pixels · ten classes · TensorFlow/Keras")
  model, history, x, y = MulticlassClassification.train
  Reference.epoch_trace(history)
  logits = model.predict(x, verbose: 0)
  prediction = Reference.numpy.argmax(logits, axis: 1)
  accuracy = Reference.accuracy(y, prediction)
  step("architecture 400→25→15→10; parameters=#{model.count_params}; training accuracy=#{accuracy}")
  step("first ten predictions #{Reference.list(prediction).first(10)}")
  raise "Ten-digit recognition did not train" unless accuracy >= 0.9 && Reference.shape(logits) == [5000, 10]

  Dir.mkdir("output") unless Dir.exist?("output")
  model.save("output/multiclass_digits.keras")
  ok("all ten original digit classes remain in the application")
end
