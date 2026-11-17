# frozen_string_literal: true

# Original C2 W3 advice: 800 samples, six stated centers, std .4, random_state 2.
# Compare 120→40→6, 6→6 and L2(.1) 120→40→6; retain the seven-lambda sweep.
require_relative "../lib/style"
require_relative "../lib/reference"
module NeuralModelSelection
  module_function

  def build(widths, input_size:, lambda_: 0.0)
    tf = Reference.tensorflow
    layers = [tf.keras.Input(shape: [input_size])]
    widths.each_with_index do |width, index|
      options = { activation: index == widths.size - 1 ? "linear" : "relu" }
      options[:kernel_regularizer] = tf.keras.regularizers.L2.new(lambda_) if lambda_ > 0 && index < widths.size - 1
      layers << tf.keras.layers.Dense.new(width, **options)
    end
    tf.keras.Sequential.new(layers)
  end

  def error(model, x, y)
    logits = model.predict(x, verbose: 0)
    1.0 - Reference.accuracy(y, Reference.numpy.argmax(logits, axis: 1))
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Neural Bias, Variance and Regularization", "original 800 points · six classes · TensorFlow")
  tf = Reference.tensorflow
  tf.keras.backend.set_floatx("float64")
  centers = [[-1, 0], [1, 0], [0, 1], [0, -1], [-2, 1], [-2, -1]]
  x, y = PyCall.import_module("sklearn.datasets").make_blobs(n_samples: 800, centers: centers, cluster_std: 0.4, random_state: 2, n_features: 2).to_a
  splitter = PyCall.import_module("sklearn.model_selection")
  train, other, train_y, other_y = splitter.train_test_split(x, y, test_size: 0.5, random_state: 1).to_a
  cv, test, cv_y, test_y = splitter.train_test_split(other, other_y, test_size: 0.2, random_state: 1).to_a
  epochs = Integer(ENV.fetch("ML_EPOCHS", 1000))
  configurations = [["complex", [120, 40, 6], 0.0, 2], ["simple", [6, 6], 0.0, 2], ["regularized", [120, 40, 6], 0.1, 2]]
  choices = configurations.map do |label, widths, lambda_, passes|
    Reference.seed(1234)
    model = NeuralModelSelection.build(widths, input_size: 2, lambda_: lambda_)
    model.compile(loss: tf.keras.losses.SparseCategoricalCrossentropy.new(from_logits: true), optimizer: tf.keras.optimizers.Adam.new(learning_rate: 0.01))
    passes.times { model.fit(train, train_y, epochs: epochs, batch_size: 32, verbose: 0) }
    train_error = NeuralModelSelection.error(model, train, train_y)
    cv_error = NeuralModelSelection.error(model, cv, cv_y)
    step("#{label} #{widths.inspect}: #{passes}×#{epochs} epochs; train error=#{train_error}; validation=#{cv_error}")
    [cv_error, label, model]
  end
  choices.each { |_err, label, model| step("#{label}: held-out test error=#{NeuralModelSelection.error(model, test, test_y)}") }
  Reference.seed(1234)
  [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3].each do |lambda_|
    model = NeuralModelSelection.build([120, 40, 6], input_size: 2, lambda_: lambda_)
    model.compile(loss: tf.keras.losses.SparseCategoricalCrossentropy.new(from_logits: true), optimizer: tf.keras.optimizers.Adam.new(learning_rate: 0.01))
    model.fit(train, train_y, epochs: epochs, batch_size: 32, verbose: 0)
    step("lambda=#{lambda_}: train error=#{NeuralModelSelection.error(model, train, train_y)}; validation=#{NeuralModelSelection.error(model, cv, cv_y)}")
  end
  ok("all original network capacities and regularization candidates remain available")
end
