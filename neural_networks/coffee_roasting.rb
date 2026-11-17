# frozen_string_literal: true

# Original coffee generator: PCG64 seed 2, 200 temperature/duration pairs.
# Preserve normalization, 1000-fold tiling, 2→3→1 sigmoid model, Adam(.01),
# ten epochs, and the notebook's supplied prediction weights after training.
require_relative "dense_layer"
require_relative "../lib/reference"
module CoffeeRoasting
  module_function

  def data
    np = Reference.numpy
    x = np.random.default_rng(2).random(400).reshape([-1, 2])
    x.__setitem__(PyCall.tuple([PyCall::Slice.new(nil), 0]), np.add(np.multiply(Reference.columns(x, 0, 1).reshape([-1]), 135), 150))
    x.__setitem__(PyCall.tuple([PyCall::Slice.new(nil), 1]), np.add(np.multiply(Reference.columns(x, 1, 2).reshape([-1]), 4), 11.5))
    labels = Reference.list(x).map do |row|
      t, d = row.to_a
      t > 175 && t < 260 && d > 12 && d < 15 && d <= (-3.0 / 85 * t + 21) ? 1.0 : 0.0
    end
    [x, np.asarray(labels).reshape([-1, 1])]
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Coffee Roasting", "original temperature/duration data · library training and handwritten inference")
  np, tf = Reference.numpy, Reference.tensorflow
  x, y = CoffeeRoasting.data
  normalization = tf.keras.layers.Normalization.new(axis: -1)
  normalization.adapt(x)
  xn = normalization.__call__(x)
  train_x, train_y = np.tile(xn, [1000, 1]), np.tile(y, [1000, 1])
  Reference.seed(1234)
  model = tf.keras.Sequential.new([tf.keras.Input(shape: [2]), tf.keras.layers.Dense.new(3, activation: "sigmoid", name: "layer1"), tf.keras.layers.Dense.new(1, activation: "sigmoid", name: "layer2")])
  model.compile(loss: tf.keras.losses.BinaryCrossentropy.new, optimizer: tf.keras.optimizers.Adam.new(learning_rate: 0.01))
  history = model.fit(train_x, train_y, epochs: 10, batch_size: 32, verbose: 0)
  Reference.epoch_trace(history)
  step("trained #{Reference.shape(train_x).inspect} tiled observations; original #{Reference.shape(x).inspect}")
  w1, b1 = [[-8.94, 0.29, 12.89], [-0.17, -7.34, 10.79]], [-9.87, -9.28, 1.01]
  w2, b2 = [[-31.38], [-27.86], [-32.79]], [15.54]
  model.get_layer("layer1").set_weights([np.asarray(w1), np.asarray(b1)])
  model.get_layer("layer2").set_weights([np.asarray(w2), np.asarray(b2)])
  test = normalization.__call__(np.asarray([[200, 13.9], [200, 17]]))
  prediction = model.predict(test, verbose: 0)
  step("course supplied-weight predictions #{prediction}; decisions #{np.greater_equal(prediction, 0.5).astype('int32')}")
  Reference.list(test.numpy).each_with_index do |row, i|
    hidden, _ = DenseLayer.forward(row.to_a, w1.transpose, b1)
    output, _ = DenseLayer.forward(hidden, w2.transpose, b2)
    expected = prediction.__getitem__(i).__getitem__(0).item
    step("handwritten forward pass #{i}: #{output.first}")
    raise "Coffee inference failed" unless (output.first - expected).abs < 1e-5
  end
  ok("the original coffee experiment retains both training and explicit forward propagation")
end
