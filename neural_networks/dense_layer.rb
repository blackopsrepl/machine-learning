# frozen_string_literal: true

# Dense layer — z_j=Σ_i W_ji*a_i+b_j; a_out=g(z).
# Weight rows belong to output neurons. This file is inference, not training.
# Run me: ruby neural_networks/dense_layer.rb
require_relative "../lib/style"
require_relative "../lib/numeric"
require_relative "../supervised/logistic_regression"
require_relative "softmax"
module DenseLayer
  module_function

  def activate(z, kind)
    case kind
    when :sigmoid then z.map { |v| LogisticRegression.sigmoid(v) }
    when :relu then z.map { |v| [v, 0.0].max }
    when :linear then z.dup
    when :softmax then Softmax.probabilities(z)
    else raise ArgumentError, "unknown activation #{kind}"
    end
  end

  def forward(input, weights, bias, activation: :sigmoid)
    z = weights.zip(bias).map { |row, b| NumericTools.dot(row, input) + b }
    [activate(z, activation), z]
  end
end
if __FILE__ == $PROGRAM_NAME
  require_relative "binary_classification"
  include Style
  banner("Dense-Layer Inference", "original digit image and 400→25→15→1 weights · handwritten Ruby")
  tf = Reference.tensorflow
  model = if File.exist?("output/binary_digits.keras")
            tf.keras.models.load_model("output/binary_digits.keras")
          else
            BinaryClassification.train.first
          end
  x, y = Reference.digits(:binary)
  row = Reference.list(x.__getitem__(0))
  activation = row
  model.layers.to_a.each_with_index do |layer, i|
    weights, bias = layer.get_weights.to_a
    activation, z = DenseLayer.forward(activation, Reference.list(weights).map(&:to_a).transpose, Reference.list(bias))
    step("layer #{i + 1}: #{z.size} neurons; logits #{vector(z.first(5))}; activations #{vector(activation.first(5))}")
  end
  expected = model.predict(Reference.rows(x, 0, 1), verbose: 0).__getitem__(0).__getitem__(0).item
  step("Ruby probability=#{activation.first}; Keras probability=#{expected}; label=#{y.__getitem__(0)}")
  raise "Original layer weights were not applied correctly" unless (activation.first - expected).abs < 1e-6

  ok("the NumPy-style forward pass is preserved in Ruby on the real digit input")
end
