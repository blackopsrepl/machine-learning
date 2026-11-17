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
  def activate(z,kind)
    case kind
    when :sigmoid then z.map { |v| LogisticRegression.sigmoid(v) }
    when :relu then z.map { |v| [v,0.0].max }
    when :linear then z.dup
    when :softmax then Softmax.probabilities(z)
    else raise ArgumentError,"unknown activation #{kind}"
    end
  end
  def forward(input,weights,bias,activation: :sigmoid)
    z=weights.zip(bias).map { |row,b| NumericTools.dot(row,input)+b }
    [activate(z,activation),z]
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Dense Layer", "each neuron is an inspectable dot product")
  input,weights,bias=[1.0,2.0],[[1.0,-1.0],[0.5,1.0]],[0.0,-1.0]
  output,z=DenseLayer.forward(input,weights,bias)
  z.each_index { |j| step("neuron #{j}: #{vector(weights[j])} · #{vector(input)} + #{bias[j]} = #{z[j]}; sigmoid=#{output[j].round(6)}") }
  raise "wrong affine transform" unless z==[-1.0,1.5]
  ok("composition needs only weights, biases and an activation")
end
