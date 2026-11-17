# frozen_string_literal: true

# ReLU — max(0,z). Hidden nonlinearities prevent a stack of affine maps
# collapsing into one affine map. Two hinges already express |x| exactly.
# Run me: ruby neural_networks/relu.rb
require_relative "dense_layer"
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("ReLU", "two hinges represent a function no straight line can")
  [-2.0,-1.0,0.0,1.0,2.0].each do |x|
    hidden,z=DenseLayer.forward([x],[[1.0],[-1.0]],[0.0,0.0],activation: :relu)
    output,_=DenseLayer.forward(hidden,[[1.0,1.0]],[0.0],activation: :linear)
    step("x=#{x}: preactivations #{vector(z)} → hinges #{vector(hidden)} → #{output[0]}")
    raise "hinges fail" unless output[0]==x.abs
  end
  ok("the hidden nonlinearity constructs absolute value")
end
