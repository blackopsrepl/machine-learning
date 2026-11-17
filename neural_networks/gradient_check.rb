# frozen_string_literal: true

# Gradient checking — central differences approximate each partial derivative.
# Check all weights AND biases, with L2, across the three output/loss contracts.
# This checks the real trainer rather than a separately reimplemented formula.
# Run me: ruby neural_networks/gradient_check.rb
require_relative "backpropagation"
module GradientCheck
  module_function
  def check(network,x,y,loss:,l2: 0.3,epsilon: 1e-5)
    original=network.parameters
    _cost,analytic=network.gradients(x,y,loss: loss,l2: l2)
    numerical=original.each_index.map do |j|
      plus=original.dup; plus[j]+=epsilon; network.parameters=plus
      right=network.gradients(x,y,loss: loss,l2: l2)[0]
      minus=original.dup; minus[j]-=epsilon; network.parameters=minus
      left=network.gradients(x,y,loss: loss,l2: l2)[0]
      (right-left)/(2*epsilon)
    end
    analytic.zip(numerical).map { |a,n| (a-n).abs }.max
  ensure
    network.parameters=original if original
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Gradient Checking", "finite differences versus every backprop parameter")
  [[:binary,[:sigmoid,:sigmoid],1,[0,1]],[:categorical,[:relu,:softmax],3,[0,2]],[:mse,[:relu,:linear],2,[[0.2,-0.3],[0.8,0.4]]]].each do |loss,acts,width,targets|
    model=Backpropagation::Network.new([2,3,width],activations: acts)
    error=GradientCheck.check(model,[[0.2,0.7],[-0.4,0.3]],targets,loss: loss)
    step(format("%-11s %2d parameters; max |analytic−numerical| %.3e",loss,model.parameters.size,error))
    raise "gradient mismatch" unless error<1e-7
  end
  ok("BCE, categorical CE, squared error and L2 agree with numerical derivatives")
end
