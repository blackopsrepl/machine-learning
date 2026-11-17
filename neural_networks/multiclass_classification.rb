# frozen_string_literal: true

# Multiclass MLP — ReLU hidden units, linear logits, stable softmax cross-entropy.
# The course uses digit images; three small clusters make all probabilities visible.
# Run me: ruby neural_networks/multiclass_classification.rb
require_relative "backpropagation"
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Multiclass Neural Network", "ReLU → logits → softmax · seed 42")
  x,y=Problem.classes,Problem.class_targets
  model=Backpropagation::Network.new([2,6,3],activations: [:relu,:softmax])
  model.fit(x,y,loss: :categorical,iterations: 400,optimizer: Adam.new) { |i,cost,_g| step("epoch #{i} cross-entropy=#{cost.round(6)}") if i<=2 || (i%100).zero? }
  predicted=x.map do |row|
    p=model.predict(row); label=p.each_index.max_by { |j| p[j] }
    step("#{vector(row)} → #{vector(p)} → class #{label}"); label
  end
  raise "multiclass failure" unless predicted==y
  ok("the same manually differentiated layers separate three classes")
end
