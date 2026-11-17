# frozen_string_literal: true

# Binary MLP classification — same sigmoid/BCE model family as course C2 W1.
# XOR is a tiny substitute for images: a line cannot separate it, hidden units can.
# Full-batch Adam training is explicit; prediction remains a separate forward pass.
# Run me: ruby neural_networks/binary_classification.rb
require_relative "backpropagation"
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Binary Neural Network", "XOR defeats the linear baseline · seed 42")
  x,y=Problem.xor,Problem.xor_targets
  w,b=LogisticRegression.fit(x,y)
  linear=x.map { |r| LogisticRegression.probability(r,w,b)>=0.5 ? 1 : 0 }
  step("linear model accuracy #{NumericTools.accuracy(y,linear)}")
  model=Backpropagation::Network.new([2,4,1],activations: [:sigmoid,:sigmoid])
  model.fit(x,y,iterations: 2000,optimizer: Adam.new) { |i,cost,_g| step("epoch #{i} BCE=#{cost.round(6)}") if i<=2 || (i%500).zero? }
  predicted=x.map do |r|
    p=model.predict(r)[0]; step("#{vector(r)} probability=#{p.round(5)} class=#{p>=0.5 ? 1 : 0}"); p>=0.5 ? 1 : 0
  end
  raise "nonlinear classification story failed" unless predicted==y && NumericTools.accuracy(y,linear)<1
  ok("hidden sigmoid units solve the nonlinear problem the line cannot")
end
