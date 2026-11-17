# frozen_string_literal: true

# Neural model selection — compare capacity and L2 on the same noisy labels.
# Choose by validation classification error, not by lowest training loss.
# Validation/test points are separate observations, not fit-time preprocessing.
# Run me: ruby evaluation/neural_model_selection.rb
require_relative "../neural_networks/backpropagation"
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Neural Model Selection", "simple, complex and regularized · same noisy training rows")
  x=Problem.classes; noisy=Problem.class_targets; noisy[1]=1; noisy[4]=2
  validation=x.map { |a,b| [a+0.15,b+0.1] }; labels=Problem.class_targets
  choices=[[[2,2,3],0.0],[[2,12,12,3],0.0],[[2,12,12,3],0.5]].map do |sizes,lambda|
    network=Backpropagation::Network.new(sizes,activations: Array.new(sizes.size-2,:relu)+[:softmax])
    network.fit(x,noisy,loss: :categorical,l2: lambda,iterations: 500,optimizer: Adam.new)
    classify=->(rows) { rows.map { |r| p=network.predict(r); p.each_index.max_by { |j| p[j] } } }
    train_error=1-NumericTools.accuracy(noisy,classify.call(x))
    validation_error=1-NumericTools.accuracy(labels,classify.call(validation))
    step("sizes=#{sizes.inspect} λ=#{lambda} train error=#{train_error.round(3)} validation error=#{validation_error.round(3)}")
    [validation_error,network,lambda]
  end
  error,best,lambda=choices.min_by(&:first)
  test=x.map { |a,b| [a-0.15,b-0.1] }
  predicted=test.map { |r| p=best.predict(r); p.each_index.max_by { |j| p[j] } }
  test_error=1-NumericTools.accuracy(labels,predicted)
  step("selected λ=#{lambda}; validation error=#{error.round(3)}; untouched test error=#{test_error.round(3)}")
  raise "selection lost useful classification" unless test_error<0.5
  ok("capacity and regularization are measured on held-out labels, not guessed from fit")
end
