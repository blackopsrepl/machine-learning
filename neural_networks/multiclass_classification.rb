# frozen_string_literal: true

# Course C2 W2 assignment: ten handwritten digits, 400→25→15→10 with ReLU
# hidden units and linear logits; sparse categorical cross-entropy from
# logits, Adam(0.001), 40 epochs, minibatches of 32. Softmax only for reading
# probabilities, never inside the loss.
# Run me: ruby neural_networks/multiclass_classification.rb
require_relative "sequential"
require_relative "softmax"
require_relative "../lib/course_data"
module MulticlassClassification
  module_function

  def train(epochs: 40, &trace)
    x, y = CourseData.digits(:multiclass)
    model = Sequential.new(400, [[25, :relu], [15, :relu], [10, :linear]], rng: Random.new(1234))
    model.compile(loss: :sparse_categorical_crossentropy, optimizer: Adam.new(learning_rate: 0.001))
    history = model.fit(x, y, epochs: epochs, rng: Random.new(1234), &trace)
    [model, history, x, y]
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Multiclass Digit Recognition", "5000 images · ten classes · 400→25→15→10")
  model, history, x, y = MulticlassClassification.train { |epoch, loss| step("epoch #{epoch}: loss=#{loss}") }
  logits = model.logits(x)
  predicted = logits.map { |row| row.index(row.max) }
  accuracy = predicted.zip(y).count { |p, t| p == t }.to_f / y.size
  step("parameters=#{model.count_params}; training accuracy=#{accuracy}; errors=#{predicted.zip(y).count { |p, t| p != t }}")
  [1015, 4021].each { |i| step("image #{i}: label #{y[i]}; probabilities #{vector(Softmax.probabilities(logits[i]))}") }
  raise "architecture differs from 400→25→15→10" unless model.count_params == 10_575
  raise "training did not reduce the loss" unless history.last < history.first

  ok("all ten digit classes trained from logits")
end
