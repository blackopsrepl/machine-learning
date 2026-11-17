# frozen_string_literal: true

# Course C2 W1 assignment: recognize handwritten zeros and ones from 400 pixels.
# 400→25→15→1 sigmoid network, binary cross-entropy, Adam(0.001), 20 epochs,
# minibatches of 32 shuffled each epoch.
# Run me: ruby neural_networks/binary_classification.rb
require_relative "sequential"
require_relative "../lib/course_data"
module BinaryClassification
  module_function

  def train(epochs: 20, &trace)
    x, y = CourseData.digits(:binary)
    model = Sequential.new(400, [[25, :sigmoid], [15, :sigmoid], [1, :sigmoid]], rng: Random.new(1234))
    model.compile(loss: :binary_crossentropy, optimizer: Adam.new(learning_rate: 0.001))
    history = model.fit(x, y, epochs: epochs, rng: Random.new(1234), &trace)
    [model, history, x, y]
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Binary Digit Recognition", "1000 images × 400 pixels · 400→25→15→1")
  model, history, x, y = BinaryClassification.train { |epoch, loss| step("epoch #{epoch}: loss=#{loss}") }
  probabilities = model.predict(x).map(&:first)
  accuracy = probabilities.zip(y).count { |p, t| (p >= 0.5 ? 1 : 0) == t }.to_f / y.size
  step("parameters=#{model.count_params}; training accuracy=#{accuracy}")
  [0, 500].each { |i| step("image #{i}: label #{y[i]}, P(one)=#{probabilities[i]}") }
  raise "architecture differs from 400→25→15→1" unless model.count_params == 10_431
  raise "training did not reduce the loss" unless history.last < history.first

  ok("zero/one recognition trained by backpropagation and Adam")
end
