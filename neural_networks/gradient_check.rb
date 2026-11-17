# frozen_string_literal: true

# Central finite differences check the actual handwritten digit-network gradient.
# The default demonstration checks selected parameters across every layer on
# original images; callers may request all parameters or another explicit subset.
require_relative "backpropagation"
module GradientCheck
  module_function

  def check(network, x, y, loss:, l2: 0.3, epsilon: 1e-5, indices: nil)
    original = network.parameters
    _cost, analytic = network.gradients(x, y, loss: loss, l2: l2)
    indices ||= original.each_index.to_a
    numerical = indices.map do |j|
      plus = original.dup
      plus[j] += epsilon
      network.parameters = plus
      right = network.gradients(x, y, loss: loss, l2: l2)[0]
      minus = original.dup
      minus[j] -= epsilon
      network.parameters = minus
      left = network.gradients(x, y, loss: loss, l2: l2)[0]
      (right - left) / (2 * epsilon)
    end
    indices.zip(numerical).map { |j, n| (analytic[j] - n).abs }.max
  ensure
    network.parameters = original if original
  end
end
if __FILE__ == $PROGRAM_NAME
  require_relative "../lib/course_data"
  include Style
  banner("Gradient Checking", "actual digit inputs and course-sized handwritten networks")
  [[:binary, :binary, 1, [:sigmoid, :sigmoid, :sigmoid]], [:categorical, :multiclass, 10, [:relu, :relu, :softmax]]].each do |loss, dataset, width, activations|
    x, y = CourseData.digits(dataset)
    network = Backpropagation::Network.new([400, 25, 15, width], activations: activations, rng: Random.new(1234))
    indices = [0, 399, 10024, network.parameters.size - 1] + (0...network.parameters.size).to_a.sample(12, random: Random.new(42))
    error = GradientCheck.check(network, x.first(2), y.first(2), loss: loss, indices: indices)
    step("#{loss}: #{indices.size} selected derivatives across #{network.parameters.size} parameters; max central-difference error=#{error}")
    raise "Digit network derivative failed" unless error < 1e-7
  end
  ok("finite differences exercise the actual image model dimensions")
end
