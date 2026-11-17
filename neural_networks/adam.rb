# frozen_string_literal: true

# Adam — exponential averages of gradients and squared gradients, bias corrected.
# m_hat=m/(1-β1^t), v_hat=v/(1-β2^t); θ -= α*m_hat/(sqrt(v_hat)+ε).
# The optimizer is explicit, not a library hiding behind a custom training loop.
# Run me: ruby neural_networks/adam.rb
require_relative "../lib/style"
class Adam
  def initialize(rate: 0.001, beta1: 0.9, beta2: 0.999, epsilon: 1e-8)
    @rate, @beta1, @beta2, @epsilon = rate, beta1, beta2, epsilon
    @m, @v, @t = nil, nil, 0
  end

  def step(parameters, gradient)
    raise ArgumentError, "gradient shape differs" unless parameters.size == gradient.size

    @m ||= Array.new(parameters.size, 0.0)
    @v ||= Array.new(parameters.size, 0.0)
    raise ArgumentError, "reuse optimizer only for same parameter shape" unless @m.size == parameters.size

    @t += 1
    parameters.each_index.map do |i|
      @m[i] = @beta1 * @m[i] + (1 - @beta1) * gradient[i]
      @v[i] = @beta2 * @v[i] + (1 - @beta2) * gradient[i]**2
      mh = @m[i] / (1 - @beta1**@t)
      vh = @v[i] / (1 - @beta2**@t)
      updated = parameters[i] - @rate * mh / (Math.sqrt(vh) + @epsilon)
      yield @t, i, gradient[i], mh, vh, updated if block_given?
      updated
    end
  end
end
if __FILE__ == $PROGRAM_NAME
  require_relative "backpropagation"
  require_relative "../lib/course_data"
  include Style
  banner("Adam", "handwritten updates on the course digit-network gradient")
  x, y = CourseData.digits(:binary)
  network = Backpropagation::Network.new([400, 25, 15, 1], activations: [:sigmoid, :sigmoid, :sigmoid], rng: Random.new(1234))
  optimizer = Adam.new(rate: 0.001)
  10.times do
    cost, gradient = network.gradients(x.first(32), y.first(32))
    size = network.parameters.size
    network.parameters = optimizer.step(network.parameters, gradient) do |t, j, g, m, v, value|
      step("step #{t}: batch BCE=#{cost}; output-bias gradient=#{g}; m_hat=#{m}; v_hat=#{v}; updated bias=#{value}") if j == size - 1
    end
  end
  raise "Adam update produced nonfinite weights" unless network.parameters.all?(&:finite?)

  ok("moments and bias correction update a full course-sized parameter vector")
end
