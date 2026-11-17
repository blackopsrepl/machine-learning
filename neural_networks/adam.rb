# frozen_string_literal: true

# Adam, as Keras implements it (tf.keras.optimizers.Adam, TF 2.11+).
# m+=(g−m)(1−β1), v+=(g²−v)(1−β2), α_t=lr·√(1−β2^t)/(1−β1^t),
# θ−=α_t·m/(√v+ε) with ε=1e-7. Algebraically this is the textbook
# m̂=m/(1−β1^t), v̂=v/(1−β2^t), θ−=lr·m̂/(√v̂+ε̂) with ε̂=ε/√(1−β2^t):
# the bias corrections are folded into one step size.
# Run me: ruby neural_networks/adam.rb
require_relative "../lib/style"
class Adam
  attr_reader :iterations

  def initialize(learning_rate: 0.001, beta1: 0.9, beta2: 0.999, epsilon: 1e-7)
    @rate, @beta1, @beta2, @epsilon = learning_rate, beta1, beta2, epsilon
    @m, @v, @iterations = nil, nil, 0
  end

  # Update each tensor (a flat Float array) in place from its gradient.
  def apply(tensors, gradients)
    @m ||= tensors.map { |t| Array.new(t.size, 0.0) }
    @v ||= tensors.map { |t| Array.new(t.size, 0.0) }
    raise ArgumentError, "optimizer state belongs to other tensors" unless @m.map(&:size) == tensors.map(&:size)

    @iterations += 1
    alpha = @rate * Math.sqrt(1 - @beta2**@iterations) / (1 - @beta1**@iterations)
    one_minus_b1, one_minus_b2 = 1 - @beta1, 1 - @beta2
    tensors.each_with_index do |tensor, t|
      m, v, g = @m[t], @v[t], gradients[t]
      i = 0
      size = tensor.size
      while i < size
        gi = g[i]
        m[i] += (gi - m[i]) * one_minus_b1
        v[i] += (gi * gi - v[i]) * one_minus_b2
        tensor[i] -= m[i] * alpha / (Math.sqrt(v[i]) + @epsilon)
        i += 1
      end
    end
    yield @iterations, alpha, @m, @v if block_given?
    tensors
  end
end
if __FILE__ == $PROGRAM_NAME
  require_relative "sequential"
  require_relative "../lib/course_data"
  include Style
  banner("Adam", "Keras' update rule on the 400→25→15→1 digit network")
  x, y = CourseData.digits(:binary)
  model = Sequential.new(400, [[25, :sigmoid], [15, :sigmoid], [1, :sigmoid]], rng: Random.new(1234))
  optimizer = Adam.new(learning_rate: 0.001)
  model.compile(loss: :binary_crossentropy, optimizer: optimizer)
  batch = (0...x.size).step(x.size / 32).first(32)
  textbook_m = textbook_v = 0.0
  10.times do |step_index|
    activations, logits = model.forward(batch.map { |i| x[i] })
    loss, delta = model.loss_and_delta(activations.last, logits.last, batch.map { |i| y[i] }, batch.size)
    gradients, = model.backward(activations, logits, delta)
    g = gradients.last.first
    bias_before = model.layers.last.bias.first
    textbook_m = 0.9 * textbook_m + 0.1 * g
    textbook_v = 0.999 * textbook_v + 0.001 * g * g
    t = step_index + 1
    textbook = bias_before - 0.001 * (textbook_m / (1 - 0.9**t)) / (Math.sqrt(textbook_v / (1 - 0.999**t)) + 1e-7 / Math.sqrt(1 - 0.999**t))
    optimizer.apply(model.tensors, gradients) do |iteration, alpha, m, v|
      step("step #{iteration}: batch BCE=#{loss}; output-bias gradient=#{g}; m=#{m.last.first}; v=#{v.last.first}; α_t=#{alpha}")
    end
    raise "Keras and textbook Adam disagree" unless (model.layers.last.bias.first - textbook).abs <= 1e-12 * textbook.abs
  end
  ok("the folded Keras step equals the bias-corrected textbook step")
end
