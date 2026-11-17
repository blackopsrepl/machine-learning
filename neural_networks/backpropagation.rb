# frozen_string_literal: true

# Backpropagation — cache each activation/preactivation, propagate deltas backward.
# For sigmoid+BCE and softmax+CE, output delta is prediction-target.
# Hidden delta=(W_nextᵀ*delta_next) ⊙ g'(z); dW=delta*a_previousᵀ.
# Batch-average gradients; L2 touches weights only. Parameter rows are neurons.
# Run me: ruby neural_networks/backpropagation.rb
require_relative "dense_layer"
require_relative "adam"
require_relative "../lib/problem"
module Backpropagation
  class Network
    attr_reader :weights, :biases, :activations

    def initialize(sizes, activations:, rng: Random.new(42))
      raise ArgumentError, "one activation per layer" unless activations.size == sizes.size - 1
      raise ArgumentError, "softmax is output-only" if activations[0...-1].include?(:softmax)

      @activations = activations
      @weights = sizes.each_cons(2).map do |inputs, outputs|
        limit = Math.sqrt(6.0 / (inputs + outputs))
        Array.new(outputs) { Array.new(inputs) { rng.rand * 2 * limit - limit } }
      end
      @biases = sizes.drop(1).map { |n| Array.new(n, 0.0) }
    end

    def forward(row)
      values, zs = [row], []
      @weights.each_index do |layer|
        output, z = DenseLayer.forward(values.last, @weights[layer], @biases[layer], activation: @activations[layer])
        values << output
        zs << z
      end
      [values, zs]
    end

    def predict(row) = forward(row)[0].last
    def parameters = @weights.each_index.flat_map { |l| @weights[l].flatten + @biases[l] }

    def parameters=(values)
      raise ArgumentError, "parameter count differs" unless values.size == parameters.size

      offset = 0
      @weights.each_index do |l|
        @weights[l].each do |row|
          row.each_index do |j|
            row[j] = values[offset]
            offset += 1
          end
        end
        @biases[l].each_index do |j|
          @biases[l][j] = values[offset]
          offset += 1
        end
      end
    end

    # Vector-Jacobian product from output LOGITS, reusing the cached forward pass.
    # Callers supply dL/dz_out; activation/loss fusion belongs to the objective.
    def backward(values, zs, delta)
      gw = @weights.map { |layer| layer.map { |row| Array.new(row.size, 0.0) } }
      gb = @biases.map { |layer| Array.new(layer.size, 0.0) }
      (@weights.size - 1).downto(0) do |l|
        delta.each_index do |j|
          gb[l][j] = delta[j]
          gw[l][j].each_index { |k| gw[l][j][k] = delta[j] * values[l][k] }
        end
        next if l.zero?

        delta = values[l].each_index.map do |k|
          propagated = @weights[l].each_index.sum { |j| @weights[l][j][k] * delta[j] }
          derivative = case @activations[l - 1]
                       when :sigmoid then values[l][k] * (1 - values[l][k])
                       when :relu then zs[l - 1][k] > 0 ? 1.0 : 0.0
                       when :linear then 1.0
                       end
          propagated * derivative
        end
      end
      @weights.each_index.flat_map { |l| gw[l].flatten + gb[l] }
    end

    def gradients(x, y, loss: :binary, l2: 0.0)
      expected = { binary: :sigmoid, categorical: :softmax, mse: :linear }.fetch(loss)
      raise ArgumentError, "loss/output activation mismatch" unless @activations.last == expected

      gradient = Array.new(parameters.size, 0.0)
      cost = 0.0
      x.each_with_index do |row, i|
        values, zs = forward(row)
        output = values.last
        target = loss == :categorical ? Array.new(output.size) { |j| j == y[i] ? 1.0 : 0.0 } : Array(y[i])
        raise ArgumentError, "target width differs" unless target.size == output.size

        cost += case loss
                when :binary then zs.last.zip(target).sum { |z, t| LogisticRegression.softplus(z) - t * z }
                when :categorical then Softmax.loss(zs.last, y[i])
                when :mse then output.zip(target).sum { |pred, t| (pred - t)**2 / 2.0 }
                end
        delta = output.zip(target).map { |pred, t| pred - t }
        sample_gradient = backward(values, zs, delta)
        gradient = gradient.zip(sample_gradient).map { |total, g| total + g }
      end
      penalty_gradient = @weights.each_index.flat_map { |l| @weights[l].flatten.map { |w| l2 * w } + Array.new(@biases[l].size, 0.0) }
      gradient = gradient.zip(penalty_gradient).map { |g, r| (g + r) / x.size }
      cost = (cost + l2 * @weights.flatten.sum { |w| w * w } / 2.0) / x.size
      [cost, gradient]
    end

    def fit(x, y, loss: :binary, iterations: 1000, rate: 0.1, l2: 0.0, optimizer: nil)
      iterations.times do |i|
        cost, gradient = gradients(x, y, loss: loss, l2: l2)
        self.parameters = optimizer ? optimizer.step(parameters, gradient) : parameters.zip(gradient).map { |p, g| p - rate * g }
        yield i + 1, cost, gradient if block_given?
      end
      self
    end
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Backpropagation", "explicit chain rule · sigmoid hidden layer")
  network = Backpropagation::Network.new([2, 3, 1], activations: [:sigmoid, :sigmoid])
  values, zs = network.forward([1.0, 0.0])
  values.each_with_index { |a, l| step("layer #{l} activations #{vector(a)}") }
  note("preactivations #{zs.map { |z| vector(z) }.join(' → ')}")
  cost, gradient = network.gradients(Problem.xor, Problem.xor_targets)
  step("initial batch BCE #{cost.round(6)}; first weight gradients #{vector(gradient.first(6))}")
  network.fit(Problem.xor, Problem.xor_targets, iterations: 2000, optimizer: Adam.new) { |i, j, _g| step("it #{i}: J=#{j.round(6)}") if i <= 2 || (i % 500).zero? }
  predicted = Problem.xor.map { |x| network.predict(x)[0] >= 0.5 ? 1 : 0 }
  raise "XOR not learned" unless predicted == Problem.xor_targets

  ok("handwritten hidden deltas learn XOR; no tape or automatic derivatives")
end
