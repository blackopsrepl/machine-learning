# frozen_string_literal: true

# Sequential dense networks, trained the way the course's Keras models are.
# Dense layers hold a kernel (Glorot-uniform, limit √(6/(fan_in+fan_out)))
# and zero biases. Forward: z=W·a+b, a=g(z). Backward: δ_out=∂L/∂z_out,
# δ=(Wᵀδ_next)⊙g'(z), ∂L/∂W=δ·aᵀ, ∂L/∂b=δ, averaged over the minibatch.
# Losses match Keras: a sigmoid/softmax output with binary/sparse categorical
# cross-entropy is evaluated from its logits (Keras does the same through the
# activation's cached logits), so ∂L/∂z=(p−y)/batch. MSE averages the output
# axis; L2(λ) adds λΣw² to the batch loss. fit() shuffles each epoch and
# steps Keras' Adam once per minibatch of 32, reporting the sample-weighted
# mean batch loss like Keras' history.
# Run me: ruby neural_networks/sequential.rb
require_relative "../lib/style"
require_relative "adam"
class Sequential
  Layer = Struct.new(:inputs, :units, :activation, :l2, :kernel, :bias)

  LOSSES = %i[binary_crossentropy sparse_categorical_crossentropy mse].freeze
  attr_reader :layers

  # layers: [[units, activation, {l2: λ}], ...]; activation in :linear, :relu, :sigmoid, :softmax.
  def initialize(input_size, layers, rng: Random.new(1234))
    @layers = []
    inputs = input_size
    layers.each do |units, activation, options = {}|
      limit = Math.sqrt(6.0 / (inputs + units))
      kernel = Array.new(units * inputs) { (rng.rand * 2 - 1) * limit }
      @layers << Layer.new(inputs, units, activation, options.fetch(:l2, 0.0), kernel, Array.new(units, 0.0))
      inputs = units
    end
  end

  def count_params = @layers.sum { |l| l.kernel.size + l.bias.size }
  def tensors = @layers.flat_map { |l| [l.kernel, l.bias] }

  def compile(loss:, optimizer:)
    raise ArgumentError, "unknown loss #{loss}" unless LOSSES.include?(loss)

    @loss, @optimizer = loss, optimizer
    self
  end

  # Forward pass over a batch; returns per-layer activations (index 0 = input)
  # and pre-activations, each a flat row-major batch×units array.
  def forward(rows)
    batch = rows.size
    activations = [rows.flatten]
    logits = []
    @layers.each do |layer|
      input = activations.last
      n_in, n_out, kernel, bias = layer.inputs, layer.units, layer.kernel, layer.bias
      z = Array.new(batch * n_out, 0.0)
      b = 0
      while b < batch
        in_offset = b * n_in
        out_offset = b * n_out
        j = 0
        while j < n_out
          z[out_offset + j] = bias[j]
          j += 1
        end
        k = 0
        while k < n_in
          value = input[in_offset + k]
          if value != 0.0
            j = 0
            index = k
            while j < n_out
              z[out_offset + j] += kernel[index] * value
              index += n_in
              j += 1
            end
          end
          k += 1
        end
        b += 1
      end
      logits << z
      activations << activate(z, layer.activation, n_out)
    end
    [activations, logits]
  end

  def activate(z, kind, width)
    case kind
    when :linear then z.dup
    when :relu then z.map { |v| v > 0.0 ? v : 0.0 }
    when :sigmoid then z.map { |v| v >= 0 ? 1.0 / (1.0 + Math.exp(-v)) : Math.exp(v) / (1.0 + Math.exp(v)) }
    when :softmax
      z.each_slice(width).flat_map do |row|
        top = row.max
        e = row.map { |v| Math.exp(v - top) }
        total = e.sum
        e.map { |v| v / total }
      end
    else raise ArgumentError, "unknown activation #{kind}"
    end
  end

  def predict(rows, batch_size: 256)
    width = @layers.last.units
    rows.each_slice(batch_size).flat_map { |chunk| forward(chunk)[0].last.each_slice(width).to_a }
  end

  def logits(rows, batch_size: 256)
    width = @layers.last.units
    rows.each_slice(batch_size).flat_map { |chunk| forward(chunk)[1].last.each_slice(width).to_a }
  end

  # Data loss (batch mean) and ∂loss/∂z_out for the compiled Keras loss.
  def loss_and_delta(output, z, targets, batch)
    width = @layers.last.units
    delta = Array.new(batch * width, 0.0)
    total = 0.0
    case @loss
    when :binary_crossentropy
      raise ArgumentError, "binary cross-entropy needs one sigmoid or logit output" unless width == 1

      batch.times do |b|
        logit, target = z[b], targets[b].to_f
        total += [logit, 0.0].max - logit * target + Math.log(1 + Math.exp(-logit.abs))
        probability = @layers.last.activation == :sigmoid ? output[b] : 1.0 / (1.0 + Math.exp(-logit))
        delta[b] = (probability - target) / batch
      end
    when :sparse_categorical_crossentropy
      batch.times do |b|
        row = z[b * width, width]
        top = row.max
        exps = row.map { |v| Math.exp(v - top) }
        sum = exps.sum
        target = targets[b].to_i
        total += Math.log(sum) + top - row[target]
        width.times { |j| delta[b * width + j] = (exps[j] / sum - (j == target ? 1.0 : 0.0)) / batch }
      end
    when :mse
      batch.times do |b|
        target = Array(targets[b])
        width.times do |j|
          error = output[b * width + j] - target[j]
          total += error * error / width
          derivative = case @layers.last.activation
                       when :linear then 1.0
                       when :relu then z[b * width + j] > 0 ? 1.0 : 0.0
                       when :sigmoid then output[b * width + j] * (1 - output[b * width + j])
                       else raise ArgumentError, "MSE with #{@layers.last.activation} output is not supported"
                       end
          delta[b * width + j] = 2.0 * error / width / batch * derivative
        end
      end
    end
    [total / batch, delta]
  end

  def regularization = @layers.sum { |l| l.l2.zero? ? 0.0 : l.l2 * l.kernel.sum { |w| w * w } }

  # Backpropagate ∂L/∂z_out; returns gradients for every tensor and, when
  # requested, ∂L/∂input for models composed with other differentiable parts.
  def backward(activations, logits, delta, input_gradient: false)
    batch = activations.first.size / @layers.first.inputs
    gradients = Array.new(@layers.size * 2)
    input_delta = nil
    (@layers.size - 1).downto(0) do |l|
      layer = @layers[l]
      n_in, n_out, kernel = layer.inputs, layer.units, layer.kernel
      input = activations[l]
      grad_kernel = Array.new(n_in * n_out, 0.0)
      grad_bias = Array.new(n_out, 0.0)
      need_previous = l.positive? || input_gradient
      previous = need_previous ? Array.new(batch * n_in, 0.0) : nil
      b = 0
      while b < batch
        in_offset = b * n_in
        out_offset = b * n_out
        j = 0
        while j < n_out
          d = delta[out_offset + j]
          if d != 0.0
            grad_bias[j] += d
            row = j * n_in
            k = 0
            while k < n_in
              value = input[in_offset + k]
              grad_kernel[row + k] += d * value if value != 0.0
              previous[in_offset + k] += kernel[row + k] * d if need_previous
              k += 1
            end
          end
          j += 1
        end
        b += 1
      end
      if layer.l2.positive?
        scale = 2.0 * layer.l2
        grad_kernel.each_index { |i| grad_kernel[i] += scale * kernel[i] }
      end
      gradients[2 * l] = grad_kernel
      gradients[2 * l + 1] = grad_bias
      next unless need_previous

      if l.zero?
        input_delta = previous
        next
      end
      below = @layers[l - 1]
      z = logits[l - 1]
      a = activations[l]
      delta = case below.activation
              when :linear then previous
              when :relu then previous.each_index.map { |i| z[i] > 0.0 ? previous[i] : 0.0 }
              when :sigmoid then previous.each_index.map { |i| previous[i] * a[i] * (1 - a[i]) }
              else raise ArgumentError, "#{below.activation} is output-only"
              end
    end
    [gradients, input_delta]
  end

  # One optimizer step on a minibatch; returns the Keras batch loss.
  def train_on_batch(rows, targets)
    activations, logits = forward(rows)
    data_loss, delta = loss_and_delta(activations.last, logits.last, targets, rows.size)
    gradients, = backward(activations, logits, delta)
    penalty = regularization
    @optimizer.apply(tensors, gradients)
    data_loss + penalty
  end

  def fit(x, y, epochs:, batch_size: 32, shuffle: true, rng: Random.new(1234))
    raise "compile before fit" unless @loss

    history = []
    order = (0...x.size).to_a
    epochs.times do |epoch|
      order = order.shuffle(random: rng) if shuffle
      weighted, seen = 0.0, 0
      order.each_slice(batch_size) do |indices|
        loss = train_on_batch(indices.map { |i| x[i] }, indices.map { |i| y[i] })
        weighted += loss * indices.size
        seen += indices.size
      end
      history << weighted / seen
      yield epoch + 1, history.last if block_given?
    end
    history
  end

  # Mean data loss plus regularization over a whole dataset (Keras evaluate).
  def evaluate(x, y, batch_size: 256)
    total = 0.0
    x.each_slice(batch_size).with_index do |chunk, i|
      activations, logits = forward(chunk)
      loss, = loss_and_delta(activations.last, logits.last, y[i * batch_size, chunk.size], chunk.size)
      total += loss * chunk.size
    end
    total / x.size + regularization
  end

  def weights = @layers.map { |l| [l.kernel.dup, l.bias.dup] }

  def weights=(values)
    @layers.zip(values).each do |layer, (kernel, bias)|
      raise ArgumentError, "weight shapes differ" unless kernel.size == layer.kernel.size && bias.size == layer.bias.size

      layer.kernel.replace(kernel.map(&:to_f))
      layer.bias.replace(bias.map(&:to_f))
    end
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Sequential Training", "Keras dense layers, losses, minibatches and Adam in Ruby")
  model = Sequential.new(2, [[3, :sigmoid], [1, :sigmoid]], rng: Random.new(7))
  model.compile(loss: :binary_crossentropy, optimizer: Adam.new(learning_rate: 0.01))
  x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 8
  y = [0, 1, 1, 0] * 8
  step("2→3→1 sigmoid network: #{model.count_params} parameters; Glorot limits #{vector(model.layers.map { |l| Math.sqrt(6.0 / (l.inputs + l.units)) })}")
  history = model.fit(x, y, epochs: 3, batch_size: 8, rng: Random.new(3)) { |epoch, loss| step("epoch #{epoch}: loss=#{loss}") }
  raise "loss is not finite" unless history.all?(&:finite?)

  ok("forward, backward and optimizer steps compose into Keras-style fit")
end
