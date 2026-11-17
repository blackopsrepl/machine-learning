# frozen_string_literal: true

# Softmax — p_j=exp(z_j)/Σexp(z_k). Subtract max(z) before exponentiating.
# Categorical cross-entropy is logsumexp(z)-z_target; dL/dz=p-one_hot.
# Translation invariance is the reason subtracting the maximum changes nothing.
# Run me: ruby neural_networks/softmax.rb
require_relative "../lib/style"
module Softmax
  module_function

  def probabilities(logits)
    shifted = logits.map { |z| Math.exp(z - logits.max) }
    total = shifted.sum
    shifted.map { |e| e / total }
  end

  def loss(logits, target)
    maximum = logits.max
    Math.log(logits.sum { |z| Math.exp(z - maximum) }) + maximum - logits.fetch(target)
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Softmax", "stable probabilities and categorical cross-entropy")
  [[1.0, 2.0, 3.0], [1001.0, 1002.0, 1003.0]].each do |z|
    p = Softmax.probabilities(z)
    step("logits #{vector(z)} → #{vector(p)}; loss(target 2)=#{Softmax.loss(z, 2).round(6)}")
    note("logit gradient #{vector(p.each_with_index.map { |v, j| v - (j == 2 ? 1.0 : 0.0) })}")
    raise "probabilities invalid" unless p.all?(&:finite?) && (p.sum - 1).abs < 1e-12
  end
  a = Softmax.probabilities([1.0, 2.0, 3.0])
  b = Softmax.probabilities([1001.0, 1002.0, 1003.0])
  raise "translation invariance lost" unless a.zip(b).all? { |x, y| (x - y).abs < 1e-12 }

  ok("large logits preserve both probabilities and finite loss")
end
