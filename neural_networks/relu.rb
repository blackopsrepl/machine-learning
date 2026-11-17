# frozen_string_literal: true

# Original ReLU slider experiment: one fixed hinge and two adjustable hinges.
# CLI weight/bias options replace the sliders without changing their model.
# The original piecewise-linear target is evaluated at the same 300 inputs.
require_relative "dense_layer"
if __FILE__ == $PROGRAM_NAME
  require "optparse"
  include Style
  options = { w1: 0.0, b1: 0.0, w2: 0.0, b2: 0.0 }
  OptionParser.new { |p| options.keys.each { |name| p.on("--#{name} VALUE", Float) { |v| options[name] = v } } }.parse!
  banner("ReLU Activation", "original three-unit slider model · parameters supplied as CLI options")
  error = 0.0
  300.times do |i|
    x = 3.0 * i / 299
    target = i < 100 ? -2 * x + 2 : i < 200 ? x - 1 : 3 * x - 5
    hidden = [[-2.0, 2.0], [options[:w1], options[:b1]], [options[:w2], options[:b2]]].map { |w, b| [0.0, w * x + b].max }
    predicted = hidden.sum
    error += (predicted - target)**2
    step("x=#{x}: units=#{vector(hidden)} output=#{predicted} target=#{target}") if (i % 50).zero? || i == 299
  end
  step("mean squared error=#{error / 300}; w0=-2 b0=2 fixed; #{options.inspect}")
  note("the original target is matched by --w1 1 --b1 -1 --w2 2 --b2 -4")
  ok("the reference hinge model and adjustable weights are preserved without notebook widgets")
end
