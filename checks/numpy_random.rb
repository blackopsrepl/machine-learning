# frozen_string_literal: true

require_relative "../lib/numpy_random"

# Values printed by NumPy 1.26 for the same seeds; the Ruby streams must match exactly.
expected = {
  "RandomState(1).random_sample" => [NumpyRandom::RandomState.new(1).then { |r| Array.new(3) { r.random_sample } }, [0.417022004702574, 0.7203244934421581, 0.00011437481734488664]],
  "RandomState(30).normal" => [NumpyRandom::RandomState.new(30).then { |r| Array.new(3) { r.standard_normal } }, [-1.2640526647764023, 1.5279053516384395, -0.9707109380088592]],
  "RandomState(42).permutation" => [NumpyRandom::RandomState.new(42).permutation(10), [8, 1, 5, 0, 7, 2, 9, 4, 3, 6]],
  "default_rng(2).random" => [NumpyRandom::Generator.new(2).then { |g| Array.new(3) { g.random } }, [0.2616121342493164, 0.2984911434141233, 0.8142257405942803]]
}
expected.each do |name, (actual, reference)|
  raise "#{name} differs: #{actual.inspect} vs #{reference.inspect}" unless actual == reference

  puts "#{name}: identical to NumPy"
end
