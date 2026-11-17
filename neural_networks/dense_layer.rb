# frozen_string_literal: true

# The course's NumPy forward pass: my_dense computes each unit j as
# g(W[:,j]·a_in + b[j]); my_sequential chains layers. W is inputs×units.
# Run once with the coffee lab's supplied weights and once with weights the
# digit assignment trains, comparing against the training engine's predict.
# Run me: ruby neural_networks/dense_layer.rb
require_relative "../lib/style"
require_relative "../lib/numeric"
require_relative "../supervised/logistic_regression"
module DenseLayer
  module_function

  def sigmoid(z) = LogisticRegression.sigmoid(z)

  def my_dense(a_in, w, b, activation = method(:sigmoid))
    units = w.first.size
    Array.new(units) { |j| activation.call(NumericTools.dot(w.map { |row| row[j] }, a_in) + b[j]) }
  end

  def my_sequential(x, layers) = layers.reduce(x) { |a, (w, b)| my_dense(a, w, b) }
end
if __FILE__ == $PROGRAM_NAME
  require_relative "coffee_roasting"
  require_relative "binary_classification"
  include Style
  banner("Dense Layers in NumPy Style", "my_dense / my_sequential · coffee and digit weights")
  x_tst = [0.1, 0.2]
  w_tst = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
  a_tst = DenseLayer.my_dense(x_tst, w_tst, [0.1, 0.2, 0.3])
  step("assignment quick check: my_dense → #{vector(a_tst)}")
  raise "my_dense quick check differs from the assignment" unless a_tst.zip([0.54735762, 0.57932425, 0.61063923]).all? { |a, e| (a - e).abs < 1e-8 }

  x, = CoffeeRoasting.data
  normalize = CoffeeRoasting.normalization(x)
  coffee = [[[-8.93, 0.29, 12.9], [-0.1, -7.32, 10.81]], [-9.82, -9.28, 0.96]], [[[-31.18], [-27.59], [-32.56]], [15.41]]
  normalize.call([[200, 13.9], [200, 17]]).each_with_index do |row, i|
    p = DenseLayer.my_sequential(row, coffee).first
    step("coffee lab roast #{i}: p=#{p}; decision=#{p >= 0.5 ? 1 : 0}")
  end
  model, = BinaryClassification.train
  digits, labels = CourseData.digits(:binary)
  layers = model.layers.map { |l| [l.kernel.each_slice(l.inputs).to_a.transpose, l.bias] }
  [0, 500].each do |i|
    mine = DenseLayer.my_sequential(digits[i], layers).first
    engine = model.predict([digits[i]]).first.first
    step("digit #{i} (label #{labels[i]}): my_sequential=#{mine}; trained model=#{engine}")
    raise "NumPy-style forward pass differs from the trained model" unless (mine - engine).abs < 1e-12
  end
  ok("unit-by-unit forward passes reproduce the network outputs")
end
