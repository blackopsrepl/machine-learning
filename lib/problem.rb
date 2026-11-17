# frozen_string_literal: true

# Tiny shared fixtures: inspect every residual, split and recommendation by eye.
# Regression has an exact answer (y = 1 + 2x). XOR defeats a linear classifier.
# All methods return fresh arrays so running one demo cannot mutate another.
module Problem
  module_function

  def line = [[0.0], [1.0], [2.0], [3.0]]
  def line_targets = [1.0, 3.0, 5.0, 7.0]
  def binary = [[-2.0], [-1.0], [1.0], [2.0]]
  def binary_targets = [0, 0, 1, 1]
  def xor = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
  def xor_targets = [0, 1, 1, 0]
  def classes = [[-2.0, -1.0], [-2.0, 0.0], [-1.0, -1.0], [2.0, -1.0], [2.0, 0.0], [1.0, -1.0], [0.0, 2.0], [-0.5, 2.0], [0.5, 2.0]]
  def class_targets = [0, 0, 0, 1, 1, 1, 2, 2, 2]
  def clusters = [[0.0, 0.0], [0.0, 2.0], [2.0, 0.0], [8.0, 8.0], [8.0, 10.0], [10.0, 8.0]]
  def mushrooms = [[1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 0, 0]]
  def edible = [1, 1, 0, 0, 1, 0, 0, 1, 1, 0]
  def ratings = [[5.0, 4.0, nil], [4.0, 5.0, 1.0], [1.0, nil, 5.0], [nil, 1.0, 4.0]]
  def genres = [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]]
  def users = [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]
end
