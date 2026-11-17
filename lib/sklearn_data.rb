# frozen_string_literal: true

require_relative "numpy_random"

# scikit-learn's dataset helpers that the course calls with a fixed
# random_state. Both consume NumPy's RandomState exactly as scikit-learn 1.3
# does, so the same seeds give the course's samples and splits.
module SklearnData
  module_function

  # make_blobs with explicit centers: n//k samples per center (the first
  # n%k centers get one more), drawn row by row as N(center, std²), then
  # shuffled with RandomState.shuffle on an index array.
  def make_blobs(n_samples:, centers:, cluster_std: 1.0, random_state:, shuffle: true)
    rng = NumpyRandom::RandomState.new(random_state)
    counts = Array.new(centers.size, n_samples / centers.size)
    (n_samples % centers.size).times { |i| counts[i] += 1 }
    x, y = [], []
    centers.each_with_index do |center, label|
      counts[label].times do
        x << center.map { |c| rng.normal(c, cluster_std) }
        y << label
      end
    end
    return [x, y] unless shuffle

    order = rng.shuffle!((0...n_samples).to_a)
    [order.map { |i| x[i] }, order.map { |i| y[i] }]
  end

  # train_test_split: one RandomState permutation; the first n_test indices
  # form the test set, the next n_train the training set.
  def train_test_split(*arrays, test_size: nil, train_size: nil, random_state:)
    n = arrays.first.size
    raise ArgumentError, "arrays differ in length" unless arrays.all? { |a| a.size == n }

    if test_size
      n_test = (test_size * n).ceil
      n_train = train_size ? (train_size * n).floor : n - n_test
    else
      n_train = (train_size * n).floor
      n_test = n - n_train
    end
    permutation = NumpyRandom::RandomState.new(random_state).permutation(n)
    test, train = permutation.first(n_test), permutation[n_test, n_train]
    arrays.flat_map { |a| [train.map { |i| a[i] }, test.map { |i| a[i] }] }
  end
end
