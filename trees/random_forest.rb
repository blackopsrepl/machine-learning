# frozen_string_literal: true

# Course tree-ensemble lab: heart.csv one-hot encoded and split 80/20 with
# random_state=55; DecisionTreeClassifier sweeps of min_samples_split and
# max_depth, the final tree (min_samples_split=50, max_depth=3), forest
# sweeps, and the final forest of 100 trees, max_depth=8, min_samples_split=10.
# RandomForestClassifier as scikit-learn builds it: each tree gets
# random_state=RandomState(55).randint(2³¹−1) in turn, trains on bootstrap
# counts drawn from RandomState(tree_seed) as sample weights, samples
# ⌊√20⌋=4 features per node, and the forest averages leaf class proportions.
# The notebook leaves its final forest unseeded; it uses random_state=55 here.
# Run me: ruby trees/random_forest.rb
require_relative "cart"
require_relative "../lib/heart_data"
module RandomForest
  module_function

  def fit(x, y, n_estimators: 100, max_depth: nil, min_samples_split: 2, random_state: 55)
    seeds = NumpyRandom::RandomState.new(random_state)
    max_features = [Math.sqrt(x.first.size).floor, 1].max
    Array.new(n_estimators) do
      seed = seeds.interval(2_147_483_646)
      draws = NumpyRandom::RandomState.new(seed)
      counts = Array.new(x.size, 0.0)
      x.size.times { counts[draws.interval(x.size - 1)] += 1 }
      CART.fit(x, y, max_depth: max_depth, min_samples_split: min_samples_split, max_features: max_features, random_state: seed, sample_weight: counts)
    end
  end

  def proba(forest, row)
    sums = forest.map { |tree| CART.proba(tree, row) }.transpose.map(&:sum)
    sums.map { |s| s / forest.size }
  end

  def predict(forest, row)
    p = proba(forest, row)
    p.index(p.max)
  end

  def accuracy(forest, x, y) = x.zip(y).count { |row, t| predict(forest, row) == t }.to_f / y.size
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Heart Disease Trees and Random Forest", "course heart.csv · one-hot features · random_state 55")
  names, x_train, x_test, y_train, y_test = HeartData.split
  step("#{names.size} features after one-hot encoding; #{x_train.size} training / #{x_test.size} test rows; positive rate #{(y_train.sum.to_f / y_train.size).round(4)}")
  [[:min_samples_split, [2, 10, 30, 50, 100, 200, 300, 700]], [:max_depth, [1, 2, 3, 4, 8, 16, 32, 64, nil]]].each do |name, values|
    values.each do |value|
      tree = CART.fit(x_train, y_train, name => value, random_state: 55)
      step("tree #{name}=#{value.inspect}: train=#{CART.accuracy(tree, x_train, y_train).round(4)} test=#{CART.accuracy(tree, x_test, y_test).round(4)}")
    end
  end
  tree = CART.fit(x_train, y_train, min_samples_split: 50, max_depth: 3, random_state: 55)
  step("final tree (min_samples_split=50, max_depth=3): train=#{CART.accuracy(tree, x_train, y_train).round(4)} test=#{CART.accuracy(tree, x_test, y_test).round(4)}")
  [[:min_samples_split, [2, 10, 30, 50, 100, 200, 300, 700]], [:max_depth, [2, 4, 8, 16, 32, 64, nil]], [:n_estimators, [10, 50, 100, 500]]].each do |name, values|
    values.each do |value|
      forest = RandomForest.fit(x_train, y_train, name => value)
      step("forest #{name}=#{value.inspect}: train=#{RandomForest.accuracy(forest, x_train, y_train).round(4)} test=#{RandomForest.accuracy(forest, x_test, y_test).round(4)}")
    end
  end
  forest = RandomForest.fit(x_train, y_train, n_estimators: 100, max_depth: 8, min_samples_split: 10)
  train, test = RandomForest.accuracy(forest, x_train, y_train), RandomForest.accuracy(forest, x_test, y_test)
  step("final forest (100 trees, max_depth=8, min_samples_split=10): train=#{train.round(4)} test=#{test.round(4)}")
  raise "the forest does not fit its training data better than chance" unless train > 0.5

  ok("bootstrap trees with feature sampling, compared with single trees")
end
