# frozen_string_literal: true

# Course tree-ensemble lab: original heart disease data, categorical one-hot
# encoding, 80/20 split(seed 55), actual sklearn CART and RandomForestClassifier.
# Preserve the parameter sweeps and the final 100-tree/depth-8/min-split-10 model.
require_relative "../lib/style"
require_relative "../lib/reference"
module RandomForest
  module_function

  def fit(x, y, count: 100, max_depth: 8, min_samples_split: 10, random_state: 55)
    model = PyCall.import_module("sklearn.ensemble").RandomForestClassifier.new(n_estimators: count, max_depth: max_depth, min_samples_split: min_samples_split, random_state: random_state)
    model.fit(x, y)
    model
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Heart Disease Trees and Random Forest", "original heart.csv · scikit-learn · seed 55")
  x_train, x_test, y_train, y_test = Reference.heart_data
  step("training #{Reference.shape(x_train).inspect}; held-out #{Reference.shape(x_test).inspect}")
  tree_class = PyCall.import_module("sklearn.tree").DecisionTreeClassifier
  [["min_samples_split", [2, 10, 30, 50, 100, 200, 300, 700]], ["max_depth", [1, 2, 3, 4, 8, 16, 32, 64, nil]]].each do |name, values|
    values.each do |value|
      model = tree_class.new(**{ name.to_sym => value, random_state: 55 })
      model.fit(x_train, y_train)
      step("CART #{name}=#{value.inspect}: train=#{model.score(x_train, y_train)} test=#{model.score(x_test, y_test)}")
    end
  end
  tree = tree_class.new(min_samples_split: 50, max_depth: 3, random_state: 55)
  tree.fit(x_train, y_train)
  step("final CART: train=#{tree.score(x_train, y_train)} test=#{tree.score(x_test, y_test)}")
  forest_class = PyCall.import_module("sklearn.ensemble").RandomForestClassifier
  [["min_samples_split", [2, 10, 30, 50, 100, 200, 300, 700]], ["max_depth", [2, 4, 8, 16, 32, 64, nil]], ["n_estimators", [10, 50, 100, 500]]].each do |name, values|
    values.each do |value|
      model = forest_class.new(**{ name.to_sym => value, random_state: 55 })
      model.fit(x_train, y_train)
      step("forest #{name}=#{value.inspect}: train=#{model.score(x_train, y_train)} test=#{model.score(x_test, y_test)}")
    end
  end
  model = RandomForest.fit(x_train, y_train)
  score = Reference.scalar(model.score(x_test, y_test))
  step("final forest: 100 trees, max_depth=8, min_samples_split=10; train=#{model.score(x_train, y_train)} test=#{score}")
  raise "Heart disease forest failed" unless score >= 0.75

  ok("the actual library estimators and course parameter sweeps are retained")
end
