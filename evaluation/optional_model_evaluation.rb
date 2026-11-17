# frozen_string_literal: true

# Course optional lab "Model Evaluation and Selection": data_w3_ex1.csv
# (degrees 1–10, scaled LinearRegression) and the lab's three networks
# 25→15→1, 20→12→12→20→1, 32→16→8→4→12→1 (ReLU, linear output) trained with
# MSE/Adam(0.1)/300 epochs on degree-1 scaled inputs; then data_w3_ex2.csv
# with BCE from logits/Adam(0.01)/200 epochs, thresholding sigmoid at 0.5.
# Splits: train_test_split 60/20/20 with random_state=1.
# Run me: ruby evaluation/optional_model_evaluation.rb
require_relative "model_selection"
require_relative "neural_model_selection"
require_relative "../lib/course_data"
module OptionalModelEvaluation
  WIDTHS = [[25, 15, 1], [20, 12, 12, 20, 1], [32, 16, 8, 4, 12, 1]].freeze

  module_function

  def models(input_size)
    WIDTHS.map { |widths| NeuralModelSelection.build(widths, input_size: input_size, seed: 20) }
  end

  def classification_error(model, x, y)
    decisions = model.logits(x).map { |(z)| 1.0 / (1 + Math.exp(-z)) >= 0.5 ? 1 : 0 }
    decisions.zip(y).count { |d, t| d != t.to_i }.to_f / y.size
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Model Evaluation and Selection", "course lab CSVs · polynomial and neural candidates")
  rows = CourseData.table(:evaluation, "data/data_w3_ex1.csv")
  x, y = rows.map { |r| [r[0]] }, rows.map(&:last)
  train, train_y, cv, cv_y, test, test_y = ModelSelection.split(x, y)
  scaler = NumericTools.fit_scaler(train)
  step("training mean #{scaler[0].first.round(2)}, standard deviation #{scaler[1].first.round(2)}")
  results = (1..10).map do |degree|
    model = ModelSelection.fit(train, train_y, degree: degree)
    a, b = ModelSelection.errors(model, train, train_y, cv, cv_y)
    step("degree #{degree}: train=#{a}; validation=#{b}")
    [b, degree, model]
  end
  error, degree, best = results.min_by(&:first)
  step("lowest validation error at degree #{degree}: #{error}; test=#{ModelSelection.half_mse(test_y, ModelSelection.predict(best, test))}")
  scaled_train, scaled_cv, scaled_test = [train, cv, test].map { |part| NumericTools.transform(part, scaler) }
  networks = OptionalModelEvaluation.models(1)
  networks.each_with_index do |model, i|
    model.compile(loss: :mse, optimizer: Adam.new(learning_rate: 0.1))
    model.fit(scaled_train, train_y.map { |v| [v] }, epochs: 300, rng: Random.new(20 + i))
    step("model #{i + 1} #{OptionalModelEvaluation::WIDTHS[i].inspect}: train=#{ModelSelection.half_mse(train_y, model.predict(scaled_train))}; validation=#{ModelSelection.half_mse(cv_y, model.predict(scaled_cv))}")
  end
  step("course-selected model 3: test=#{ModelSelection.half_mse(test_y, networks[2].predict(scaled_test))}")
  rows = CourseData.table(:evaluation, "data/data_w3_ex2.csv")
  x, y = rows.map { |r| r[0...-1] }, rows.map(&:last)
  train, train_y, cv, cv_y, test, test_y = ModelSelection.split(x, y)
  scaler = NumericTools.fit_scaler(train)
  scaled_train, scaled_cv, scaled_test = [train, cv, test].map { |part| NumericTools.transform(part, scaler) }
  probabilities, truth = [0.2, 0.6, 0.7, 0.3, 0.8], [1, 1, 1, 1, 1]
  step("lab example: predictions #{probabilities.map { |p| p >= 0.5 ? 1 : 0 }.inspect}; fraction misclassified=#{probabilities.zip(truth).count { |p, t| (p >= 0.5 ? 1 : 0) != t } / 5.0}")
  networks = OptionalModelEvaluation.models(2)
  networks.each_with_index do |model, i|
    model.compile(loss: :binary_crossentropy, optimizer: Adam.new(learning_rate: 0.01))
    model.fit(scaled_train, train_y, epochs: 200, rng: Random.new(20 + i))
    step("model #{i + 1}: train error=#{OptionalModelEvaluation.classification_error(model, scaled_train, train_y)}; validation=#{OptionalModelEvaluation.classification_error(model, scaled_cv, cv_y)}")
  end
  step("course-selected model 3: test error=#{OptionalModelEvaluation.classification_error(networks[2], scaled_test, test_y)}")
  ok("each candidate is chosen on validation data and reported once on test data")
end
