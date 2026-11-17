# frozen_string_literal: true

# Course XGBoost lab: use the real XGBClassifier, not a reduced homemade booster.
# Original heart.csv preprocessing/split; 500 rounds, eta=.1, random_state=55.
# Hold out the final 20% of training rows for early stopping with patience 50.
require_relative "../lib/style"
require_relative "../lib/reference"
module XGBoost
  module_function

  def fit(x, y, eval_x:, eval_y:)
    model = PyCall.import_module("xgboost").XGBClassifier.new(n_estimators: 500, learning_rate: 0.1, verbosity: 1, random_state: 55, n_jobs: 2)
    model.fit(x, y, eval_set: [PyCall.tuple([eval_x, eval_y])], early_stopping_rounds: 50, verbose: false)
    model
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("XGBoost Heart Disease Classifier", "actual XGBoost engine · original heart.csv")
  x_train, x_test, y_train, y_test = Reference.heart_data
  n = (Reference.shape(x_train)[0] * 0.8).to_i
  model = XGBoost.fit(Reference.rows(x_train, 0, n), Reference.rows(y_train, 0, n), eval_x: Reference.rows(x_train, n), eval_y: Reference.rows(y_train, n))
  history = model.evals_result.__getitem__("validation_0").__getitem__("logloss").to_a
  history.each_with_index { |loss, i| step("round #{i}: validation logloss=#{loss}") if i < 3 || (i % 20).zero? || i == history.size - 1 }
  prediction = model.predict(x_test)
  accuracy = Reference.accuracy(y_test, prediction)
  step("best iteration=#{model.best_iteration}; held-out accuracy=#{accuracy}; parameters=#{model.get_params}")
  raise "XGBoost heart classifier failed" unless accuracy >= 0.75

  Dir.mkdir("output") unless Dir.exist?("output")
  model.save_model("output/heart_xgboost.json")
  ok("the original library, early stopping and held-out prediction workflow are used")
end
