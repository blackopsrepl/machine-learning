# frozen_string_literal: true

# Original optional model-evaluation lab: both data_w3 CSVs, degrees 1..10,
# and the three exact regression/classification architectures from utils.py.
require_relative "model_selection"
require_relative "neural_model_selection"
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Model Evaluation and Selection", "original optional-lab CSVs · sklearn and TensorFlow")
  np, tf = Reference.numpy, Reference.tensorflow
  tf.keras.backend.set_floatx("float32")
  data = np.loadtxt(CourseData.path(:evaluation, "data/data_w3_ex1.csv"), delimiter: ",")
  x, y = Reference.columns(data, 0, 1), Reference.columns(data, 1, 2)
  train, train_y, cv, cv_y, test, test_y = ModelSelection.split(x, y)
  choices = (1..10).map do |degree|
    fit = ModelSelection.fit(train, train_y, degree: degree)
    train_cost = ModelSelection.half_mse(train_y, ModelSelection.predict(fit, train))
    cv_cost = ModelSelection.half_mse(cv_y, ModelSelection.predict(fit, cv))
    step("degree #{degree}: train half-MSE=#{train_cost}; validation=#{cv_cost}")
    [cv_cost, degree, fit]
  end
  error, degree, best = choices.min_by(&:first)
  step("selected degree #{degree}: validation=#{error}; test=#{ModelSelection.half_mse(test_y, ModelSelection.predict(best, test))}")
  widths = [[25, 15, 1], [20, 12, 12, 20, 1], [32, 16, 8, 4, 12, 1]]
  Reference.seed(20)
  scaler = PyCall.import_module("sklearn.preprocessing").StandardScaler.new
  scaled_train = scaler.fit_transform(train)
  scaled_cv, scaled_test = scaler.transform(cv), scaler.transform(test)
  regression_epochs = Integer(ENV.fetch("ML_EPOCHS", 300))
  models = widths.map { |layers| NeuralModelSelection.build(layers, input_size: 1) }
  models.each_with_index do |model, i|
    layers = widths[i]
    model.compile(loss: "mse", optimizer: tf.keras.optimizers.Adam.new(learning_rate: 0.1))
    model.fit(scaled_train, train_y, epochs: regression_epochs, batch_size: 32, verbose: 0)
    step("regression network #{i + 1} #{layers.inspect}: #{regression_epochs} epochs; train half-MSE=#{ModelSelection.half_mse(train_y, model.predict(scaled_train, verbose: 0))}; validation=#{ModelSelection.half_mse(cv_y, model.predict(scaled_cv, verbose: 0))}")
    model
  end
  step("course-selected regression model 3: test half-MSE=#{ModelSelection.half_mse(test_y, models[2].predict(scaled_test, verbose: 0))}")
  data = np.loadtxt(CourseData.path(:evaluation, "data/data_w3_ex2.csv"), delimiter: ",")
  x, y = Reference.columns(data, 0, -1), Reference.columns(data, -1)
  train, train_y, cv, cv_y, test, test_y = ModelSelection.split(x, y)
  scaler = PyCall.import_module("sklearn.preprocessing").StandardScaler.new
  scaled_train = scaler.fit_transform(train)
  scaled_cv, scaled_test = scaler.transform(cv), scaler.transform(test)
  Reference.seed(20)
  classification_epochs = Integer(ENV.fetch("ML_EPOCHS", 200))
  classify = lambda { |model, rows| np.greater_equal(tf.math.sigmoid(model.predict(rows, verbose: 0)).numpy, 0.5).astype("int32") }
  models = widths.map { |layers| NeuralModelSelection.build(layers, input_size: Reference.shape(train)[1]) }
  models.each_with_index do |model, i|
    layers = widths[i]
    model.compile(loss: tf.keras.losses.BinaryCrossentropy.new(from_logits: true), optimizer: tf.keras.optimizers.Adam.new(learning_rate: 0.01))
    model.fit(scaled_train, train_y, epochs: classification_epochs, batch_size: 32, verbose: 0)
    step("classification network #{i + 1} #{layers.inspect}: #{classification_epochs} epochs; train error=#{1 - Reference.accuracy(train_y, classify.call(model, scaled_train))}; validation=#{1 - Reference.accuracy(cv_y, classify.call(model, scaled_cv))}")
    model
  end
  step("course-selected classification model 3: test error=#{1 - Reference.accuracy(test_y, classify.call(models[2], scaled_test))}")
  ok("both original CSV applications and every original candidate architecture are retained")
end
