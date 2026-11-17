# frozen_string_literal: true

# C2 W3 advice: the original noisy quadratic generator, splits, degree sweep
# and Ridge sweep. Actual sklearn PolynomialFeatures/StandardScaler/estimators.
require_relative "../lib/style"
require_relative "../lib/reference"
module ModelSelection
  module_function

  def generated(m, seed: 1, scale: 0.7)
    np = Reference.numpy
    x = np.linspace(0, 49, m)
    np.random.seed(seed)
    ideal = np.square(x)
    noise = np.subtract(np.random.sample([m]), 0.5)
    [x, np.add(ideal, np.multiply(scale, np.multiply(ideal, noise)))]
  end

  def split(x, y, seed: 1)
    method = PyCall.import_module("sklearn.model_selection")
    train, other, train_y, other_y = method.train_test_split(x, y, test_size: 0.4, random_state: seed).to_a
    cv, test, cv_y, test_y = method.train_test_split(other, other_y, test_size: 0.5, random_state: seed).to_a
    [train, train_y, cv, cv_y, test, test_y]
  end

  def fit(x, y, degree:, l2: nil)
    x = Reference.array(x)
    x = x.reshape([-1, 1]) if Reference.shape(x).size == 1
    prep = PyCall.import_module("sklearn.preprocessing")
    polynomial = prep.PolynomialFeatures.new(degree, include_bias: false)
    scaler = prep.StandardScaler.new
    features = scaler.fit_transform(polynomial.fit_transform(x))
    estimators = PyCall.import_module("sklearn.linear_model")
    model = l2.nil? ? estimators.LinearRegression.new : estimators.Ridge.new(alpha: l2)
    model.fit(features, y)
    { polynomial: polynomial, scaler: scaler, model: model }
  end

  def predict(fit, x)
    x = Reference.array(x)
    x = x.reshape([-1, 1]) if Reference.shape(x).size == 1
    fit[:model].predict(fit[:scaler].transform(fit[:polynomial].transform(x)))
  end

  def half_mse(actual, predicted)
    np = Reference.numpy
    np.mean(np.square(np.subtract(actual.reshape([-1]), predicted.reshape([-1])))).item / 2.0
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Model Selection", "original C2 W3 generated datasets, degrees and regularization grid")
  x, y = ModelSelection.generated(18, seed: 2)
  x_train, x_test, y_train, y_test = PyCall.import_module("sklearn.model_selection").train_test_split(x, y, test_size: 0.33, random_state: 1).to_a
  initial = ModelSelection.fit(x_train, y_train, degree: 10)
  step("18-row degree-10 example: train half-MSE=#{ModelSelection.half_mse(y_train, ModelSelection.predict(initial, x_train))}; test=#{ModelSelection.half_mse(y_test, ModelSelection.predict(initial, x_test))}")
  x, y = ModelSelection.generated(40, seed: 5)
  train, train_y, cv, cv_y, test, test_y = ModelSelection.split(x, y)
  candidates = (1..9).map do |degree|
    model = ModelSelection.fit(train, train_y, degree: degree)
    error = ModelSelection.half_mse(cv_y, ModelSelection.predict(model, cv))
    step("degree #{degree}: train half-MSE=#{ModelSelection.half_mse(train_y, ModelSelection.predict(model, train))}; validation=#{error}")
    [error, degree, model]
  end
  error, degree, best = candidates.min_by(&:first)
  step("selected degree #{degree}: validation=#{error}; untouched test=#{ModelSelection.half_mse(test_y, ModelSelection.predict(best, test))}")
  [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100].each do |lambda_|
    model = ModelSelection.fit(train, train_y, degree: 10, l2: lambda_)
    step("degree 10, lambda #{lambda_}: train=#{ModelSelection.half_mse(train_y, ModelSelection.predict(model, train))}; validation=#{ModelSelection.half_mse(cv_y, ModelSelection.predict(model, cv))}")
  end
  raise "Selection produced a nonfinite error" unless error.finite?

  ok("the course's generated observations and complete selection grids are retained")
end
