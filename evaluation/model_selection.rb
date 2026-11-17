# frozen_string_literal: true

# Course C2 W3 assignment: choose polynomial degree and regularization on a
# validation set. Pipeline as the course's lin_model: PolynomialFeatures
# (include_bias=False) → StandardScaler fitted on training rows →
# LinearRegression or Ridge(α), scored by half mean squared error.
# Data: gen_data(m, seed, 0.7) — x=linspace(0,49,m), y=x²(1+0.7(U−0.5)) from
# np.random.seed(seed), split 60/20/20 by train_test_split(random_state=1).
# Run me: ruby evaluation/model_selection.rb
require_relative "../lib/style"
require_relative "../lib/numeric"
require_relative "../lib/numpy_random"
require_relative "../lib/sklearn_data"
require_relative "../supervised/normal_equation"
module ModelSelection
  module_function

  def generated(m, seed: 1, scale: 0.7)
    rng = NumpyRandom::RandomState.new(seed)
    x = Array.new(m) { |i| m == 1 ? 0.0 : 49.0 * i / (m - 1) }
    [x, x.map { |v| v * v + scale * v * v * (rng.random_sample - 0.5) }]
  end

  def split(x, y, seed: 1)
    train, other, train_y, other_y = SklearnData.train_test_split(x, y, test_size: 0.4, random_state: seed)
    cv, test, cv_y, test_y = SklearnData.train_test_split(other, other_y, test_size: 0.5, random_state: seed)
    [train, train_y, cv, cv_y, test, test_y]
  end

  def rows(x) = x.map { |v| v.is_a?(Array) ? v : [v] }

  # PolynomialFeatures: every monomial of total degree 1..d, generated as
  # combinations with replacement of the feature indices, in sklearn's order.
  def polynomial(x, degree)
    x = rows(x)
    terms = (1..degree).flat_map { |d| (0...x.first.size).to_a.repeated_combination(d).to_a }
    x.map { |row| terms.map { |term| term.reduce(1.0) { |product, j| product * row[j] } } }
  end

  # Ridge (alpha>0) or ordinary least squares on centered features; the
  # penalty enters as √α·I rows appended to the least-squares system.
  def linear(x, y, alpha: 0.0)
    means = x.transpose.map { |column| NumericTools.mean(column) }
    target = NumericTools.mean(y)
    a = x.map { |row| row.each_with_index.map { |v, j| v - means[j] } }
    b = y.map { |v| v - target }
    if alpha.positive?
      a += Array.new(means.size) { |i| Array.new(means.size) { |j| i == j ? Math.sqrt(alpha) : 0.0 } }
      b += Array.new(means.size, 0.0)
    end
    weights = OrdinaryLeastSquares.solve(a, b)
    [weights, target - NumericTools.dot(means, weights)]
  end

  def fit(x, y, degree:, l2: nil)
    features = polynomial(x, degree)
    scaler = NumericTools.fit_scaler(features)
    weights, bias = linear(NumericTools.transform(features, scaler), y, alpha: l2.to_f)
    { degree: degree, scaler: scaler, weights: weights, bias: bias }
  end

  def predict(model, x)
    NumericTools.transform(polynomial(x, model[:degree]), model[:scaler]).map { |row| NumericTools.dot(row, model[:weights]) + model[:bias] }
  end

  def half_mse(actual, predicted) = actual.zip(predicted).sum { |a, p| (Array(a).first - Array(p).first)**2 } / (2.0 * actual.size)
  def errors(model, train, train_y, cv, cv_y) = [half_mse(train_y, predict(model, train)), half_mse(cv_y, predict(model, cv))]
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Model Selection", "course gen_data samples · degree and λ chosen on validation data")
  x, y = ModelSelection.generated(18, seed: 2)
  x_train, x_test, y_train, y_test = SklearnData.train_test_split(x, y, test_size: 0.33, random_state: 1)
  overfit = ModelSelection.fit(x_train, y_train, degree: 10)
  train_error = ModelSelection.half_mse(y_train, ModelSelection.predict(overfit, x_train))
  test_error = ModelSelection.half_mse(y_test, ModelSelection.predict(overfit, x_test))
  step("18 points, degree 10: train half-MSE=#{train_error}; test=#{test_error}")
  x, y = ModelSelection.generated(40, seed: 5)
  train, train_y, cv, cv_y, test, test_y = ModelSelection.split(x, y)
  step("40 points split #{train.size}/#{cv.size}/#{test.size}")
  candidates = (1..9).map do |degree|
    model = ModelSelection.fit(train, train_y, degree: degree)
    train_error, cv_error = ModelSelection.errors(model, train, train_y, cv, cv_y)
    step("degree #{degree}: train=#{train_error}; validation=#{cv_error}")
    [cv_error, degree, model]
  end
  cv_error, degree, best = candidates.min_by(&:first)
  step("selected degree #{degree}: validation=#{cv_error}; test=#{ModelSelection.half_mse(test_y, ModelSelection.predict(best, test))}")
  sweep = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100].map do |lambda_|
    model = ModelSelection.fit(train, train_y, degree: 10, l2: lambda_)
    train_error, cv_error = ModelSelection.errors(model, train, train_y, cv, cv_y)
    step("degree 10, λ=#{lambda_}: train=#{train_error}; validation=#{cv_error}")
    [cv_error, lambda_]
  end
  step("selected λ=#{sweep.min_by(&:first).last}")
  raise "training error must not rise with degree for nested least squares" unless candidates.map { |_e, d, m| ModelSelection.half_mse(train_y, ModelSelection.predict(m, train)) }.each_cons(2).all? { |a, b| b <= a * (1 + 1e-6) }

  ok("validation, not training error, picks the degree and the penalty")
end
