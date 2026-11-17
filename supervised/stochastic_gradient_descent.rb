# frozen_string_literal: true

# Original Python SGD lab: sklearn StandardScaler and SGDRegressor(max_iter=1000)
# on the actual four-feature housing data. The library-backed example remains
# separate from the handwritten batch-gradient implementation.
require_relative "linear_regression"
if __FILE__ == $PROGRAM_NAME
  require_relative "../lib/reference"
  include Style
  banner("SGDRegressor", "course housing data · scikit-learn")
  np = Reference.numpy
  rows = np.loadtxt(CourseData.path(:houses, "data/houses.txt"), delimiter: ",", skiprows: 1)
  x = rows.__getitem__(PyCall.tuple([PyCall::Slice.new(nil), PyCall::Slice.new(0, 4)]))
  y = rows.__getitem__(PyCall.tuple([PyCall::Slice.new(nil), -1]))
  scaler = PyCall.import_module("sklearn.preprocessing").StandardScaler.new
  normalized = scaler.fit_transform(x)
  model = PyCall.import_module("sklearn.linear_model").SGDRegressor.new(max_iter: 1000, random_state: 42)
  model.fit(normalized, y)
  step("#{Reference.shape(x).inspect} housing table; iterations=#{model.n_iter_}; updates=#{model.t_}")
  step("weights #{model.coef_}; intercept #{model.intercept_}")
  predicted = model.predict(normalized)
  step("first predictions #{Reference.list(predicted).first(4)}")
  raise "SGD fit is not finite" unless np.isfinite(predicted).all.item

  ok("the course estimator is trained on the actual housing records")
end
