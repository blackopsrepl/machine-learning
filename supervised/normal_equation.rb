# frozen_string_literal: true

# Original Python OLS lab: sklearn LinearRegression on the course housing data.
# Keep the reference estimator and numerical solver, not an invented inverse.
# The same four housing columns and 1200-sqft prediction request are retained.
require_relative "linear_regression"
if __FILE__ == $PROGRAM_NAME
  require_relative "../lib/reference"
  include Style
  banner("Ordinary Least Squares", "course housing data · sklearn LinearRegression")
  np = Reference.numpy
  rows = np.loadtxt(CourseData.path(:houses, "data/houses.txt"), delimiter: ",", skiprows: 1)
  x = rows.__getitem__(PyCall.tuple([PyCall::Slice.new(nil), PyCall::Slice.new(0, 4)]))
  y = rows.__getitem__(PyCall.tuple([PyCall::Slice.new(nil), -1]))
  model = PyCall.import_module("sklearn.linear_model").LinearRegression.new
  model.fit(x, y)
  step("#{Reference.shape(x).inspect} housing table; weights #{model.coef_}; intercept #{model.intercept_}")
  prediction = model.predict(np.asarray([[1200, 3, 1, 40]])).__getitem__(0).item
  step("1200 sqft, 3 bedrooms, 1 floor, 40 years: $#{(prediction * 1000).round(2)}")
  raise "OLS prediction is not finite" unless prediction.finite?

  ok("the original estimator and house-price application are retained")
end
