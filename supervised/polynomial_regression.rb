# frozen_string_literal: true

# Polynomial regression — nonlinear features, linear coefficients.
# φ(x)=[x,x²,...,x^degree]. The bias is separate; the solver is unchanged.
# Three points on x² distinguish a straight-line model from a quadratic.
# Run me: ruby supervised/polynomial_regression.rb
require_relative "normal_equation"
module PolynomialRegression
  module_function
  def features(values, degree)
    raise ArgumentError, "positive degree required" unless degree.positive?
    values.map { |x| (1..degree).map { |power| x.to_f**power } }
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Polynomial Regression", "change the features, not the learning rule")
  x,y = [-2,-1,0,1,2], [4.0,1.0,0.0,1.0,4.0]
  losses = [1,2].map do |degree|
    features = PolynomialRegression.features(x,degree)
    w,b = NormalEquation.fit(features,y)
    cost = LinearRegression.loss(features,y,w,b)
    step("degree #{degree}: w=#{vector(w)} b=#{b.round(4)} J=#{cost.round(6)}")
    cost
  end
  raise "quadratic not recovered" unless losses[0] > 0.1 && losses[1] < 1e-12
  ok("the quadratic basis explains curvature that a line cannot")
end
