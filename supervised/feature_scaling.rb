# frozen_string_literal: true

# Feature scaling — z_j=(x_j-mean_j)/population_std_j.
# Fit statistics on training rows only; transform held-out rows unchanged.
# Standardization improves conditioning, not the information in the features.
# Run me: ruby supervised/feature_scaling.rb
require_relative "linear_regression"
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Feature Scaling", "population standard deviation · train-only statistics")
  x = [[1000.0,1.0],[2000.0,2.0],[3000.0,1.0],[4000.0,2.0]]
  y = [3.0,6.0,7.0,10.0]
  scaler = NumericTools.fit_scaler(x)
  scaled = NumericTools.transform(x,scaler)
  step("means #{vector(scaler[0])}; standard deviations #{vector(scaler[1])}")
  x.zip(scaled).each { |raw,z| note("#{vector(raw)} → #{vector(z)}") }
  initial = LinearRegression.loss(x,y,[0.0,0.0],0.0)
  w,b = LinearRegression.fit(x,y,rate: 0.1,iterations: 3)
  divergent = LinearRegression.loss(x,y,w,b)
  sw,sb = LinearRegression.fit(scaled,y,rate: 0.1,iterations: 500)
  final = LinearRegression.loss(scaled,y,sw,sb)
  step(format("same rate 0.1: raw J %.2f → %.2e; scaled J → %.8f",initial,divergent,final))
  raise "conditioning story failed" unless divergent > initial && final < 1e-8
  constant = NumericTools.transform([[3.0],[3.0]],NumericTools.fit_scaler([[3.0],[3.0]]))
  raise "constant feature" unless constant == [[0.0],[0.0]]
  ok("scaling makes a useful learning rate possible; constant columns stay finite")
end
