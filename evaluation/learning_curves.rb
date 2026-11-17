# frozen_string_literal: true

# Learning curves — refit on growing subsets; report unpenalized train/CV error.
# A persistent gap suggests variance; high errors on both suggest bias.
# More data is not guaranteed to help every particular sample or model.
# Run me: ruby evaluation/learning_curves.rb
require_relative "model_selection"
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Learning Curves", "training size and capacity are different levers")
  x = [-1.0, 0.0, 1.0, -0.5, 0.5, -0.75, 0.75, -0.25, 0.25]
  y = x.map { |v| v * v }
  validation = [-0.9, -0.3, 0.3, 0.9]
  truth = validation.map { |v| v * v }
  results = {}
  [1, 2].each do |degree|
    [3, 5, 9].each do |size|
      model = ModelSelection.fit(x.first(size), y.first(size), degree: degree)
      train = NumericTools.mse(y.first(size), ModelSelection.predict(model, x.first(size)))
      cv = NumericTools.mse(truth, ModelSelection.predict(model, validation))
      results[[degree, size]] = cv
      step(format("degree=%d m=%d train MSE=%.6f validation MSE=%.6f", degree, size, train, cv))
    end
  end
  raise "bias example failed" unless results[[1, 9]] > 0.01 && results[[2, 9]] < 1e-12

  ok("more rows do not turn a line into a quadratic; changing the basis does")
end
