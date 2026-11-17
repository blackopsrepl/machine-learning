# frozen_string_literal: true

# Original tune_m helper: degree 16, m=50..750 by 50, seed 5/noise .7,
# 60/20/20 split(seed 1), unregularized polynomial/scaled LinearRegression.
require_relative "model_selection"
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Learning Curves", "original degree-16 sample-size experiment")
  (1..15).each do |multiple|
    count = 50 * multiple
    x, y = ModelSelection.generated(count, seed: 5, scale: 0.7)
    train, train_y, cv, cv_y, _test, _test_y = ModelSelection.split(x, y)
    fit = ModelSelection.fit(train, train_y, degree: 16)
    train_cost = ModelSelection.half_mse(train_y, ModelSelection.predict(fit, train))
    cv_cost = ModelSelection.half_mse(cv_y, ModelSelection.predict(fit, cv))
    step("m=#{count}: training half-MSE=#{train_cost}; validation half-MSE=#{cv_cost}")
    raise "Learning curve produced nonfinite values" unless train_cost.finite? && cv_cost.finite?
  end
  ok("all fifteen original training-set sizes are measured")
end
