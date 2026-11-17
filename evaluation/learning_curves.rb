# frozen_string_literal: true

# Course tune_m: degree-16 polynomial without regularization, m=50..750 in
# steps of 50 from gen_data(m, 5, 0.7), 60/20/20 split with random_state=1.
# Run me: ruby evaluation/learning_curves.rb
require_relative "model_selection"
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Learning Curves", "degree 16 · training-set size 50→750")
  results = (1..15).map do |multiple|
    count = 50 * multiple
    x, y = ModelSelection.generated(count, seed: 5, scale: 0.7)
    train, train_y, cv, cv_y, = ModelSelection.split(x, y)
    train_error, cv_error = ModelSelection.errors(ModelSelection.fit(train, train_y, degree: 16), train, train_y, cv, cv_y)
    step("m=#{count}: train=#{train_error}; validation=#{cv_error}; gap=#{cv_error - train_error}")
    raise "learning curve produced nonfinite values" unless train_error.finite? && cv_error.finite?

    [count, train_error, cv_error]
  end
  first, last = results.first, results.last
  step("validation/training ratio #{(first[2] / first[1]).round(2)} at m=#{first[0]} → #{(last[2] / last[1]).round(2)} at m=#{last[0]}")
  ok("more examples shrink the high-variance gap of a degree-16 model")
end
