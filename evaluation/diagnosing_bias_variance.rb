# frozen_string_literal: true

# Course optional lab "Diagnosing Bias and Variance": prepare_dataset splits
# each CSV 60/20/20 with random_state=80; train_plot_poly, train_plot_reg_params,
# train_plot_diff_datasets and train_plot_learning_curve with the lab's degrees,
# λ lists and 10%…100% subsets.
# Run me: ruby evaluation/diagnosing_bias_variance.rb
require_relative "model_selection"
require_relative "../lib/course_data"
module DiagnosingBiasVariance
  module_function

  def data(number)
    rows = CourseData.table(:evaluation, "data/c2w3_lab2_data#{number}.csv")
    ModelSelection.split(rows.map { |r| r[0...-1] }, rows.map(&:last), seed: 80)
  end

  def report(train, train_y, cv, cv_y, degree:, lambda_: nil)
    ModelSelection.errors(ModelSelection.fit(train, train_y, degree: degree, l2: lambda_), train, train_y, cv, cv_y)
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Diagnosing Bias and Variance", "course lab CSVs · polynomial and Ridge fits")
  [[1, 10], [2, 6]].each do |part, max_degree|
    train, train_y, cv, cv_y, = DiagnosingBiasVariance.data(part)
    (1..max_degree).each do |degree|
      a, b = DiagnosingBiasVariance.report(train, train_y, cv, cv_y, degree: degree)
      step("data#{part} (#{train.first.size} features), degree #{degree}: train=#{a}; validation=#{b}")
    end
  end
  train, train_y, cv, cv_y, = DiagnosingBiasVariance.data(2)
  [[10, 5, 2, 1, 0.5, 0.2, 0.1], [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]].each do |lambdas|
    lambdas.each do |lambda_|
      a, b = DiagnosingBiasVariance.report(train, train_y, cv, cv_y, degree: 4, lambda_: lambda_)
      step("data2, degree 4, λ=#{lambda_}: train=#{a}; validation=#{b}")
    end
  end
  [3, 2].each do |part|
    train, train_y, cv, cv_y, = DiagnosingBiasVariance.data(part)
    (1..4).each do |degree|
      a, b = DiagnosingBiasVariance.report(train, train_y, cv, cv_y, degree: degree)
      step("data#{part} (#{train.first.size} features), degree #{degree}: train=#{a}; validation=#{b}")
    end
  end
  train, train_y, cv, cv_y, = DiagnosingBiasVariance.data(4)
  (10..100).step(10) do |percent|
    nt = (train.size * percent / 100.0).round
    nv = (cv.size * percent / 100.0).round
    a, b = DiagnosingBiasVariance.report(train.first(nt), train_y.first(nt), cv.first(nv), cv_y.first(nv), degree: 4)
    step("data4, #{percent}%: #{nt} train / #{nv} validation rows; train=#{a}; validation=#{b}")
  end
  ok("feature, degree, penalty and sample-size diagnoses on every lab dataset")
end
