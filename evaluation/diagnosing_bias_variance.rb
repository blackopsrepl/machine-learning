# frozen_string_literal: true

# Original optional bias/variance lab: all four CSVs, feature subsets, lambda
# sweeps and 10%-100% learning curves. No invented replacement observations.
require_relative "model_selection"
module DiagnosingBiasVariance
  module_function

  def data(number)
    rows = Reference.numpy.loadtxt(CourseData.path(:evaluation, "data/c2w3_lab2_data#{number}.csv"), delimiter: ",")
    ModelSelection.split(Reference.columns(rows, 0, -1), Reference.columns(rows, -1).reshape([-1]), seed: 80)
  end

  def report(train, train_y, cv, cv_y, degree:, lambda_: nil)
    model = ModelSelection.fit(train, train_y, degree: degree, l2: lambda_)
    [ModelSelection.half_mse(train_y, ModelSelection.predict(model, train)), ModelSelection.half_mse(cv_y, ModelSelection.predict(model, cv))]
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Diagnosing Bias and Variance", "original optional-lab CSVs · scikit-learn")
  [[1, 10], [2, 6], [3, 4]].each do |part, max_degree|
    train, train_y, cv, cv_y, _test, _test_y = DiagnosingBiasVariance.data(part)
    (1..max_degree).each do |degree|
      a, b = DiagnosingBiasVariance.report(train, train_y, cv, cv_y, degree: degree)
      step("dataset #{part}, #{Reference.shape(train)[1]} features, degree #{degree}: train=#{a}; validation=#{b}")
    end
  end
  train, train_y, cv, cv_y, *_ = DiagnosingBiasVariance.data(2)
  [10, 5, 2, 1, 0.5, 0.2, 0.1, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1].each do |lambda_|
    a, b = DiagnosingBiasVariance.report(train, train_y, cv, cv_y, degree: 4, lambda_: lambda_)
    step("degree 4, lambda #{lambda_}: train=#{a}; validation=#{b}")
  end
  train, train_y, cv, cv_y, *_ = DiagnosingBiasVariance.data(4)
  (10..100).step(10) do |percent|
    nt = (Reference.shape(train)[0] * percent / 100.0).round
    nv = (Reference.shape(cv)[0] * percent / 100.0).round
    a, b = DiagnosingBiasVariance.report(Reference.rows(train, 0, nt), Reference.rows(train_y, 0, nt), Reference.rows(cv, 0, nv), Reference.rows(cv_y, 0, nv), degree: 4)
    step("#{percent}%: train rows=#{nt}, validation rows=#{nv}, train=#{a}, validation=#{b}")
  end
  ok("the original feature and sample-size diagnosis experiments run on their CSVs")
end
