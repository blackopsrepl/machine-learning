# frozen_string_literal: true

# Evaluation metrics — distinguish MSE from half-MSE used in course objectives.
# For rare positives, accuracy can hide failure; precision/recall/F1 expose it.
# Undefined precision/recall are reported as zero rather than NaN.
# Run me: ruby evaluation/metrics.rb
require_relative "../lib/style"
require_relative "../lib/numeric"
module Metrics
  module_function

  def binary(actual, predicted)
    tp = actual.zip(predicted).count { |a, p| a == 1 && p == 1 }
    fp = actual.zip(predicted).count { |a, p| a == 0 && p == 1 }
    fn = actual.zip(predicted).count { |a, p| a == 1 && p == 0 }
    precision = tp + fp > 0 ? tp.to_f / (tp + fp) : 0.0
    recall = tp + fn > 0 ? tp.to_f / (tp + fn) : 0.0
    f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0.0
    { tp: tp, fp: fp, fn: fn, precision: precision, recall: recall, f1: f1 }
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Evaluation Metrics", "original C2 W3 exercise values · handwritten")
  actual, predicted = [2.3, 4.1], [2.4, 4.2]
  step("half-MSE=#{NumericTools.mse(actual, predicted) / 2.0}")
  [[1, 2, 3], [1, 2, 1, 3]].zip([[1, 2, 0], [1, 2, 0, 3]]).each do |labels, estimates|
    step("labels=#{labels.inspect}; predictions=#{estimates.inspect}; categorical error=#{1 - NumericTools.accuracy(labels, estimates)}")
  end
  raise "Original MSE calculation failed" unless (NumericTools.mse(actual, predicted) / 2 - 0.005).abs < 1e-12

  ok("the course's MSE and classification-error conventions are explicit")
end
