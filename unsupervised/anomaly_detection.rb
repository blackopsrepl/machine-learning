# frozen_string_literal: true

# Gaussian anomaly detection — independent feature densities, summed in log space.
# μ_j=mean(x_j), variance=mean((x_j-μ_j)²); flag log p(x)<threshold.
# Choose the threshold using labeled VALIDATION anomalies, not training accuracy.
# Run me: ruby unsupervised/anomaly_detection.rb
require_relative "../lib/style"
require_relative "../lib/numeric"
require_relative "../evaluation/metrics"
module AnomalyDetection
  module_function

  def fit(x, variance_floor: 1e-9)
    means = x.transpose.map { |column| NumericTools.mean(column) }
    variances = x.transpose.each_with_index.map { |column, j| [NumericTools.mean(column.map { |v| (v - means[j])**2 }), variance_floor].max }
    [means, variances]
  end

  def log_density(row, model)
    means, variances = model
    row.each_index.sum { |j| -0.5 * (Math.log(2 * Math::PI * variances[j]) + (row[j] - means[j])**2 / variances[j]) }
  end

  def select_threshold(scores, labels, steps: 1000)
    minimum, maximum = scores.minmax
    candidates = minimum == maximum ? [minimum] : Array.new(steps) { |i| minimum + i * (maximum - minimum) / steps }
    best_threshold, best_f1 = 0.0, 0.0
    candidates.each do |threshold|
      predicted = scores.map { |value| value < threshold ? 1 : 0 }
      metrics = Metrics.binary(labels, predicted)
      yield threshold, metrics if block_given?
      if metrics[:f1] > best_f1
        best_threshold, best_f1 = threshold, metrics[:f1]
      end
    end
    [best_threshold, best_f1]
  end
end
if __FILE__ == $PROGRAM_NAME
  require_relative "../lib/course_data"
  include Style
  banner("Gaussian Anomaly Detection", "original server datasets · 1000-step validation F1 search")
  [1, 2].each do |part|
    training = CourseData.npy(:anomaly, "data/X_part#{part}.npy")
    validation = CourseData.npy(:anomaly, "data/X_val_part#{part}.npy")
    labels = CourseData.npy(:anomaly, "data/y_val_part#{part}.npy").flatten
    model = AnomalyDetection.fit(training)
    scores = validation.map { |row| Math.exp(AnomalyDetection.log_density(row, model)) }
    threshold, f1 = AnomalyDetection.select_threshold(scores, labels)
    outliers = training.count { |row| Math.exp(AnomalyDetection.log_density(row, model)) < threshold }
    step("part #{part}: #{training.size}×#{training.first.size} training; epsilon=#{threshold}; validation F1=#{f1}; training outliers=#{outliers}")
    raise "Original anomaly example failed" unless f1 > (part == 1 ? 0.8 : 0.6)
  end
  ok("both original low- and high-dimensional server examples run unchanged in scope")
end
