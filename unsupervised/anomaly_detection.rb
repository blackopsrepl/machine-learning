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
  def fit(x,variance_floor: 1e-9)
    means=x.transpose.map { |column| NumericTools.mean(column) }
    variances=x.transpose.each_with_index.map { |column,j| [NumericTools.mean(column.map { |v| (v-means[j])**2 }),variance_floor].max }
    [means,variances]
  end
  def log_density(row,model)
    means,variances=model
    row.each_index.sum { |j| -0.5*(Math.log(2*Math::PI*variances[j])+(row[j]-means[j])**2/variances[j]) }
  end
  def select_threshold(scores,labels)
    candidates=[-Float::INFINITY]+scores.uniq.sort.map(&:next_float)
    candidates.map do |threshold|
      predicted=scores.map { |value| value<threshold ? 1 : 0 }
      metrics=Metrics.binary(labels,predicted)
      yield threshold,metrics if block_given?
      [metrics[:f1],threshold]
    end.max_by(&:first).reverse
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Gaussian Anomaly Detection", "log densities · validation-selected F1")
  training=[[9.0,10.0],[10.0,9.0],[11.0,10.0],[10.0,11.0]]
  validation=[[10.0,10.0],[9.5,10.5],[15.0,16.0],[3.0,4.0]]; labels=[0,0,1,1]
  model=AnomalyDetection.fit(training)
  step("means #{vector(model[0])}; variances #{vector(model[1])}")
  scores=validation.map { |row| AnomalyDetection.log_density(row,model) }
  validation.zip(scores).each { |row,p| note("#{vector(row)} log p=#{p.round(5)}") }
  threshold,f1=AnomalyDetection.select_threshold(scores,labels) { |t,m| step("threshold #{t.round(5)} precision=#{m[:precision]} recall=#{m[:recall]} F1=#{m[:f1].round(4)}") }
  step("selected log-density threshold #{threshold}; validation F1=#{f1}")
  raise "threshold selection failed" unless f1==1.0
  constant=AnomalyDetection.fit([[1.0],[1.0]])
  raise "constant variance unstable" unless AnomalyDetection.log_density([1.0],constant).finite?
  raise "empty positive class" unless AnomalyDetection.select_threshold([0.0,0.0],[0,0])[1]==0
  ok("validation separates both outliers; constant features and zero positives remain defined")
end
