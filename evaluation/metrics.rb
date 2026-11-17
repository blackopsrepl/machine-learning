# frozen_string_literal: true

# Evaluation metrics — distinguish MSE from half-MSE used in course objectives.
# For rare positives, accuracy can hide failure; precision/recall/F1 expose it.
# Undefined precision/recall are reported as zero rather than NaN.
# Run me: ruby evaluation/metrics.rb
require_relative "../lib/style"
require_relative "../lib/numeric"
module Metrics
  module_function
  def binary(actual,predicted)
    tp=actual.zip(predicted).count { |a,p| a==1 && p==1 }
    fp=actual.zip(predicted).count { |a,p| a==0 && p==1 }
    fn=actual.zip(predicted).count { |a,p| a==1 && p==0 }
    precision=tp+fp>0 ? tp.to_f/(tp+fp) : 0.0
    recall=tp+fn>0 ? tp.to_f/(tp+fn) : 0.0
    f1=precision+recall>0 ? 2*precision*recall/(precision+recall) : 0.0
    {tp: tp,fp: fp,fn: fn,precision: precision,recall: recall,f1: f1}
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Evaluation Metrics", "accuracy can reward a useless rare-event detector")
  labels=[0,0,0,0,0,0,0,0,0,1]
  [[0]*10,[0,0,0,0,0,0,0,0,1,1]].each do |prediction|
    m=Metrics.binary(labels,prediction)
    step("predictions #{prediction.inspect}: accuracy=#{NumericTools.accuracy(labels,prediction)} precision=#{m[:precision]} recall=#{m[:recall]} F1=#{m[:f1].round(4)}")
  end
  mse=NumericTools.mse([1.0,3.0],[2.0,1.0])
  step("residuals [1,-2]: MSE=#{mse}; course half-MSE=#{mse/2}")
  raise "wrong metrics" unless mse==2.5 && Metrics.binary([1,0],[0,0])[:f1]==0
  ok("false negatives are visible even when accuracy looks respectable")
end
