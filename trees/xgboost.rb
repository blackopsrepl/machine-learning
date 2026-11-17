# frozen_string_literal: true

# XGBoost's core binary boosting algorithm — second-order regularized trees.
# For logistic loss g=p-y, h=p(1-p); leaf weight=-Σg/(Σh+λ).
# Split gain=½[G_L²/(H_L+λ)+G_R²/(H_R+λ)-G²/(H+λ)]-γ.
# This exact-greedy educational implementation is not the external XGBoost engine:
# no histogram/distributed/sparse machinery. The objective and updates are explicit.
# Run me: ruby trees/xgboost.rb
require_relative "decision_tree"
require_relative "../supervised/logistic_regression"
module XGBoost
  Node=Struct.new(:value,:feature,:threshold,:left,:right,keyword_init: true)
  Model=Struct.new(:base,:trees,:rate,keyword_init: true)
  module_function
  def build(x,g,h,depth: 2,l2: 1.0,gamma: 0.0,min_child_weight: 0.01)
    total_g,total_h=g.sum,h.sum
    node=Node.new(value: -total_g/(total_h+l2))
    return node if depth.zero?
    best=nil
    x[0].each_index do |feature|
      x.map { |row| row[feature] }.uniq.sort.each_cons(2) do |a,b|
        threshold=(a+b)/2.0
        left,right=x.each_index.partition { |i| x[i][feature]<=threshold }
        gl,gr=left.sum { |i| g[i] },right.sum { |i| g[i] }
        hl,hr=left.sum { |i| h[i] },right.sum { |i| h[i] }
        next if hl<min_child_weight || hr<min_child_weight
        gain=0.5*(gl**2/(hl+l2)+gr**2/(hr+l2)-total_g**2/(total_h+l2))-gamma
        best=[gain,feature,threshold,left,right] if gain>0 && (!best || gain>best[0])
      end
    end
    return node unless best
    _gain,node.feature,node.threshold,left,right=best
    options={depth: depth-1,l2: l2,gamma: gamma,min_child_weight: min_child_weight}
    node.left=build(left.map { |i| x[i] },left.map { |i| g[i] },left.map { |i| h[i] },**options)
    node.right=build(right.map { |i| x[i] },right.map { |i| g[i] },right.map { |i| h[i] },**options)
    node
  end
  def value(tree,row)
    tree=row[tree.feature]<=tree.threshold ? tree.left : tree.right while tree.feature
    tree.value
  end
  def logit(model,row) = model.base+model.rate*model.trees.sum { |tree| value(tree,row) }
  def probability(model,row) = LogisticRegression.sigmoid(logit(model,row))
  def loss(model,x,y) = x.zip(y).sum { |row,t| z=logit(model,row); LogisticRegression.softplus(z)-t*z }.to_f/x.size
  def fit(x,y,rounds: 30,rate: 0.3,depth: 2,l2: 1.0,gamma: 0.0,validation: nil,patience: 5)
    prevalence=NumericTools.mean(y).clamp(1e-6,1-1e-6)
    model=Model.new(base: Math.log(prevalence/(1-prevalence)),trees: [],rate: rate)
    best_cost,best_count,stale=Float::INFINITY,0,0
    rounds.times do |iteration|
      p=x.map { |row| probability(model,row) }
      g=p.zip(y).map { |pred,t| pred-t }; h=p.map { |pred| pred*(1-pred) }
      tree=build(x,g,h,depth: depth,l2: l2,gamma: gamma)
      model.trees << tree
      cost=loss(model,x,y)
      yield iteration+1,cost,tree,g,h if block_given?
      next unless validation
      validation_cost=loss(model,*validation)
      if validation_cost<best_cost-1e-10
        best_cost,best_count,stale=validation_cost,model.trees.size,0
      else
        stale+=1
        break if stale>=patience
      end
    end
    model.trees=model.trees.first(best_count) if validation
    model
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("XGBoost", "logistic gradients + Hessians + regularized leaf weights")
  x,y=Problem.mushrooms,Problem.edible
  model=XGBoost.fit(x,y) do |round,cost,tree,g,h|
    next unless round<=3 || (round%10).zero?
    step(format("round %-2d BCE=%.6f root weight=% .4f split=%s",round,cost,tree.value,tree.feature.inspect))
    note("gradients #{vector(g)}; Hessians #{vector(h)}") if round==1
  end
  predicted=x.map { |row| XGBoost.probability(model,row)>=0.5 ? 1 : 0 }
  raise "boosting failed" unless predicted==y && XGBoost.loss(model,x,y)<0.2
  stopped=XGBoost.fit(Problem.binary,Problem.binary_targets,rounds: 40,validation: [Problem.binary,Problem.binary_targets.map { |v| 1-v }],patience: 2)
  raise "validation did not restore best prefix" unless stopped.trees.size==1
  step("training accuracy #{NumericTools.accuracy(y,predicted)}; held-out loss can stop and restore an earlier tree prefix")
  ok("additive second-order trees fit the nonlinear data without an estimator library")
end
