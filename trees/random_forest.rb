# frozen_string_literal: true

# Random forest — bootstrap rows for each tree; sample features at EACH split.
# Average leaf probabilities, then threshold. Out-of-bag votes use only trees
# whose bootstrap never included that row; these are not training predictions.
# Run me: ruby trees/random_forest.rb
require_relative "decision_tree"
module RandomForest
  Model=Struct.new(:trees,:samples,keyword_init: true)
  module_function
  def fit(x,y,count: 41,max_depth: 5,rng: Random.new(42))
    features=[Math.sqrt(x[0].size).floor,1].max
    samples=[]
    trees=Array.new(count) do |t|
      ids=Array.new(x.size) { rng.rand(x.size) }; samples << ids
      tree=DecisionTree.fit(ids.map { |i| x[i] },ids.map { |i| y[i] },max_depth: max_depth,max_features: features,rng: rng)
      yield t+1,ids,tree if block_given?
      tree
    end
    Model.new(trees: trees,samples: samples)
  end
  def probability(model,row) = NumericTools.mean(model.trees.map { |tree| DecisionTree.probability(tree,row) })
  def oob(model,x)
    x.each_index.map do |i|
      votes=model.trees.each_index.filter_map { |t| DecisionTree.probability(model.trees[t],x[i]) unless model.samples[t].include?(i) }
      votes.empty? ? nil : NumericTools.mean(votes)
    end
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Random Forest", "bootstrap rows + per-node random features · seed 42")
  x,y=Problem.mushrooms,Problem.edible
  model=RandomForest.fit(x,y) { |t,ids,_tree| step("tree #{t}: bootstrap row indices #{ids.inspect}") if t<=3 }
  predicted=x.map { |row| RandomForest.probability(model,row)>=0.5 ? 1 : 0 }
  predicted.each_index { |i| step(format("row %d vote p(edible)=%.4f prediction=%d target=%d",i,RandomForest.probability(model,x[i]),predicted[i],y[i])) }
  oob=RandomForest.oob(model,x)
  covered=oob.each_index.reject { |i| oob[i].nil? }
  score=NumericTools.accuracy(covered.map { |i| y[i] },covered.map { |i| oob[i]>=0.5 ? 1 : 0 })
  note("out-of-bag accuracy #{score.round(3)} on #{covered.size} rows; tiny data makes this noisy")
  raise "forest failed" unless NumericTools.accuracy(y,predicted)>=0.9 && model.samples.uniq.size>1 && covered.size==x.size
  ok("different bootstraps and feature choices combine into a probability vote")
end
