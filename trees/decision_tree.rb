# frozen_string_literal: true

# Decision tree — choose the split with largest entropy reduction, then recurse.
# H(y)=-Σp log2 p; gain=parent H - weighted child H. Pure/no-gain nodes stop.
# Binary mushroom features match the course exercise; numeric thresholds also work.
# Run me: ruby trees/decision_tree.rb
require_relative "../lib/style"
require_relative "../lib/problem"
require_relative "../lib/numeric"
module DecisionTree
  Node=Struct.new(:feature,:threshold,:left,:right,:counts,keyword_init: true)
  module_function
  def entropy(labels)
    return 0.0 if labels.empty?
    labels.tally.values.sum { |n| p=n.to_f/labels.size; -p*Math.log2(p) }
  end
  def fit(x,y,max_depth: 3,min_leaf: 1,max_features: nil,rng: Random.new(42),depth: 0,&trace)
    raise ArgumentError,"nonempty aligned rows required" if x.empty? || x.size!=y.size
    node=Node.new(counts: y.tally)
    return node if depth>=max_depth || node.counts.size==1
    features=(0...x[0].size).to_a
    features=features.sample(max_features,random: rng) if max_features
    best=nil
    features.sort.each do |feature|
      x.map { |row| row[feature] }.uniq.sort.each_cons(2) do |a,b|
        threshold=(a+b)/2.0
        left,right=x.each_index.partition { |i| x[i][feature]<=threshold }
        next if [left.size,right.size].min<min_leaf
        gain=entropy(y)-[left,right].sum { |ids| ids.size.to_f/y.size*entropy(ids.map { |i| y[i] }) }
        trace&.call(:candidate,depth,feature,threshold,gain)
        best=[gain,feature,threshold,left,right] if gain>1e-12 && (!best || gain>best[0]+1e-12)
      end
    end
    return node unless best
    gain,node.feature,node.threshold,left,right=best
    trace&.call(:chosen,depth,node.feature,node.threshold,gain)
    options={max_depth: max_depth,min_leaf: min_leaf,max_features: max_features,rng: rng,depth: depth+1}
    node.left=fit(left.map { |i| x[i] },left.map { |i| y[i] },**options,&trace)
    node.right=fit(right.map { |i| x[i] },right.map { |i| y[i] },**options,&trace)
    node
  end
  def leaf(node,row)
    node=row[node.feature]<=node.threshold ? node.left : node.right while node.feature
    node
  end
  def predict(node,row) = leaf(node,row).counts.max_by { |label,count| [count,-label] }[0]
  def probability(node,row)
    counts=leaf(node,row).counts
    counts.fetch(1,0).to_f/counts.values.sum
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Decision Tree", "entropy → information gain → recursive partition")
  x,y=Problem.mushrooms,Problem.edible
  tree=DecisionTree.fit(x,y) do |event,depth,feature,threshold,gain|
    step(format("%-9s depth=%d feature=%d ≤ %.1f information gain=%.6f",event,depth,feature,threshold,gain))
  end
  predicted=x.map { |row| DecisionTree.predict(tree,row) }
  step("predictions #{predicted.inspect}; targets #{y.inspect}")
  raise "tree did not fit" unless predicted==y
  pure=DecisionTree.fit([[0],[1]],[1,1]); constant=DecisionTree.fit([[0],[0]],[0,1])
  raise "stopping rule failed" if pure.feature || constant.feature
  ok("all mushrooms classified; pure and unsplittable nodes become leaves")
end
