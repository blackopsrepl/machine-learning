# frozen_string_literal: true

# K-means — alternate nearest-center assignment and per-cluster mean updates.
# Each step cannot increase squared distance. Empty clusters keep their center.
# Random initialization chooses actual rows; a local solution is not a guarantee.
# Run me: ruby unsupervised/k_means.rb
require_relative "../lib/style"
require_relative "../lib/problem"
require_relative "../lib/numeric"
module KMeans
  module_function
  def distance(a,b) = a.zip(b).sum { |x,y| (x-y)**2 }
  def assign(x,centers) = x.map { |row| centers.each_index.min_by { |k| distance(row,centers[k]) } }
  def objective(x,centers,ids) = x.each_index.sum { |i| distance(x[i],centers[ids[i]]) }.to_f/x.size
  def fit(x,k,initial: nil,iterations: 30,rng: Random.new(42))
    raise ArgumentError,"K must be between one and row count" unless k.between?(1,x.size)
    centers=(initial || x.sample(k,random: rng)).map(&:dup)
    iterations.times do |iteration|
      ids=assign(x,centers)
      old=centers
      centers=centers.each_index.map do |cluster|
        rows=x.each_index.select { |i| ids[i]==cluster }.map { |i| x[i] }
        rows.empty? ? old[cluster].dup : rows.transpose.map { |column| NumericTools.mean(column) }
      end
      current_ids=assign(x,centers)
      yield iteration+1,centers.map(&:dup),current_ids,objective(x,centers,current_ids) if block_given?
      break if centers==old
    end
    [centers,assign(x,centers)]
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("K-means", "assignment → means · Random.new(42)")
  costs=[]
  _centers,ids=KMeans.fit(Problem.clusters,2) do |iteration,c,assignments,cost|
    costs << cost
    step("it #{iteration}: assignments #{assignments.inspect}; mean squared distance #{cost.round(6)}")
    c.each_with_index { |center,k| note("center #{k}: #{vector(center)}") }
  end
  raise "objective increased" unless costs.each_cons(2).all? { |a,b| b<=a+1e-12 }
  raise "clusters not separated" unless ids.first(3).uniq.size==1 && ids.last(3).uniq.size==1 && ids[0]!=ids[-1]
  empty,_=KMeans.fit([[0.0],[1.0]],2,initial: [[0.0],[0.0]])
  raise "empty cluster produced NaN" unless empty.flatten.all?(&:finite?)
  ok("the two clouds separate through alternating minimization")
end
