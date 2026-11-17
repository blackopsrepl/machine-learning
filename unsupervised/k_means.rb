# frozen_string_literal: true

# K-means — alternate nearest-center assignment and per-cluster mean updates.
# Each step cannot increase squared distance. Empty clusters keep their center.
# Random initialization chooses actual rows; a local solution is not a guarantee.
# Run me: ruby unsupervised/k_means.rb
require_relative "../lib/style"
require_relative "../lib/numeric"
module KMeans
  module_function

  def distance(a, b) = a.zip(b).sum { |x, y| (x - y)**2 }
  def assign(x, centers) = x.map { |row| centers.each_index.min_by { |k| distance(row, centers[k]) } }
  def objective(x, centers, ids) = x.each_index.sum { |i| distance(x[i], centers[ids[i]]) }.to_f / x.size

  def fit(x, k, initial: nil, iterations: 30, rng: Random.new(42))
    raise ArgumentError, "K must be between one and row count" unless k.between?(1, x.size)

    centers = (initial || x.sample(k, random: rng)).map(&:dup)
    iterations.times do |iteration|
      ids = assign(x, centers)
      old = centers
      centers = centers.each_index.map do |cluster|
        rows = x.each_index.select { |i| ids[i] == cluster }.map { |i| x[i] }
        rows.empty? ? old[cluster].dup : rows.transpose.map { |column| NumericTools.mean(column) }
      end
      current_ids = assign(x, centers)
      yield iteration + 1, centers.map(&:dup), current_ids, objective(x, centers, current_ids) if block_given?
    end
    [centers, assign(x, centers)]
  end
end
if __FILE__ == $PROGRAM_NAME
  require_relative "../lib/course_data"
  include Style
  banner("K-means", "original 300-point course dataset · three clusters · ten iterations")
  x = CourseData.npy(:kmeans, "data/ex7_X.npy")
  centers, ids = KMeans.fit(x, 3, initial: [[3.0, 3.0], [6.0, 2.0], [8.0, 5.0]], iterations: 10) do |iteration, c, assignments, cost|
    step("iteration #{iteration}: mean squared distance #{cost}")
    c.each_with_index { |center, k| note("centroid #{k}: #{vector(center)}; assigned #{assignments.count(k)}") }
  end
  raise "Original dataset shape changed" unless x.size == 300 && centers.size == 3 && ids.size == x.size

  ok("all original observations participate in the course clustering run")
end
