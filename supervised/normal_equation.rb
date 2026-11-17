# frozen_string_literal: true

# Course C1 W2 lab: scikit-learn's LinearRegression — ordinary least squares.
# Center X and y, solve min ||Xc w - yc|| by Householder QR (never forming or
# inverting XᵀX), then intercept = mean(y) - mean(x)·w, as the estimator does.
# Run me: ruby supervised/normal_equation.rb
require_relative "../lib/style"
require_relative "../lib/numeric"
module OrdinaryLeastSquares
  module_function

  # Householder QR least squares for a full-column-rank m×n matrix (m ≥ n).
  def solve(a, b)
    a = a.map(&:dup)
    b = b.dup
    m, n = a.size, a.first.size
    raise ArgumentError, "need at least as many rows as columns" if m < n

    n.times do |k|
      norm = Math.sqrt((k...m).sum { |i| a[i][k]**2 })
      raise ArgumentError, "matrix is rank deficient" if norm < 1e-12 * (1 + a.flatten.map(&:abs).max)

      alpha = a[k][k] > 0 ? -norm : norm
      v = Array.new(m, 0.0)
      v[k] = a[k][k] - alpha
      (k + 1...m).each { |i| v[i] = a[i][k] }
      vv = (k...m).sum { |i| v[i]**2 }
      (k...n).each do |j|
        s = 2 * (k...m).sum { |i| v[i] * a[i][j] } / vv
        (k...m).each { |i| a[i][j] -= s * v[i] }
      end
      s = 2 * (k...m).sum { |i| v[i] * b[i] } / vv
      (k...m).each { |i| b[i] -= s * v[i] }
    end
    x = Array.new(n, 0.0)
    (n - 1).downto(0) { |i| x[i] = (b[i] - (i + 1...n).sum { |j| a[i][j] * x[j] }) / a[i][i] }
    x
  end

  def fit(x, y)
    means = x.transpose.map { |column| NumericTools.mean(column) }
    target_mean = NumericTools.mean(y)
    centered = x.map { |row| row.each_with_index.map { |v, j| v - means[j] } }
    weights = solve(centered, y.map { |v| v - target_mean })
    [weights, target_mean - NumericTools.dot(means, weights)]
  end
end
if __FILE__ == $PROGRAM_NAME
  require_relative "../lib/course_data"
  include Style
  banner("Ordinary Least Squares", "course housing data · scikit-learn's LinearRegression in Ruby")
  x, y = CourseData.supervised(:houses, "data/houses.txt", skip: 1)
  w, b = OrdinaryLeastSquares.fit(x, y)
  step("#{x.size}×#{x.first.size} housing table; weights #{vector(w)}; intercept #{b}")
  residual = x.zip(y).map { |row, target| NumericTools.dot(row, w) + b - target }
  gradient = w.each_index.map { |j| x.each_index.sum { |i| residual[i] * x[i][j] } }
  step("normal-equation residual Xᵀ(Xw+b-y)=#{vector(gradient)}; Σ residual=#{residual.sum}")
  prediction = NumericTools.dot([1200, 3, 1, 40], w) + b
  step("1200 sqft, 3 bedrooms, 1 floor, 40 years: $#{(prediction * 1000).round(2)}")
  scale = x.flatten.map(&:abs).max * y.map(&:abs).max * x.size
  raise "least-squares optimality failed" unless gradient.all? { |g| g.abs < 1e-9 * scale } && residual.sum.abs < 1e-9 * y.map(&:abs).sum

  ok("the house-price fit satisfies the least-squares optimality conditions")
end
