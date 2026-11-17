# frozen_string_literal: true

# Ordinary least squares — solve (XᵀX)θ=Xᵀy, with an intercept column.
# Tiny full-rank problems only. Forming XᵀX squares the condition number:
# production implementations should use QR/SVD, not an explicit inverse.
# This is the mathematics behind the course's LinearRegression comparison.
# Run me: ruby supervised/normal_equation.rb
require_relative "linear_regression"
module NormalEquation
  module_function

  def fit(x, y, l2: 0.0)
    design = x.map { |row| [1.0] + row }
    columns = design.transpose
    gram = columns.each_with_index.map do |a, i|
      columns.each_with_index.map { |b, j| NumericTools.dot(a, b) + (i == j && i > 0 ? l2 : 0.0) }
    end
    rhs = columns.map { |column| NumericTools.dot(column, y) }
    yield gram, rhs if block_given?
    theta = NumericTools.solve(gram, rhs)
    [theta.drop(1), theta[0]]
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Ordinary Least Squares", "normal equations · pivoted linear solve")
  w, b = NormalEquation.fit(Problem.line, Problem.line_targets) do |gram, rhs|
    gram.each_with_index { |r, i| step("equation #{i}: #{vector(r)} · [b,w] = #{rhs[i]}") }
  end
  step("weights #{vector(w)}, bias #{b}")
  raise "closed-form mismatch" unless (w[0] - 2).abs < 1e-10 && (b - 1).abs < 1e-10

  begin
    NormalEquation.fit([[1.0], [1.0]], [1.0, 2.0])
    raise "singular system silently accepted"
  rescue ArgumentError
    note("rank-deficient inputs are rejected explicitly; no pretend inverse")
  end
  ok("the stationary point recovers y=1+2x without iterations")
end
