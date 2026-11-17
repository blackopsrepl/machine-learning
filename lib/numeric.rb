# frozen_string_literal: true

# Array arithmetic, not a hidden estimator. Algorithm updates stay in demo files.
module NumericTools
  module_function

  def dot(a, b)
    raise ArgumentError, "vector dimensions differ" unless a.size == b.size

    a.zip(b).sum { |x, y| x * y }
  end

  def mean(values) = values.sum.to_f / values.size
  def mse(actual, predicted) = actual.zip(predicted).sum { |a, b| (a - b)**2 }.to_f / actual.size
  def accuracy(actual, predicted) = actual.zip(predicted).count { |a, b| a == b }.to_f / actual.size

  # Population variance, matching NumPy's default; constant columns map to zero.
  def fit_scaler(rows)
    centers = rows.transpose.map { |column| mean(column) }
    scales = rows.transpose.each_with_index.map do |column, j|
      sigma = Math.sqrt(mean(column.map { |x| (x - centers[j])**2 }))
      sigma.zero? ? 1.0 : sigma
    end
    [centers, scales]
  end

  def transform(rows, scaler)
    centers, scales = scaler
    rows.map { |row| row.each_with_index.map { |x, j| (x - centers[j]) / scales[j] } }
  end

  # Partial-pivoted elimination for small nonsingular systems, not an inverse.
  def solve(matrix, rhs)
    n = rhs.size
    a = matrix.each_with_index.map { |row, i| row.map(&:to_f) + [rhs[i].to_f] }
    n.times do |column|
      pivot = (column...n).max_by { |row| a[row][column].abs }
      raise ArgumentError, "singular system" if a[pivot][column].abs < 1e-12

      a[column], a[pivot] = a[pivot], a[column]
      divisor = a[column][column]
      (column..n).each { |j| a[column][j] /= divisor }
      n.times do |row|
        next if row == column

        factor = a[row][column]
        (column..n).each { |j| a[row][j] -= factor * a[column][j] }
      end
    end
    a.map(&:last)
  end
end
