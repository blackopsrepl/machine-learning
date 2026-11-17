# frozen_string_literal: true

require_relative "course_data"
require_relative "sklearn_data"

# The tree-ensemble lab's preprocessing: read heart.csv, one-hot encode the
# five categorical columns like pandas.get_dummies (remaining columns first,
# then <column>_<category> in sorted category order), drop HeartDisease from
# the features, and split 80/20 with train_test_split(random_state=55).
module HeartData
  CATEGORICAL = %w[Sex ChestPainType RestingECG ExerciseAngina ST_Slope].freeze
  TARGET = "HeartDisease"

  module_function

  def frame
    header, rows = CourseData.csv(:heart, "heart.csv")
    numeric = header - CATEGORICAL
    dummies = CATEGORICAL.flat_map do |column|
      index = header.index(column)
      rows.map { |row| row[index] }.uniq.sort.map { |category| [column, category] }
    end
    columns = numeric + dummies.map { |column, category| "#{column}_#{category}" }
    table = rows.map do |row|
      numeric.map { |name| Float(row[header.index(name)]) } + dummies.map { |column, category| row[header.index(column)] == category ? 1.0 : 0.0 }
    end
    [columns, table]
  end

  def features
    columns, table = frame
    target = columns.index(TARGET)
    names = columns.reject { |name| name == TARGET }
    [names, table.map { |row| row.reject.with_index { |_v, j| j == target } }, table.map { |row| row[target].to_i }]
  end

  def split
    names, x, y = features
    [names, *SklearnData.train_test_split(x, y, train_size: 0.8, random_state: 55)]
  end
end
