# frozen_string_literal: true

# Collaborative filtering — factorize ratings into movie and user vectors.
# J=½Σ_observed(x_i·w_j+b_j-Ynorm_ij)² + λ/2 (||X||²+||W||²).
# Missing ratings contribute NOTHING, not a zero-valued target. Movie means
# are restored after prediction; user biases are not regularized.
# Run me: ruby recommenders/collaborative_filtering.rb
require_relative "../lib/style"
require_relative "../lib/problem"
require_relative "../lib/numeric"
require_relative "../neural_networks/adam"
module CollaborativeFiltering
  class Model
    attr_reader :movies, :users, :bias, :means

    def initialize(ratings, features: 2, rng: Random.new(42))
      @means = ratings.map do |row|
        values = row.compact
        values.empty? ? 0.0 : NumericTools.mean(values)
      end
      @movies = Array.new(ratings.size) { Array.new(features) { rng.rand - 0.5 } }
      @users = Array.new(ratings[0].size) { Array.new(features) { rng.rand - 0.5 } }
      @bias = Array.new(@users.size, 0.0)
    end

    def parameters = @movies.flatten + @users.flatten + @bias

    def parameters=(values)
      offset = 0
      [@movies, @users].each do |matrix|
        matrix.each do |row|
          row.each_index do |k|
            row[k] = values[offset]
            offset += 1
          end
        end
      end
      @bias.each_index do |j|
        @bias[j] = values[offset]
        offset += 1
      end
    end

    def predict(movie, user) = NumericTools.dot(@movies[movie], @users[user]) + @bias[user] + @means[movie]

    def gradients(ratings, l2: 0.01)
      gx = @movies.map { |row| row.map { |v| l2 * v } }
      gw = @users.map { |row| row.map { |v| l2 * v } }
      gb = Array.new(@users.size, 0.0)
      cost = l2 * (@movies.flatten + @users.flatten).sum { |v| v * v } / 2.0
      ratings.each_with_index do |row, i|
        row.each_with_index do |target, j|
          next if target.nil?

          error = predict(i, j) - target
          cost += error**2 / 2.0
          @movies[i].each_index do |k|
            gx[i][k] += error * @users[j][k]
            gw[j][k] += error * @movies[i][k]
          end
          gb[j] += error
        end
      end
      [cost, gx.flatten + gw.flatten + gb]
    end

    def fit(ratings, iterations: 1000, l2: 0.01, optimizer: Adam.new)
      iterations.times do |i|
        cost, gradient = gradients(ratings, l2: l2)
        self.parameters = optimizer.step(parameters, gradient)
        yield i + 1, cost if block_given?
      end
      self
    end
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Collaborative Filtering", "observed mask · analytic factor gradients · seed 42")
  ratings = Problem.ratings
  model = CollaborativeFiltering::Model.new(ratings)
  original = model.parameters
  _cost, analytic = model.gradients(ratings)
  errors = original.each_index.map do |j|
    plus = original.dup
    plus[j] += 1e-5
    model.parameters = plus
    right = model.gradients(ratings)[0]
    minus = original.dup
    minus[j] -= 1e-5
    model.parameters = minus
    left = model.gradients(ratings)[0]
    (analytic[j] - (right - left) / 2e-5).abs
  end
  model.parameters = original
  raise "factor gradient mismatch" unless errors.max < 1e-7

  step("all #{original.size} factor/bias gradients checked; max error #{errors.max}")
  model.fit(ratings) { |i, cost| step("it #{i}: regularized loss=#{cost.round(6)}") if i <= 2 || (i % 250).zero? }
  squared = []
  ratings.each_with_index do |row, i|
    predictions = row.each_index.map { |j| model.predict(i, j) }
    step("movie #{i} mean=#{model.means[i].round(2)} observed=#{row.inspect} predicted=#{vector(predictions)}")
    row.each_with_index { |target, j| squared << (predictions[j] - target)**2 unless target.nil? }
  end
  raise "ratings not fitted" unless Math.sqrt(NumericTools.mean(squared)) < 0.1

  [0, 1, 2].each do |user|
    missing = ratings.each_index.select { |i| ratings[i][user].nil? }.sort_by { |i| -model.predict(i, user) }
    note("user #{user}: unseen movies ranked #{missing.inspect}")
  end
  ok("observed ratings train the factors; missing cells are predictions, never targets")
end
