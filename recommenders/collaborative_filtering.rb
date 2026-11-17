# frozen_string_literal: true

# Course C3 W2 assignment: collaborative filtering on the 4778×443 ratings
# matrix plus a column of the notebook's own ratings. Ratings are normalized
# by each movie's observed mean (Σ Y·R / (Σ R + 1e-12)); movie features X,
# user weights W and user biases b (100 features, standard-normal start)
# minimize J = ½Σ_R (x_i·w_j + b_j − y_ij)² + λ/2(ΣX² + ΣW²) with λ=1 over
# 200 Adam(0.1) steps. Gradients are written out, summing only rated cells:
# ∂J/∂x_i = Σ_j e_ij w_j + λx_i, ∂J/∂w_j = Σ_i e_ij x_i + λw_j, ∂J/∂b_j = Σ_i e_ij.
# Run me: ruby recommenders/collaborative_filtering.rb
require_relative "../lib/style"
require_relative "../lib/numeric"
require_relative "../lib/course_data"
require_relative "../lib/numpy_random"
require_relative "../neural_networks/adam"
module CollaborativeFiltering
  MY_RATINGS = { 2700 => 5, 2609 => 2, 929 => 5, 246 => 5, 2716 => 3, 1150 => 5, 382 => 2, 366 => 5, 622 => 5, 988 => 3, 2925 => 1, 2937 => 1, 793 => 5 }.freeze

  module_function

  # The assignment's loop form of the cost, X/W as row arrays and b a vector.
  def cost(x, w, b, y, r, lambda_)
    j = 0.0
    y.each_index do |i|
      w.each_index do |u|
        next unless r[i][u] == 1

        j += (NumericTools.dot(w[u], x[i]) + b[u] - y[i][u])**2
      end
    end
    j / 2 + lambda_ / 2.0 * (w.flatten.sum { |v| v * v } + x.flatten.sum { |v| v * v })
  end

  def normalize(y, r)
    means = y.each_index.map { |i| y[i].each_index.sum { |u| y[i][u] * r[i][u] } / (r[i].sum + 1e-12) }
    [y.each_index.map { |i| y[i].each_index.map { |u| y[i][u] - means[i] * r[i][u] } }, means]
  end

  # Flat row-major parameters; ratings as [movie, user, normalized rating].
  def train(ynorm, observed, movies, users, features: 100, lambda_: 1.0, iterations: 200, rate: 0.1, seed: 1234)
    rng = NumpyRandom::RandomState.new(seed)
    w = Array.new(users * features) { rng.standard_normal }
    x = Array.new(movies * features) { rng.standard_normal }
    b = Array.new(users) { rng.standard_normal }
    optimizer = Adam.new(learning_rate: rate)
    iterations.times do |iteration|
      gx = x.map { |v| lambda_ * v }
      gw = w.map { |v| lambda_ * v }
      gb = Array.new(users, 0.0)
      j = lambda_ / 2.0 * (x.sum { |v| v * v } + w.sum { |v| v * v })
      observed.each do |i, u|
        xo, wo = i * features, u * features
        prediction = b[u]
        k = 0
        while k < features
          prediction += x[xo + k] * w[wo + k]
          k += 1
        end
        error = prediction - ynorm[i][u]
        j += error * error / 2
        gb[u] += error
        k = 0
        while k < features
          gx[xo + k] += error * w[wo + k]
          gw[wo + k] += error * x[xo + k]
          k += 1
        end
      end
      optimizer.apply([w, x, b], [gw, gx, gb])
      yield iteration, j if block_given?
    end
    [x, w, b]
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Collaborative Filtering", "4778 movies × 443 users + your ratings · 100 features · Adam")
  y = CourseData.table(:collaborative, "data/small_movies_Y.csv")
  r = CourseData.table(:collaborative, "data/small_movies_R.csv").map { |row| row.map(&:to_i) }
  x_pre = CourseData.table(:collaborative, "data/small_movies_X.csv")
  w_pre = CourseData.table(:collaborative, "data/small_movies_W.csv")
  b_pre = CourseData.table(:collaborative, "data/small_movies_b.csv").first
  _header, movies = CourseData.csv(:collaborative, "data/small_movie_list.csv")
  step("Y #{y.size}×#{y.first.size}; #{r.flatten.sum} ratings; average rating of movie 0: #{(y[0].each_index.sum { |u| y[0][u] * r[0][u] } / r[0].sum.to_f).round(3)}")
  x_r, w_r, b_r = x_pre.first(5).map { |row| row.first(3) }, w_pre.first(4).map { |row| row.first(3) }, b_pre.first(4)
  y_r, r_r = y.first(5).map { |row| row.first(4) }, r.first(5).map { |row| row.first(4) }
  [0.0, 1.5].each { |lambda_| step("reduced 5×4 problem, 3 features, λ=#{lambda_}: cost=#{CollaborativeFiltering.cost(x_r, w_r, b_r, y_r, r_r, lambda_).round(2)}") }
  mine = Array.new(y.size, 0.0)
  CollaborativeFiltering::MY_RATINGS.each { |i, rating| mine[i] = rating.to_f }
  CollaborativeFiltering::MY_RATINGS.each { |i, rating| note("you rated #{rating}: #{movies[i][3]}") }
  y = y.each_index.map { |i| [mine[i]] + y[i] }
  r = r.each_index.map { |i| [mine[i].zero? ? 0 : 1] + r[i] }
  ynorm, means = CollaborativeFiltering.normalize(y, r)
  observed = []
  r.each_index { |i| r[i].each_index { |u| observed << [i, u] if r[i][u] == 1 } }
  costs = []
  x, w, b = CollaborativeFiltering.train(ynorm, observed, y.size, y.first.size) do |iteration, cost|
    costs << cost
    step("iteration #{iteration}: cost=#{cost.round(1)}") if (iteration % 20).zero?
  end
  predictions = Array.new(y.size) { |i| NumericTools.dot(x[i * 100, 100], w[0, 100]) + b[0] + means[i] }
  order = predictions.each_index.sort_by { |i| -predictions[i] }
  order.first(17).each { |i| step("predicting rating #{predictions[i].round(2)} for #{movies[i][3]}") unless mine[i].positive? }
  CollaborativeFiltering::MY_RATINGS.each { |i, rating| step("original #{rating}, predicted #{predictions[i].round(2)} for #{movies[i][3]}") }
  popular = order.first(300).select { |i| movies[i][2].to_i > 20 }.sort_by { |i| -movies[i][1].to_f }.first(5)
  popular.each { |i| note("top-300 prediction with >20 ratings: #{movies[i][3]} (mean #{movies[i][1].to_f.round(2)}, predicted #{predictions[i].round(2)})") }
  raise "Adam did not reduce the collaborative-filtering cost" unless costs.last < costs.first

  ok("unrated cells predicted from learned movie and user factors")
end
