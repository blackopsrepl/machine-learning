# frozen_string_literal: true

# Course C3 W2 assignment: content-based filtering with two neural towers.
# Users drop id/count/average (14 features), items drop the id (16); both are
# StandardScaler-scaled on all rows before the 80/20 split (the notebook's
# order), targets MinMax-scaled to [−1,1] on the training split. Each tower is
# 256→128→32 (ReLU, ReLU, linear); embeddings are L2-normalized, the rating is
# their dot product, MSE + Adam(0.01) for 30 epochs in minibatches of 32.
# Normalization's derivative: ∂û/∂u = (I − ûûᵀ)/‖u‖.
# Run me: ruby recommenders/content_based_filtering.rb
require_relative "../neural_networks/sequential"
require_relative "../lib/numeric"
require_relative "../lib/course_data"
require_relative "../lib/sklearn_data"
module ContentBasedFiltering
  NORM_EPSILON = 1e-12

  module_function

  def scaler(rows) = NumericTools.fit_scaler(rows)
  def scale(rows, s) = NumericTools.transform(rows, s)

  def normalize(flat, width)
    flat.each_slice(width).flat_map do |row|
      norm = Math.sqrt([row.sum { |v| v * v }, NORM_EPSILON].max)
      row.map { |v| v / norm }
    end
  end

  class TwoTower
    attr_reader :user, :item

    def initialize(user_features, item_features, rng: Random.new(1))
      tower = [[256, :relu], [128, :relu], [32, :linear]]
      @user = Sequential.new(user_features, tower, rng: rng)
      @item = Sequential.new(item_features, tower, rng: rng)
      @optimizer = Adam.new(learning_rate: 0.01)
    end

    def tensors = @user.tensors + @item.tensors

    def embed(tower, rows) = ContentBasedFiltering.normalize(tower.forward(rows)[1].last, 32).each_slice(32).to_a
    def predict(users, items) = embed(@user, users).zip(embed(@item, items)).map { |u, v| NumericTools.dot(u, v) }

    def train_on_batch(users, items, targets)
      ua, uz = @user.forward(users)
      ia, iz = @item.forward(items)
      batch = users.size
      du = Array.new(batch * 32, 0.0)
      dv = Array.new(batch * 32, 0.0)
      loss = 0.0
      batch.times do |b|
        u = uz.last[b * 32, 32]
        v = iz.last[b * 32, 32]
        nu = Math.sqrt([u.sum { |x| x * x }, NORM_EPSILON].max)
        nv = Math.sqrt([v.sum { |x| x * x }, NORM_EPSILON].max)
        uh, vh = u.map { |x| x / nu }, v.map { |x| x / nv }
        score = NumericTools.dot(uh, vh)
        error = score - targets[b]
        loss += error * error
        ds = 2.0 * error / batch
        32.times do |k|
          du[b * 32 + k] = ds * (vh[k] - score * uh[k]) / nu
          dv[b * 32 + k] = ds * (uh[k] - score * vh[k]) / nv
        end
      end
      gu, = @user.backward(ua, uz, du)
      gi, = @item.backward(ia, iz, dv)
      @optimizer.apply(tensors, gu + gi)
      loss / batch
    end

    def fit(users, items, targets, epochs: 30, rng: Random.new(1))
      order = (0...users.size).to_a
      Array.new(epochs) do |epoch|
        order = order.shuffle(random: rng)
        total = 0.0
        order.each_slice(32) { |ids| total += train_on_batch(ids.map { |i| users[i] }, ids.map { |i| items[i] }, ids.map { |i| targets[i] }) * ids.size }
        (total / users.size).tap { |loss| yield epoch + 1, loss if block_given? }
      end
    end
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Content-Based Filtering", "user and movie towers · 256→128→32 · normalized dot product")
  users = CourseData.table(:content, "data/content_user_train.csv")
  items = CourseData.table(:content, "data/content_item_train.csv")
  ratings = CourseData.table(:content, "data/content_y_train.csv").map(&:first)
  vectors = CourseData.table(:content, "data/content_item_vecs.csv")
  _header, catalogue = CourseData.csv(:content, "data/content_movie_list.csv")
  movie = catalogue.to_h { |id, title, genres| [id.to_i, [title, genres]] }
  step("#{ratings.size} training rows; user features #{users.first.size - 3}, item features #{items.first.size - 1}; #{vectors.size} item vectors")
  user_scaler, item_scaler = ContentBasedFiltering.scaler(users), ContentBasedFiltering.scaler(items)
  scaled_users, scaled_items = ContentBasedFiltering.scale(users, user_scaler), ContentBasedFiltering.scale(items, item_scaler)
  item_train, item_test = SklearnData.train_test_split(scaled_items, train_size: 0.8, random_state: 1)
  user_train, user_test = SklearnData.train_test_split(scaled_users, train_size: 0.8, random_state: 1)
  y_train, y_test = SklearnData.train_test_split(ratings, train_size: 0.8, random_state: 1)
  low, high = y_train.minmax
  to_unit = ->(v) { 2.0 * (v - low) / (high - low) - 1.0 }
  to_rating = ->(v) { (v + 1.0) * (high - low) / 2.0 + low }
  model = ContentBasedFiltering::TwoTower.new(14, 16)
  model.fit(user_train.map { |r| r.drop(3) }, item_train.map { |r| r.drop(1) }, y_train.map(&to_unit)) { |epoch, loss| step("epoch #{epoch}: loss=#{loss}") }
  test_loss = model.predict(user_test.map { |r| r.drop(3) }, item_test.map { |r| r.drop(1) }).zip(y_test.map(&to_unit)).sum { |p, t| (p - t)**2 } / y_test.size
  step("test MSE (scaled ratings)=#{test_loss}")
  scaled_vectors = ContentBasedFiltering.scale(vectors, item_scaler).map { |r| r.drop(1) }
  new_user = [5000, 3, 1.0, 1.0, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 5, 5, 1]
  predicted = model.predict(Array.new(vectors.size) { ContentBasedFiltering.scale([new_user], user_scaler).first.drop(3) }, scaled_vectors).map(&to_rating)
  predicted.each_index.sort_by { |i| -predicted[i] }.first(10).each do |i|
    id = vectors[i][0].to_i
    step("new user (comedy, romance, sci-fi): #{predicted[i].round(1)} for #{movie[id][0]} [#{movie[id][1]}]")
  end
  rated = users.each_index.select { |i| users[i][0].to_i == 36 }.to_h { |i| [items[i][0].to_i, ratings[i]] }
  existing = users.find { |row| row[0].to_i == 36 }
  raise "user 36 is missing" unless existing

  scores = model.predict(Array.new(vectors.size) { ContentBasedFiltering.scale([existing], user_scaler).first.drop(3) }, scaled_vectors).map(&to_rating)
  scores.each_index.sort_by { |i| -scores[i] }.select { |i| rated.key?(vectors[i][0].to_i) }.first(10).each do |i|
    id = vectors[i][0].to_i
    step("user 36: predicted #{scores[i].round(1)}, rated #{rated[id]} for #{movie[id][0]}")
  end
  embeddings = model.embed(model.item, scaled_vectors)
  [vectors.size, 50].min.times do |i|
    j = (0...vectors.size).reject { |k| k == i }.min_by { |k| embeddings[i].zip(embeddings[k]).sum { |a, c| (a - c)**2 } }
    note("#{movie[vectors[i][0].to_i][0]} [#{movie[vectors[i][0].to_i][1]}] ↔ #{movie[vectors[j][0].to_i][0]} [#{movie[vectors[j][0].to_i][1]}]") if i < 10
  end
  raise "content-based predictions are not finite" unless (predicted + scores).all?(&:finite?)

  ok("ratings predicted from user and movie features; similar movies from embeddings")
end
