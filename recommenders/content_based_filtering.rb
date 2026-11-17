# frozen_string_literal: true

# Content-based filtering — two learned towers map user/item features to vectors.
# Normalize embeddings, predict their dot product, minimize half squared error.
# d(u_hat·v_hat)/du=(v_hat-score*u_hat)/||u||: differentiate normalization
# explicitly, then feed those deltas through each tower's handwritten backprop.
# Run me: ruby recommenders/content_based_filtering.rb
require_relative "../neural_networks/backpropagation"
module ContentBasedFiltering
  class Model
    attr_reader :user_tower, :item_tower

    def initialize(rng: Random.new(42))
      @user_tower = Backpropagation::Network.new([2, 4, 2], activations: [:relu, :linear], rng: rng)
      @item_tower = Backpropagation::Network.new([2, 4, 2], activations: [:relu, :linear], rng: rng)
      # Nonzero output biases keep an all-inactive ReLU path off the zero-norm kink.
      @user_tower.biases.last.replace([0.2, -0.1])
      @item_tower.biases.last.replace([0.2, -0.1])
    end

    def parameters = @user_tower.parameters + @item_tower.parameters

    def parameters=(p)
      n = @user_tower.parameters.size
      @user_tower.parameters = p.first(n)
      @item_tower.parameters = p.drop(n)
    end

    def normalized(raw)
      length = Math.sqrt(raw.sum { |v| v * v })
      denominator = [length, 1e-8].max
      [raw.map { |v| v / denominator }, denominator, length > 1e-8]
    end

    def embeddings(user, item)
      ua, uz = @user_tower.forward(user)
      ia, iz = @item_tower.forward(item)
      u, un, free_u = normalized(ua.last)
      v, vn, free_v = normalized(ia.last)
      [u, v, un, vn, free_u, free_v, ua, uz, ia, iz]
    end

    def predict(user, item)
      u, v, *_ = embeddings(user, item)
      NumericTools.dot(u, v)
    end

    def gradients(examples)
      total = Array.new(parameters.size, 0.0)
      cost = 0.0
      examples.each do |user, item, target|
        u, v, un, vn, free_u, free_v, ua, uz, ia, iz = embeddings(user, item)
        score = NumericTools.dot(u, v)
        error = score - target
        cost += error**2 / 2.0
        du = u.each_index.map { |k| error * (v[k] - (free_u ? score * u[k] : 0.0)) / un }
        dv = v.each_index.map { |k| error * (u[k] - (free_v ? score * v[k] : 0.0)) / vn }
        g = @user_tower.backward(ua, uz, du) + @item_tower.backward(ia, iz, dv)
        total = total.zip(g).map { |a, b| a + b }
      end
      [cost / examples.size, total.map { |g| g / examples.size }]
    end

    def fit(examples, iterations: 700, optimizer: Adam.new(rate: 0.01))
      iterations.times do |i|
        cost, g = gradients(examples)
        self.parameters = optimizer.step(parameters, g)
        yield i + 1, cost if block_given?
      end
      self
    end

    def item_distance(a, b)
      av = normalized(@item_tower.predict(a))[0]
      bv = normalized(@item_tower.predict(b))[0]
      av.zip(bv).sum { |x, y| (x - y)**2 }
    end

    def similar_items(items, index)
      items.each_index.reject { |j| j == index }.sort_by { |j| item_distance(items[index], items[j]) }
    end
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Content-Based Recommender", "two towers · normalized embeddings · seed 42")
  # Genre preferences and item genre mixtures have a known cosine target.
  users, items = Problem.users, Problem.genres
  examples = users.product(items).map do |user, item|
    norm = Math.sqrt(NumericTools.dot(user, user) * NumericTools.dot(item, item))
    [user, item, NumericTools.dot(user, item) / norm]
  end
  model = ContentBasedFiltering::Model.new
  original = model.parameters
  _loss, gradient = model.gradients(examples)
  error = original.each_index.map do |j|
    plus = original.dup
    plus[j] += 1e-5
    model.parameters = plus
    right = model.gradients(examples)[0]
    minus = original.dup
    minus[j] -= 1e-5
    model.parameters = minus
    left = model.gradients(examples)[0]
    (gradient[j] - (right - left) / 2e-5).abs
  end.max
  model.parameters = original
  raise "normalization/tower gradient mismatch" unless error < 1e-6

  step("all #{original.size} tower parameters checked; max gradient error #{error}")
  model.fit(examples) { |i, cost| step("it #{i}: rating loss=#{cost.round(6)}") if i <= 2 || (i % 175).zero? }
  final = model.gradients(examples)[0]
  raise "two towers failed" unless final < 0.005

  users.each do |user|
    ranking = items.each_index.sort_by { |i| -model.predict(user, items[i]) }
    step("user #{vector(user)} ranks movies #{ranking.inspect}; scores #{vector(ranking.map { |i| model.predict(user, items[i]) })}")
  end
  near = model.item_distance(items[0], items[1])
  far = model.item_distance(items[0], items[2])
  step("movie 0 distance: similar genre #{near.round(4)}, opposite genre #{far.round(4)}")
  raise "embedding distance uninformative" unless near < far

  items.each_index do |i|
    similar = model.similar_items(items, i)
    step("movie #{i}: other movies ranked by embedding distance #{similar.inspect}")
    raise "self match retained" if similar.include?(i)
  end
  raise "nearest genre partner missed" unless model.similar_items(items, 0).first == 1

  ok("learned normalized vectors support both user ranking and item similarity")
end
