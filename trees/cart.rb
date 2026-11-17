# frozen_string_literal: true

# scikit-learn's DecisionTreeClassifier, ported: CART with Gini impurity
# 1−Σp_k², best split per node by the proxy −(w_L·G_L + w_R·G_R), threshold at
# the midpoint of adjacent distinct values, depth-first growth that stops on
# max_depth, min_samples_split, purity or no split. Features are float32 as in
# scikit-learn. Feature order per node follows its Fisher–Yates draws from
# the xorshift stream seeded by random_state, including the bookkeeping of
# constant features, so max_features sampling and ties resolve the same way.
# Sample weights carry bootstrap multiplicities for random forests.
# Run me: ruby trees/cart.rb
require_relative "../lib/style"
require_relative "../lib/numpy_random"
module CART
  EPSILON = Float::EPSILON
  FEATURE_THRESHOLD = 1e-7
  RAND_R_MAX = 2_147_483_647
  Node = Struct.new(:feature, :threshold, :left, :right, :value, :impurity, :samples)

  module_function

  def float32(value) = [value].pack("e").unpack1("e")

  def gini(counts, total)
    return 0.0 if total <= 0

    1.0 - counts.sum { |c| (c / total)**2 }
  end

  class Splitter
    def initialize(n_features, max_features, seed)
      @n_features = n_features
      @max_features = max_features
      @state = NumpyRandom::RandomState.new(seed).interval(RAND_R_MAX - 1)
      @features = (0...n_features).to_a
      @constant_features = Array.new(n_features, 0)
    end

    def rand_int(low, high)
      state = @state
      state = 1 if state.zero?
      state ^= (state << 13) & 0xffff_ffff
      state ^= state >> 17
      state ^= (state << 5) & 0xffff_ffff
      @state = state
      low + (state % (RAND_R_MAX + 1)) % (high - low)
    end

    # Returns [feature, threshold, left_rows, right_rows, left_impurity,
    # right_impurity, constants] or nil; rows are indices into x.
    def split(x, y, weights, rows, impurity, known_constants, classes)
      found = drawn = 0
      total_constants = known_constants
      f_i = @n_features
      visited = 0
      best = nil
      best_proxy = -Float::INFINITY
      while f_i > total_constants && (visited < @max_features || visited <= found + drawn)
        visited += 1
        f_j = rand_int(drawn, f_i - found)
        if f_j < known_constants
          @features[drawn], @features[f_j] = @features[f_j], @features[drawn]
          drawn += 1
          next
        end
        f_j += found
        feature = @features[f_j]
        sorted = rows.sort_by { |r| x[r][feature] }
        values = sorted.map { |r| x[r][feature] }
        if values.last <= values.first + FEATURE_THRESHOLD
          @features[f_j], @features[total_constants] = @features[total_constants], @features[f_j]
          found += 1
          total_constants += 1
          next
        end
        f_i -= 1
        @features[f_i], @features[f_j] = @features[f_j], @features[f_i]
        left = Array.new(classes, 0.0)
        total = Array.new(classes, 0.0)
        sorted.each { |r| total[y[r]] += weights[r] }
        weight_total = total.sum
        weight_left = 0.0
        consumed = 0
        p = 0
        while p < sorted.size
          p += 1 while p + 1 < sorted.size && values[p + 1] <= values[p] + FEATURE_THRESHOLD
          p += 1
          next if p >= sorted.size

          while consumed < p
            r = sorted[consumed]
            left[y[r]] += weights[r]
            weight_left += weights[r]
            consumed += 1
          end
          weight_right = weight_total - weight_left
          right = total.each_index.map { |k| total[k] - left[k] }
          proxy = -weight_right * CART.gini(right, weight_right) - weight_left * CART.gini(left, weight_left)
          next unless proxy > best_proxy

          best_proxy = proxy
          threshold = values[p - 1] / 2.0 + values[p] / 2.0
          threshold = values[p - 1] if threshold == values[p] || threshold.infinite?
          best = [feature, threshold]
        end
      end
      @features[0, known_constants] = @constant_features[0, known_constants]
      @constant_features[known_constants, found] = @features[known_constants, found]
      return nil unless best

      feature, threshold = best
      left_rows, right_rows = rows.partition { |r| x[r][feature] <= threshold }
      [feature, threshold, left_rows, right_rows, total_constants]
    end
  end

  # DepthFirstTreeBuilder: left children before right, impurities inherited.
  def fit(x, y, max_depth: nil, min_samples_split: 2, max_features: nil, random_state: 0, sample_weight: nil, classes: 2)
    x = x.map { |row| row.map { |v| float32(v) } }
    weights = sample_weight || Array.new(x.size, 1.0)
    rows = (0...x.size).select { |i| weights[i] != 0.0 }
    splitter = Splitter.new(x.first.size, max_features || x.first.size, random_state)
    depth_limit = max_depth || 2**31 - 1
    counts = lambda do |members|
      c = Array.new(classes, 0.0)
      members.each { |r| c[y[r]] += weights[r] }
      c
    end
    build = lambda do |members, depth, impurity, constants|
      value = counts.call(members)
      impurity ||= CART.gini(value, value.sum)
      node = Node.new(nil, nil, nil, nil, value, impurity, members.size)
      leaf = depth >= depth_limit || members.size < min_samples_split || members.size < 2 || impurity <= EPSILON
      return node if leaf

      result = splitter.split(x, y, weights, members, impurity, constants, classes)
      return node unless result

      feature, threshold, left_rows, right_rows, constants = result
      return node if right_rows.empty?

      node.feature, node.threshold = feature, threshold
      node.left = build.call(left_rows, depth + 1, nil, constants)
      node.right = build.call(right_rows, depth + 1, nil, constants)
      node
    end
    build.call(rows, 0, nil, 0)
  end

  def leaf(node, row)
    node = float32(row[node.feature]) <= node.threshold ? node.left : node.right while node.feature
    node
  end

  def proba(node, row)
    value = leaf(node, row).value
    total = value.sum
    value.map { |v| v / total }
  end

  def predict(node, row)
    p = proba(node, row)
    p.index(p.max)
  end

  def depth(node) = node.feature ? 1 + [depth(node.left), depth(node.right)].max : 0
  def leaves(node) = node.feature ? leaves(node.left) + leaves(node.right) : 1
  def accuracy(tree, x, y) = x.zip(y).count { |row, t| predict(tree, row) == t }.to_f / y.size
end
if __FILE__ == $PROGRAM_NAME
  require_relative "../lib/heart_data"
  include Style
  banner("CART", "scikit-learn's DecisionTreeClassifier on heart.csv")
  names, x_train, x_test, y_train, y_test = HeartData.split
  tree = CART.fit(x_train, y_train, min_samples_split: 50, max_depth: 3, random_state: 55)
  show = lambda do |node, indent|
    if node.feature
      step("#{indent}#{names[node.feature]} ≤ #{node.threshold}  (gini #{node.impurity.round(4)}, #{node.samples} samples)")
      show.call(node.left, indent + "  ")
      show.call(node.right, indent + "  ")
    else
      step("#{indent}leaf #{node.value.map(&:to_i).inspect}")
    end
  end
  show.call(tree, "")
  step("train accuracy=#{CART.accuracy(tree, x_train, y_train)}; test accuracy=#{CART.accuracy(tree, x_test, y_test)}")
  raise "depth limit not respected" unless CART.depth(tree) <= 3

  ok("Gini splits chosen on the one-hot encoded heart features")
end
