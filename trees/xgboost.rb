# frozen_string_literal: true

# Course tree-ensemble lab: XGBClassifier(n_estimators=500, learning_rate=0.1,
# random_state=55) with XGBoost 1.7 defaults for small data — exact greedy
# trees, binary:logistic, base_score 0.5, max_depth 6, λ=1, γ=0,
# min_child_weight=1 — trained on the first 80% of the training rows with
# early stopping (50 rounds) on log-loss over the remaining 20%.
# Each round: g=p−y, h=max(p(1−p),1e-16); a split must keep Σh≥1 on both
# sides and gain G_L²/(H_L+λ)+G_R²/(H_R+λ)−G²/(H+λ) > 1e-6; rows with
# x < (a+b)/2 go left; leaves add −η·G/(H+λ) to the log-odds.
# Run me: ruby trees/xgboost.rb
require_relative "../lib/style"
require_relative "../lib/heart_data"
module XGBoost
  RT_EPS = 1e-6
  Node = Struct.new(:feature, :threshold, :left, :right, :weight, :gain, :cover)

  module_function

  def float32(value) = [value].pack("e").unpack1("e")
  def sigmoid(z) = z >= 0 ? 1.0 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z))
  def gain(g, h, lambda) = g * g / (h + lambda)

  def leaf_weight(g, h, lambda, min_child_weight)
    h < min_child_weight || h <= 0 ? 0.0 : -g / (h + lambda)
  end

  # Exact greedy search over every feature and every boundary between
  # distinct sorted values; ties keep the lower feature index.
  def grow(x, grad, hess, rows, depth, params)
    g = rows.sum { |r| grad[r] }
    h = rows.sum { |r| hess[r] }
    node = Node.new(nil, nil, nil, nil, leaf_weight(g, h, params[:lambda], params[:min_child_weight]) * params[:eta], 0.0, h)
    return node if depth >= params[:max_depth]

    root = gain(g, h, params[:lambda])
    best = nil
    x.first.size.times do |feature|
      sorted = rows.sort_by { |r| x[r][feature] }
      gl = hl = 0.0
      sorted.each_with_index do |r, i|
        gl += grad[r]
        hl += hess[r]
        nxt = sorted[i + 1]
        next unless nxt && x[nxt][feature] != x[r][feature]
        next if hl < params[:min_child_weight] || h - hl < params[:min_child_weight]

        change = gain(gl, hl, params[:lambda]) + gain(g - gl, h - hl, params[:lambda]) - root
        best = [change, feature, float32((x[r][feature] + x[nxt][feature]) * 0.5)] if best.nil? || change > best[0]
      end
    end
    return node unless best && best[0] > RT_EPS && best[0] >= params[:gamma]

    node.gain, node.feature, node.threshold = best
    left, right = rows.partition { |r| x[r][node.feature] < node.threshold }
    node.left = grow(x, grad, hess, left, depth + 1, params)
    node.right = grow(x, grad, hess, right, depth + 1, params)
    node
  end

  def score(tree, row)
    tree = row[tree.feature] < tree.threshold ? tree.left : tree.right while tree.feature
    tree.weight
  end

  def logloss(y, margins)
    y.zip(margins).sum do |t, m|
      p = sigmoid(m).clamp(1e-16, 1 - 1e-16)
      -(t * Math.log(p) + (1 - t) * Math.log(1 - p))
    end / y.size
  end

  def fit(x, y, eval_x:, eval_y:, n_estimators: 500, eta: 0.1, max_depth: 6, lambda: 1.0, gamma: 0.0, min_child_weight: 1.0, early_stopping_rounds: 50, base_score: 0.5)
    params = { eta: eta, max_depth: max_depth, lambda: lambda, gamma: gamma, min_child_weight: min_child_weight }
    x = x.map { |row| row.map { |v| float32(v) } }
    eval_x = eval_x.map { |row| row.map { |v| float32(v) } }
    base = Math.log(base_score / (1 - base_score))
    margins = Array.new(x.size, base)
    eval_margins = Array.new(eval_x.size, base)
    trees, history = [], []
    best_round, best_loss = 0, Float::INFINITY
    n_estimators.times do |round|
      grad = margins.each_index.map { |i| sigmoid(margins[i]) - y[i] }
      hess = margins.map { |m| [sigmoid(m) * (1 - sigmoid(m)), 1e-16].max }
      tree = grow(x, grad, hess, (0...x.size).to_a, 0, params)
      trees << tree
      x.each_index { |i| margins[i] += score(tree, x[i]) }
      eval_x.each_index { |i| eval_margins[i] += score(tree, eval_x[i]) }
      loss = logloss(eval_y, eval_margins)
      history << loss
      yield round, loss if block_given?
      if loss < best_loss
        best_round, best_loss = round, loss
      elsif round - best_round >= early_stopping_rounds
        break
      end
    end
    { trees: trees, base: base, best_iteration: best_round, history: history }
  end

  def margin(model, row, rounds: model[:best_iteration] + 1)
    row = row.map { |v| float32(v) }
    model[:trees].first(rounds).sum(model[:base]) { |tree| score(tree, row) }
  end

  def predict(model, row) = sigmoid(margin(model, row)) >= 0.5 ? 1 : 0
  def accuracy(model, x, y) = x.zip(y).count { |row, t| predict(model, row) == t }.to_f / y.size
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("XGBoost Heart Disease Classifier", "exact greedy gradient boosting · early stopping on held-out rows")
  _names, x_train, x_test, y_train, y_test = HeartData.split
  n = (x_train.size * 0.8).to_i
  model = XGBoost.fit(x_train.first(n), y_train.first(n), eval_x: x_train.drop(n), eval_y: y_train.drop(n)) do |round, loss|
    step("round #{round}: validation logloss=#{loss}") if round < 3 || (round % 10).zero?
  end
  rounds = model[:trees].size
  step("stopped after #{rounds} rounds; best iteration=#{model[:best_iteration]} (logloss #{model[:history][model[:best_iteration]]})")
  first = model[:trees].first
  step("first tree: root split on feature #{first.feature} < #{first.threshold}, gain #{first.gain.round(4)}, cover #{first.cover}")
  train, test = XGBoost.accuracy(model, x_train, y_train), XGBoost.accuracy(model, x_test, y_test)
  step("accuracy using rounds 0..#{model[:best_iteration]}: train=#{train.round(4)} test=#{test.round(4)}")
  raise "early stopping did not keep the best round" unless model[:history].min == model[:history][model[:best_iteration]]

  ok("second-order boosted trees with shrinkage, stopped on validation loss")
end
