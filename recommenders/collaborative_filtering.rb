# frozen_string_literal: true

# C3 W2 collaborative filtering: original movie/user matrices, 100 latent features,
# observed-rating mask, movie-mean normalization and user biases. The scalar cost
# is explicit Ruby; the course vectorized training uses TensorFlow GradientTape.
require_relative "../lib/style"
require_relative "../lib/numeric"
require_relative "../lib/reference"
module CollaborativeFiltering
  module_function

  def cost(movie_factors, user_factors, bias, ratings, mask, l2: 0.0)
    total = 0.0
    ratings.each_index do |i|
      ratings[i].each_index do |j|
        next unless mask[i][j] == 1

        error = NumericTools.dot(movie_factors[i], user_factors[j]) + bias[j] - ratings[i][j]
        total += error * error
      end
    end
    (total + l2 * (movie_factors.flatten + user_factors.flatten).sum { |v| v * v }) / 2.0
  end

  def tensor_cost(x, w, b, y, r, lambda_)
    tf = Reference.tensorflow
    residual = tf.multiply(tf.subtract(tf.add(tf.linalg.matmul(x, tf.transpose(w)), b), y), r)
    penalty = tf.multiply(lambda_ / 2.0, tf.add(tf.reduce_sum(tf.square(x)), tf.reduce_sum(tf.square(w))))
    tf.add(tf.multiply(0.5, tf.reduce_sum(tf.square(residual))), penalty)
  end

  def train(iterations: 200, features: 100, rate: 0.1, lambda_: 1.0)
    np, tf = Reference.numpy, Reference.tensorflow
    Reference.seed(1234)
    y = np.loadtxt(CourseData.path(:collaborative, "data/small_movies_Y.csv"), delimiter: ",")
    r = np.loadtxt(CourseData.path(:collaborative, "data/small_movies_R.csv"), delimiter: ",")
    personal = np.zeros([Reference.shape(y)[0]])
    { 2700 => 5, 2609 => 2, 929 => 5, 246 => 5, 2716 => 3, 1150 => 5, 382 => 2, 366 => 5, 622 => 5, 988 => 3, 2925 => 1, 2937 => 1, 793 => 5 }.each { |i, rating| personal.__setitem__(i, rating) }
    y = np.column_stack([personal, y])
    r = np.column_stack([np.not_equal(personal, 0).astype("float64"), r])
    counts = np.sum(r, axis: 1, keepdims: true)
    means = np.divide(np.sum(np.multiply(y, r), axis: 1, keepdims: true), np.maximum(counts, 1))
    normalized = np.multiply(np.subtract(y, means), r)
    movies, users = Reference.shape(y)
    x = tf.Variable.new(tf.random.normal([movies, features], dtype: tf.float64), name: "X")
    w = tf.Variable.new(tf.random.normal([users, features], dtype: tf.float64), name: "W")
    b = tf.Variable.new(tf.random.normal([1, users], dtype: tf.float64), name: "b")
    optimizer = tf.keras.optimizers.Adam.new(learning_rate: rate)
    iterations.times do |iteration|
      tape = tf.GradientTape.new
      tape.__enter__
      begin
        objective = tensor_cost(x, w, b, normalized, r, lambda_)
      ensure
        tape.__exit__(nil, nil, nil)
      end
      gradients = tape.gradient(objective, [x, w, b]).to_a
      optimizer.apply_gradients(gradients.zip([x, w, b]).map { |gradient, variable| PyCall.tuple([gradient, variable]) })
      yield iteration, objective.numpy.item if block_given?
    end
    prediction = np.add(np.add(np.matmul(x.numpy, np.transpose(w.numpy)), b.numpy), means)
    [prediction, personal, Reference.shape(y)]
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Collaborative Movie Recommendations", "original 4778 movies · 443 users + personal ratings · TensorFlow")
  np = Reference.numpy
  load_matrix = ->(name) { np.loadtxt(CourseData.path(:collaborative, "data/small_movies_#{name}.csv"), delimiter: ",") }
  x = Reference.columns(Reference.rows(load_matrix.call("X"), 0, 5), 0, 3)
  w = Reference.columns(Reference.rows(load_matrix.call("W"), 0, 4), 0, 3)
  b = Reference.columns(load_matrix.call("b").reshape([1, -1]), 0, 4)
  y = Reference.columns(Reference.rows(load_matrix.call("Y"), 0, 5), 0, 4)
  r = Reference.columns(Reference.rows(load_matrix.call("R"), 0, 5), 0, 4)
  arrays = [x, w, y, r].map { |a| Reference.list(a).map(&:to_a) }
  [0.0, 1.5].each do |lambda_|
    explicit = CollaborativeFiltering.cost(arrays[0], arrays[1], Reference.list(b).first.to_a, arrays[2], arrays[3], l2: lambda_)
    vectorized = CollaborativeFiltering.tensor_cost(x, w, b, y, r, lambda_).numpy.item
    step("original 5-movie/4-user cost fixture: lambda=#{lambda_}; Ruby=#{explicit}; TensorFlow=#{vectorized}")
    raise "Scalar collaborative cost differs" unless (explicit - vectorized).abs < 1e-8
  end
  prediction, personal, shape = CollaborativeFiltering.train { |iteration, cost| step("iteration #{iteration}: cost=#{cost}") if (iteration % 20).zero? }
  scores = Reference.columns(prediction, 0, 1).reshape([-1])
  order = Reference.list(Reference.numpy.argsort(Reference.numpy.negative(scores)))
  movies = PyCall.import_module("pandas").read_csv(CourseData.path(:collaborative, "data/small_movie_list.csv"), index_col: 0)
  shown = 0
  order.each do |index|
    next if personal.__getitem__(index).item > 0

    title = movies.iloc.__getitem__(index).__getitem__("title").to_s
    step("#{title}: predicted rating #{scores.__getitem__(index).item.round(3)}")
    shown += 1
    break if shown == 10
  end
  raise "Movie prediction dimensions changed" unless shape == [4778, 444] && Reference.numpy.isfinite(prediction).all.item

  ok("the original ratings matrix and personal-rating recommendation workflow are retained")
end
