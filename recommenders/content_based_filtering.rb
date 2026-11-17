# frozen_string_literal: true

# C3 W2 content-based filtering: actual user/movie features and ratings.
# Two 256→128→32 towers, L2-normalized embeddings, dot-product rating prediction,
# MSE and Adam(0.01), 30 epochs. IDs/count columns are excluded as in the notebook.
require_relative "../lib/style"
require_relative "../lib/reference"
module ContentBasedFiltering
  module_function

  def build(user_features, item_features)
    tf = Reference.tensorflow
    tower = lambda do
      tf.keras.Sequential.new([
                                tf.keras.layers.Dense.new(256, activation: "relu"),
                                tf.keras.layers.Dense.new(128, activation: "relu"),
                                tf.keras.layers.Dense.new(32, activation: "linear")
                              ])
    end
    user_tower, item_tower = tower.call, tower.call
    input_user = tf.keras.layers.Input(shape: [user_features])
    input_item = tf.keras.layers.Input(shape: [item_features])
    user_embedding = tf.linalg.l2_normalize(user_tower.__call__(input_user), axis: 1)
    item_embedding = tf.linalg.l2_normalize(item_tower.__call__(input_item), axis: 1)
    output = tf.keras.layers.Dot.new(axes: 1).__call__([user_embedding, item_embedding])
    model = tf.keras.Model.new(inputs: [input_user, input_item], outputs: output)
    item_model = tf.keras.Model.new(inputs: input_item, outputs: item_embedding)
    [model, item_model]
  end

  def train(epochs: 30)
    np, tf = Reference.numpy, Reference.tensorflow
    Reference.seed(1)
    user = np.loadtxt(CourseData.path(:content, "data/content_user_train.csv"), delimiter: ",")
    item = np.loadtxt(CourseData.path(:content, "data/content_item_train.csv"), delimiter: ",")
    target = np.loadtxt(CourseData.path(:content, "data/content_y_train.csv"), delimiter: ",")
    preprocessing = PyCall.import_module("sklearn.preprocessing")
    user_scaler, item_scaler = preprocessing.StandardScaler.new, preprocessing.StandardScaler.new
    # Preserve the notebook's fit-before-split ordering; this source leakage is documented.
    scaled_user = user_scaler.fit_transform(user)
    scaled_item = item_scaler.fit_transform(item)
    split = PyCall.import_module("sklearn.model_selection").train_test_split(scaled_user, scaled_item, target, train_size: 0.8, shuffle: true, random_state: 1).to_a
    u_train, u_test, i_train, i_test, y_train, y_test = split
    target_scaler = preprocessing.MinMaxScaler.new(PyCall.tuple([-1, 1]))
    normalized_train = target_scaler.fit_transform(y_train.reshape([-1, 1]))
    normalized_test = target_scaler.transform(y_test.reshape([-1, 1]))
    model, item_model = build(Reference.shape(user)[1] - 3, Reference.shape(item)[1] - 1)
    tf.random.set_seed(1)
    model.compile(optimizer: tf.keras.optimizers.Adam.new(learning_rate: 0.01), loss: tf.keras.losses.MeanSquaredError.new)
    tf.random.set_seed(1)
    history = model.fit([Reference.columns(u_train, 3), Reference.columns(i_train, 1)], normalized_train, epochs: epochs, batch_size: 32, shuffle: true, verbose: 0)
    held_out = model.evaluate([Reference.columns(u_test, 3), Reference.columns(i_test, 1)], normalized_test, verbose: 0)
    [model, item_model, history, held_out, user, item, user_scaler, item_scaler, target_scaler]
  end

  def square_distance(a, b) = a.zip(b).sum { |x, y| (x - y)**2 }
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Neural Content-Based Movie Recommendations", "58187 original training pairs · TensorFlow/Keras")
  model, item_model, history, test_loss, raw_users, _raw_items, user_scaler, item_scaler, target_scaler = ContentBasedFiltering.train
  Reference.epoch_trace(history)
  step("held-out MSE=#{test_loss}; user input=14, item input=16; towers 256→128→32")
  np = Reference.numpy
  items = np.loadtxt(CourseData.path(:content, "data/content_item_vecs.csv"), delimiter: ",")
  scaled_items = Reference.columns(item_scaler.transform(items), 1)
  metadata = PyCall.import_module("pandas").read_csv(CourseData.path(:content, "data/content_movie_list.csv")).set_index("movieId")
  new_user = np.asarray([[5000, 3, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 5, 5, 1]])
  user_inputs = Reference.columns(user_scaler.transform(np.tile(new_user, [Reference.shape(items)[0], 1])), 3)
  predictions = target_scaler.inverse_transform(model.predict([user_inputs, scaled_items], verbose: 0)).reshape([-1])
  order = Reference.list(np.argsort(np.negative(predictions))).first(10)
  order.each do |index|
    id = items.__getitem__(PyCall.tuple([index, 0])).item.to_i
    title = metadata.loc.__getitem__(id).__getitem__("title").to_s
    step("new user: #{title}, predicted rating #{predictions.__getitem__(index).item.round(3)}")
  end
  # Existing user 36 is taken from actual user rows, not invented preferences.
  ids = np.where(np.equal(Reference.columns(raw_users, 0, 1).reshape([-1]), 36)).to_a.first
  raise "Original example user 36 missing" if ids.size == 0

  existing = raw_users.__getitem__(ids.__getitem__(0)).reshape([1, -1])
  existing_inputs = Reference.columns(user_scaler.transform(np.tile(existing, [Reference.shape(items)[0], 1])), 3)
  existing_scores = target_scaler.inverse_transform(model.predict([existing_inputs, scaled_items], verbose: 0)).reshape([-1])
  step("existing user 36: first predicted ratings #{Reference.list(existing_scores).first(5)}")
  vectors = item_model.predict(scaled_items, verbose: 0)
  5.times do |i|
    distances = np.sum(np.square(np.subtract(vectors, vectors.__getitem__(i))), axis: 1)
    distances.__setitem__(i, np.inf)
    other = np.argmin(distances).item
    a = items.__getitem__(PyCall.tuple([i, 0])).item.to_i
    b = items.__getitem__(PyCall.tuple([other, 0])).item.to_i
    step("similar movies: #{metadata.loc.__getitem__(a).__getitem__('title')} → #{metadata.loc.__getitem__(b).__getitem__('title')}")
    raise "Self match retained" if i == other
  end
  raise "Content recommender returned nonfinite results" unless test_loss.finite? && np.isfinite(predictions).all.item

  ok("real movie metadata supports new-user, existing-user and similar-item recommendations")
end
