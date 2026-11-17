# frozen_string_literal: true

# The course's library-backed examples retain their actual libraries. Ruby owns
# the example code; PyCall embeds the isolated reference runtime, not helper scripts.
ENV["PYTHON"] ||= File.expand_path("../.reference/python/bin/python", __dir__)
ENV["TF_CPP_MIN_LOG_LEVEL"] ||= "2"
ENV["TF_NUM_INTEROP_THREADS"] ||= "1"
ENV["TF_NUM_INTRAOP_THREADS"] ||= "2"
ENV["OPENBLAS_NUM_THREADS"] ||= "1"
ENV["OMP_NUM_THREADS"] ||= "2"
ENV["CUDA_VISIBLE_DEVICES"] ||= "-1"
ENV["SDL_VIDEODRIVER"] ||= "dummy"
ENV["BUNDLE_GEMFILE"] ||= File.expand_path("../Gemfile", __dir__)
ENV["BUNDLE_PATH"] ||= File.expand_path("../.reference/gems", __dir__)
require "bundler/setup"
require "pycall/import"
# Embedded CPython does not apply a venv's site initialization automatically.
reference_packages = Dir[File.expand_path("../.reference/python/lib/python*/site-packages", __dir__)].first
raise "Run make setup-reference before library examples" unless reference_packages

PyCall.import_module("sys").path.insert(0, reference_packages)
PyCall.import_module("sys").executable = ENV.fetch("PYTHON")
require_relative "course_data"
module Reference
  module_function

  def numpy = PyCall.import_module("numpy")
  def tensorflow = PyCall.import_module("tensorflow")
  def array(value) = numpy.asarray(value)
  def list(value) = value.tolist.to_a
  def shape(value) = value.shape.to_a

  def seed(value)
    numpy.random.seed(value)
    tensorflow.random.set_seed(value)
    PyCall.import_module("random").seed(value)
  end

  def tensor(value) = array(value).astype("float32")

  def columns(value, first = 0, last = nil)
    value.__getitem__(PyCall.tuple([PyCall::Slice.new(nil), PyCall::Slice.new(first, last)]))
  end

  def rows(value, first = nil, last = nil)
    value.__getitem__(PyCall::Slice.new(first, last))
  end

  def scalar(value) = value.respond_to?(:item) ? value.item : value
  def accuracy(actual, predicted) = numpy.mean(numpy.equal(actual.reshape([-1]), predicted.reshape([-1]))).item

  def heart_data
    pd = PyCall.import_module("pandas")
    frame = pd.read_csv(CourseData.path(:heart, "heart.csv"))
    categorical = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    frame = pd.get_dummies(data: frame, prefix: categorical, columns: categorical)
    x = frame.drop(columns: ["HeartDisease"]).to_numpy(dtype: "float64")
    y = frame.__getitem__("HeartDisease").to_numpy
    PyCall.import_module("sklearn.model_selection").train_test_split(x, y, train_size: 0.8, random_state: 55).to_a
  end

  def epoch_trace(history)
    history.history["loss"].to_a.each_with_index { |loss, i| Style.step("epoch #{i + 1}: loss=#{loss}") }
  end

  def digits(kind)
    x = numpy.load(CourseData.path(kind, "data/X.npy"))
    y = numpy.load(CourseData.path(kind, "data/y.npy"))
    if kind == :binary_digits
      x = x.__getitem__(PyCall::Slice.new(0, 1000))
      y = y.__getitem__(PyCall::Slice.new(0, 1000))
    end
    [x, y]
  end
end
