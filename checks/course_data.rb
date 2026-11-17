# frozen_string_literal: true

require_relative "../lib/course_data"
require_relative "../lib/png_image"

# Verify original files are reachable before invoking expensive course training.
expected = {
  linear: ["data/ex1data1.txt", 97, 2],
  logistic: ["data/ex2data1.txt", 100, 3],
  houses: ["data/houses.txt", 100, 5]
}
expected.each do |name, (file, rows, columns)|
  data = CourseData.table(name, file)
  raise "#{name}: expected #{rows}×#{columns}" unless data.size == rows && data.all? { |r| r.size == columns }

  puts "#{name}: #{rows}×#{columns}"
end
[[:binary_digits, 1000, 400], [:multiclass_digits, 5000, 400]].each do |name, rows, columns|
  x, y = CourseData.digits(name)
  raise "#{name}: digit dimensions differ" unless x.size == rows && x.first.size == columns && y.size == rows

  puts "#{name}: #{rows}×#{columns}; labels #{y.uniq.sort.inspect}"
end
clusters = CourseData.npy(:kmeans, "data/ex7_X.npy")
raise "K-means source dimensions differ" unless clusters.size == 300 && clusters.first.size == 2

pixels, w, h = PNGImage.read(CourseData.path(:kmeans, "bird_small.png"))
raise "Original bird dimensions differ" unless [w, h, pixels.size] == [128, 128, 16384]

[1, 2].each do |part|
  %W[X_part#{part}.npy X_val_part#{part}.npy y_val_part#{part}.npy].each { |name| CourseData.npy(:anomaly, "data/#{name}") }
end
{
  heart: ["heart.csv"],
  collaborative: %w[data/small_movies_Y.csv data/small_movies_R.csv data/small_movies_X.csv data/small_movies_W.csv data/small_movies_b.csv data/small_movie_list.csv],
  content: %w[data/content_user_train.csv data/content_item_train.csv data/content_y_train.csv data/content_item_vecs.csv data/content_movie_list.csv],
  evaluation: %w[data/data_w3_ex1.csv data/data_w3_ex2.csv data/c2w3_lab2_data1.csv data/c2w3_lab2_data2.csv data/c2w3_lab2_data3.csv data/c2w3_lab2_data4.csv],
  dqn: ["lunar_lander_model.h5"]
}.each { |dataset, names| names.each { |name| CourseData.path(dataset, name) } }
puts "Original clustering, image, anomaly, heart, movie, evaluation and Lunar Lander assets found."
