# frozen_string_literal: true

require_relative "../lib/course_data"
require_relative "../lib/png_image"

# Every course file must keep the schema the examples rely on. Row counts are
# the course's own, except the content-based tables, whose size may vary.
def table_shape(dataset, file, rows, columns, skip: 0)
  data = CourseData.table(dataset, file, skip: skip)
  raise "#{dataset}/#{file}: expected #{rows || 'n'}×#{columns}, got #{data.size}×#{data.first&.size}" unless (rows.nil? || data.size == rows) && data.all? { |r| r.size == columns }

  puts "#{dataset}/#{file}: #{data.size}×#{columns}"
end

table_shape(:linear, "data/ex1data1.txt", 97, 2)
table_shape(:logistic, "data/ex2data1.txt", 100, 3)
table_shape(:logistic, "data/ex2data2.txt", 118, 3)
table_shape(:houses, "data/houses.txt", 100, 5)
table_shape(:evaluation, "data/data_w3_ex1.csv", 50, 2)
table_shape(:evaluation, "data/data_w3_ex2.csv", 200, 3)
[[1, 100, 2], [2, 100, 3], [3, 100, 4], [4, 1000, 3]].each { |n, rows, columns| table_shape(:evaluation, "data/c2w3_lab2_data#{n}.csv", rows, columns) }
table_shape(:collaborative, "data/small_movies_Y.csv", 4778, 443)
table_shape(:collaborative, "data/small_movies_R.csv", 4778, 443)
table_shape(:collaborative, "data/small_movies_X.csv", 4778, 10)
table_shape(:collaborative, "data/small_movies_W.csv", 443, 10)
table_shape(:collaborative, "data/small_movies_b.csv", 1, 443)
header, movies = CourseData.csv(:collaborative, "data/small_movie_list.csv")
raise "movie list schema differs" unless header == ["", "mean rating", "number of ratings", "title"] && movies.size == 4778

x, y = CourseData.digits(:multiclass)
raise "digit images differ from 5000×400 with labels 0..9" unless x.size == 5000 && x.all? { |r| r.size == 400 } && y.tally.keys.sort == (0..9).to_a
binary_x, binary_y = CourseData.digits(:binary)
raise "binary digit selection differs from 1000 zeros/ones" unless binary_x.size == 1000 && binary_y.tally == { 0 => 500, 1 => 500 }

puts "digits: 5000×400, labels 0..9; binary selection 1000×400"
raise "K-means data differs from 300×2" unless CourseData.npy(:kmeans, "data/ex7_X.npy").then { |a| a.size == 300 && a.all? { |r| r.size == 2 } }

pixels, w, h = PNGImage.read(CourseData.path(:kmeans, "bird_small.png"))
raise "bird image differs from 128×128 RGB" unless [w, h, pixels.size] == [128, 128, 16_384]

{ 1 => [307, 307, 2], 2 => [1000, 100, 11] }.each do |part, (train, validation, columns)|
  a = CourseData.npy(:anomaly, "data/X_part#{part}.npy")
  b = CourseData.npy(:anomaly, "data/X_val_part#{part}.npy")
  c = CourseData.npy(:anomaly, "data/y_val_part#{part}.npy")
  raise "anomaly part #{part} schema differs" unless a.size == train && b.size == validation && c.size == validation && (a + b).all? { |r| r.size == columns }
end
users = CourseData.table(:content, "data/content_user_train.csv")
items = CourseData.table(:content, "data/content_item_train.csv")
targets = CourseData.table(:content, "data/content_y_train.csv")
vectors = CourseData.table(:content, "data/content_item_vecs.csv")
raise "content tables differ in schema" unless users.size == items.size && items.size == targets.size && (users + items + vectors).all? { |r| r.size == 17 } && targets.all? { |r| r.size == 1 }

_header, catalogue = CourseData.csv(:content, "data/content_movie_list.csv")
raise "content movie list misses item vectors" unless vectors.map { |r| r[0].to_i }.uniq.all? { |id| catalogue.any? { |row| row[0].to_i == id } }

heart_header = File.readlines(CourseData.path(:heart, "heart.csv"), chomp: true, encoding: "UTF-8").first
raise "heart.csv header differs" unless heart_header == "Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope,HeartDisease"

puts "content: #{users.size} user/item/rating rows, #{vectors.size} item vectors"
puts "All course datasets present with the course schema (#{CourseData::ROOT})."
