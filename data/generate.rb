# frozen_string_literal: true

# Regenerate the shipped course-shaped datasets: `ruby data/generate.rb`.
#
# The course's own files are not redistributable, except heart.csv (ODbL,
# shipped unchanged). Every other file here keeps the original file name,
# format, column layout, dtype and array shape, so the examples run the
# course workflow unchanged and the originals can be dropped in place of
# these (see data/README.md). Values come from documented generating models
# with fixed seeds; they are synthetic stand-ins, not the course measurements.
require "json"
require_relative "../lib/numpy_random"
require_relative "../lib/course_data"
require_relative "../lib/png_image"

module DatasetGenerator
  ROOT = __dir__

  module_function

  def path(dataset, file)
    target = File.join(ROOT, dataset, file)
    Dir.mkdir(File.dirname(File.dirname(target))) unless Dir.exist?(File.dirname(File.dirname(target)))
    Dir.mkdir(File.dirname(target)) unless Dir.exist?(File.dirname(target))
    target
  end

  def savetxt(dataset, file, rows, format: "%.18e")
    File.write(path(dataset, file), rows.map { |row| Array(row).map { |v| format(format, v) }.join(",") }.join("\n") + "\n")
  end

  def gauss(rng, mean, sd) = rng.normal(mean, sd)

  # C1 W2 assignment: city population (10,000s) → restaurant profit ($10,000s).
  def restaurants
    rng = NumpyRandom::RandomState.new(101)
    rows = Array.new(97) do
      population = 5.0 + 17.0 * rng.random_sample**2.2
      profit = -3.9 + 1.19 * population + gauss(rng, 0.0, 2.6 + 0.08 * population)
      [population.round(4), profit.round(4)]
    end
    File.write(path("linear", "data/ex1data1.txt"), rows.map { |r| r.map { |v| v.to_s }.join(",") }.join("\n") + "\n")
  end

  # C1 W3 assignment: two exam scores → admitted.
  def admissions
    rng = NumpyRandom::RandomState.new(102)
    rows = Array.new(100) do
      a = 30 + 70 * rng.random_sample
      b = 30 + 70 * rng.random_sample
      logit = -25.0 + 0.2 * a + 0.2 * b + gauss(rng, 0.0, 1.6)
      [a, b, logit.positive? ? 1 : 0]
    end
    File.write(path("logistic", "data/ex2data1.txt"), rows.map { |a, b, y| "#{a},#{b},#{y}" }.join("\n") + "\n")
  end

  # C1 W3 assignment: two microchip test results → accepted, a closed boundary.
  def microchips
    rng = NumpyRandom::RandomState.new(103)
    rows = Array.new(118) do
      a = -0.83 + 1.9 * rng.random_sample
      b = -0.77 + 1.88 * rng.random_sample
      radius = Math.sqrt((a - 0.05)**2 + ((b - 0.1) / 1.1)**2)
      boundary = 0.62 + 0.12 * Math.sin(3 * Math.atan2(b, a))
      label = radius < boundary ? 1 : 0
      label = 1 - label if rng.random_sample < 0.08
      [a.round(6), b.round(5), label]
    end
    File.write(path("logistic", "data/ex2data2.txt"), rows.map { |a, b, y| "#{a},#{b},#{y}" }.join("\n") + "\n")
  end

  # C1 W2 labs: size (sqft), bedrooms, floors, age → price ($1000s).
  def houses
    rng = NumpyRandom::RandomState.new(104)
    rows = Array.new(100) do
      size = (800 + 2400 * rng.random_sample**1.3).round
      bedrooms = [[(size / 650.0 + gauss(rng, 0.3, 0.8)).round, 0].max, 5].min
      floors = rng.random_sample < 0.4 ? 2 : 1
      age = (12 + 95 * rng.random_sample).round
      price = 220 + 0.27 * size - 32.6 * bedrooms - 67.3 * floors - 1.47 * age + gauss(rng, 0.0, 6.0)
      [size, bedrooms, floors, age, [price, 150 + rng.random_sample * 20].max.round(1)]
    end
    savetxt("houses", "data/houses.txt", rows)
  end

  # C2 W3 optional labs: one-feature regression and two-feature classification.
  def evaluation
    rng = NumpyRandom::RandomState.new(105)
    rows = Array.new(50) do |i|
      x = 1651 + 2000.0 * i / 49
      [x, 432 + 0.24 * (x - 1651) + 4.5e-6 * (x - 1651)**2 + gauss(rng, 0.0, 12.0)]
    end
    savetxt("evaluation", "data/data_w3_ex1.csv", rows)
    rows = Array.new(200) do |i|
      x1 = 1000 + 9000.0 * i / 199
      x2 = (5000 * rng.random_sample).round(-1)
      boundary = 2500 + 1400 * Math.sin(x1 / 1500.0)
      label = x2 > boundary ? 1.0 : 0.0
      label = 1.0 - label if rng.random_sample < 0.06
      [x1, x2, label]
    end
    savetxt("evaluation", "data/data_w3_ex2.csv", rows)
    target = ->(x, second) { 120 + 0.035 * (x - 1000) + 1.3e-5 * (x - 2600)**2 + 9.5 * second }
    second = Array.new(1000) { (8 * rng.random_sample).round(4) }
    noise = Array.new(1000) { gauss(rng, 0.0, 16.0) }
    one = Array.new(100) { |i| 1000 + 3000.0 * i / 99 }
    y = one.each_with_index.map { |x, i| target.call(x, second[i]) + noise[i] }
    savetxt("evaluation", "data/c2w3_lab2_data1.csv", one.zip(y))
    savetxt("evaluation", "data/c2w3_lab2_data2.csv", one.each_with_index.map { |x, i| [x, second[i], y[i]] })
    extra = one.map { |x| (1.0e7 + 0.9e7 * rng.random_sample + 120 * x).round }
    savetxt("evaluation", "data/c2w3_lab2_data3.csv", one.each_with_index.map { |x, i| [extra[i], x, second[i], y[i]] })
    many = Array.new(1000) { |i| 1000 + 3000.0 * i / 999 }
    savetxt("evaluation", "data/c2w3_lab2_data4.csv", many.each_with_index.map { |x, i| [x, second[i], target.call(x, second[i]) + noise[i]] })
  end

  # C3 W1 K-means: three two-dimensional clusters, 300 points.
  def clusters
    rng = NumpyRandom::RandomState.new(106)
    centers = [[1.95, 5.03], [3.04, 1.02], [6.03, 3.0]]
    points = centers.flat_map { |cx, cy| Array.new(100) { [gauss(rng, cx, 0.62), gauss(rng, cy, 0.48)] } }
    CourseData.write_npy(path("kmeans", "data/ex7_X.npy"), rng.shuffle!(points), fortran: true)
  end

  # C3 W1 image compression: a 128×128 RGB photograph-like scene.
  def bird
    rng = NumpyRandom::RandomState.new(107)
    pixels = []
    128.times do |row|
      128.times do |column|
        sky = [0.42 + 0.3 * row / 127.0, 0.6 + 0.25 * row / 127.0, 0.86]
        color = row > 96 + 6 * Math.sin(column / 11.0) ? [0.36, 0.28 + 0.1 * Math.sin(column / 7.0), 0.16] : sky
        bx, by = (column - 62) / 34.0, (row - 60) / 22.0
        color = [0.88, 0.52, 0.18] if bx * bx + by * by < 1
        wx, wy = (column - 58) / 20.0, (row - 52 + 0.4 * (column - 58)) / 8.0
        color = [0.28, 0.2, 0.16] if wx * wx + wy * wy < 1
        hx, hy = (column - 93) / 11.0, (row - 47) / 11.0
        color = [0.82, 0.46, 0.15] if hx * hx + hy * hy < 1
        color = [0.05, 0.05, 0.05] if (column - 97)**2 + (row - 44)**2 < 4
        color = [0.95, 0.82, 0.2] if column.between?(103, 111) && (row - 48).abs < 2 - (column - 103) / 5.0
        color = [0.46, 0.33, 0.2] if row.between?(80, 84) && column.between?(20, 110)
        pixels << color.map { |v| (v + 0.035 * (rng.random_sample - 0.5)).clamp(0.0, 1.0) }
      end
    end
    PNGImage.write(path("kmeans", "bird_small.png"), pixels, 128, 128)
  end

  # C3 W1 anomaly detection: server throughput/latency, then 11 features.
  def anomalies
    rng = NumpyRandom::RandomState.new(108)
    [[1, 307, 307, [14.1, 15.0], [1.35, 1.31], 9], [2, 1000, 100, [4.9, -9.6, 13.8, -10.5, -8.0, 10.2, -6.0, 8.0, -6.3, 2.3, 8.5], [7.8, 7.3, 7.6, 9.2, 8.1, 9.5, 7.5, 9.3, 5.4, 8.4, 7.1], 10]].each do |part, train_count, validation_count, means, sds, anomalies|
      draw = lambda do |outlier|
        means.zip(sds).map { |mean, sd| outlier ? mean + (rng.random_sample < 0.5 ? -1 : 1) * sd * (3.2 + 1.6 * rng.random_sample) : gauss(rng, mean, sd) }
      end
      training = Array.new(train_count) { |i| draw.call(i < (part == 1 ? 6 : 12)) }
      labels = Array.new(validation_count) { |i| i < anomalies ? 1 : 0 }
      rng.shuffle!(labels)
      validation = labels.map { |label| draw.call(label == 1) }
      CourseData.write_npy(path("anomaly", "data/X_part#{part}.npy"), rng.shuffle!(training), fortran: true)
      CourseData.write_npy(path("anomaly", "data/X_val_part#{part}.npy"), validation, fortran: true)
      CourseData.write_npy(path("anomaly", "data/y_val_part#{part}.npy"), labels, dtype: "|u1")
    end
  end

  # Stroke skeletons for the ten digits in a unit box (x right, y up).
  def arc(cx, cy, rx, ry, from, to, steps = 14)
    Array.new(steps + 1) do |i|
      angle = (from + (to - from) * i / steps.to_f) * Math::PI / 180
      [cx + rx * Math.cos(angle), cy + ry * Math.sin(angle)]
    end
  end

  def glyph(digit)
    case digit
    when 0 then [arc(0.5, 0.5, 0.22, 0.35, 0, 360, 24)]
    when 1 then [[[0.52, 0.12], [0.52, 0.88]], [[0.38, 0.74], [0.52, 0.88]]]
    when 2 then [arc(0.5, 0.66, 0.2, 0.2, 160, -40) + [[0.3, 0.12], [0.72, 0.12]]]
    when 3 then [arc(0.48, 0.68, 0.19, 0.19, 150, -90), arc(0.48, 0.31, 0.21, 0.2, 90, -150)]
    when 4 then [[[0.62, 0.12], [0.62, 0.88], [0.28, 0.36], [0.76, 0.36]]]
    when 5 then [[[0.7, 0.88], [0.34, 0.88], [0.31, 0.55]] + arc(0.48, 0.34, 0.21, 0.21, 120, -150)]
    when 6 then [arc(0.6, 0.6, 0.25, 0.32, 70, 180) + arc(0.5, 0.31, 0.2, 0.2, 180, 540)]
    when 7 then [[[0.28, 0.88], [0.72, 0.88], [0.42, 0.12]]]
    when 8 then [arc(0.5, 0.69, 0.17, 0.17, 0, 360, 20), arc(0.5, 0.3, 0.21, 0.2, 0, 360, 20)]
    when 9 then [arc(0.5, 0.68, 0.19, 0.19, 0, 360, 20), arc(0.4, 0.4, 0.29, 0.56, 0, -70)]
    end
  end

  def segment_distance(px, py, ax, ay, bx, by)
    dx, dy = bx - ax, by - ay
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy + 1e-12)
    t = t.clamp(0.0, 1.0)
    Math.hypot(px - ax - t * dx, py - ay - t * dy)
  end

  # Course digits: 5000 images × 400 pixels, 500 per class in label order;
  # each row stores a 20×20 image column-major, exactly as the course file.
  def digits
    rng = NumpyRandom::RandomState.new(109)
    images, labels = [], []
    10.times do |digit|
      500.times do
        angle = (rng.random_sample - 0.5) * 0.45
        scale = 0.78 + 0.22 * rng.random_sample
        shear = (rng.random_sample - 0.5) * 0.35
        shift = [(rng.random_sample - 0.5) * 0.14, (rng.random_sample - 0.5) * 0.12]
        width = 0.045 + 0.035 * rng.random_sample
        strokes = glyph(digit).map do |stroke|
          stroke.map do |x, y|
            u, v = x - 0.5 + shear * (y - 0.5), y - 0.5
            [0.5 + shift[0] + scale * (u * Math.cos(angle) - v * Math.sin(angle)), 0.5 + shift[1] + scale * (u * Math.sin(angle) + v * Math.cos(angle))]
          end
        end
        segments = strokes.flat_map { |stroke| stroke.each_cons(2).to_a }
        pixels = Array.new(400, 0.0)
        20.times do |row|
          20.times do |column|
            px, py = (column + 0.5) / 20, 1 - (row + 0.5) / 20
            distance = segments.map { |(ax, ay), (bx, by)| segment_distance(px, py, ax, ay, bx, by) }.min
            value = (1.0 - (distance - width) / 0.05).clamp(0.0, 1.0)
            pixels[column * 20 + row] = value.zero? ? 0.0 : [value + 0.04 * (rng.random_sample - 0.5), 0.0].max
          end
        end
        images << pixels
        labels << [digit]
      end
    end
    CourseData.write_npy(path("digits", "data/X.npy"), images, fortran: true)
    CourseData.write_npy(path("digits", "data/y.npy"), labels, dtype: "|u1")
  end

  NOTEBOOK_MOVIES = {
    246 => "Shrek (2001)", 366 => "Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)",
    382 => "Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)", 622 => "Harry Potter and the Chamber of Secrets (2002)",
    793 => "Pirates of the Caribbean: The Curse of the Black Pearl (2003)", 929 => "Lord of the Rings: The Return of the King, The (2003)",
    988 => "Eternal Sunshine of the Spotless Mind (2004)", 1150 => "Incredibles, The (2004)", 2609 => "Persuasion (2007)",
    2700 => "Toy Story 3 (2010)", 2716 => "Inception (2010)", 2925 => "Louis Theroux: Law & Disorder (2008)",
    2937 => "Nothing to Declare (Rien à déclarer) (2010)"
  }.freeze

  def csv_field(value)
    text = value.to_s
    text.match?(/[",\n]/) ? "\"#{text.gsub('"', '""')}\"" : text
  end

  # C3 W2 collaborative filtering: 4778 movies × 443 users, 39253 ratings in
  # half-star steps, popularity-skewed, plus 10-feature pretrained X, W and b.
  def collaborative
    rng = NumpyRandom::RandomState.new(110)
    movies, users, features = 4778, 443, 10
    x = Array.new(movies) { Array.new(features) { gauss(rng, 0.0, 0.7) } }
    w = Array.new(users) { Array.new(features) { gauss(rng, 0.0, 0.3) } }
    b = Array.new(users) { gauss(rng, 0.0, 0.25) }
    base = Array.new(movies) { gauss(rng, 3.45, 0.42) }
    popularity = Array.new(movies) { Math.exp(gauss(rng, 0.0, 1.25)) }
    NOTEBOOK_MOVIES.each_key { |i| popularity[i] *= 12 }
    activity = Array.new(users) { Math.exp(gauss(rng, 0.0, 0.9)) }
    cumulative = ->(weights) { weights.each_with_object([]) { |v, acc| acc << (acc.last || 0.0) + v } }
    movie_cdf, user_cdf = cumulative.call(popularity), cumulative.call(activity)
    pick = ->(cdf) { cdf.bsearch_index { |v| v >= rng.random_sample * cdf.last } || cdf.size - 1 }
    observed = {}
    movies.times { |i| observed[[i, pick.call(user_cdf)]] = true }
    observed[[pick.call(movie_cdf), pick.call(user_cdf)]] = true while observed.size < 39_253
    y = Array.new(movies) { Array.new(users, 0.0) }
    r = Array.new(movies) { Array.new(users, 0) }
    observed.each_key do |i, j|
      score = base[i] + x[i].zip(w[j]).sum { |a, c| a * c } + b[j] + gauss(rng, 0.0, 0.55)
      y[i][j] = ((score * 2).round / 2.0).clamp(0.5, 5.0)
      r[i][j] = 1
    end
    savetxt("collaborative", "data/small_movies_Y.csv", y, format: "%.1f")
    savetxt("collaborative", "data/small_movies_R.csv", r, format: "%d")
    savetxt("collaborative", "data/small_movies_X.csv", x, format: "%.8g")
    savetxt("collaborative", "data/small_movies_W.csv", w, format: "%.8g")
    savetxt("collaborative", "data/small_movies_b.csv", [b], format: "%.8g")
    lines = [",mean rating,number of ratings,title"]
    movies.times do |i|
      ratings = (0...users).select { |j| r[i][j] == 1 }.map { |j| y[i][j] }
      title = NOTEBOOK_MOVIES.fetch(i) { "Generated Movie #{format('%04d', i)} (#{2000 + i % 19})" }
      lines << [i, ratings.sum / ratings.size, ratings.size, csv_field(title)].join(",")
    end
    File.write(path("collaborative", "data/small_movie_list.csv"), lines.join("\n") + "\n")
  end

  GENRES = %w[Action Adventure Animation Children Comedy Crime Documentary Drama Fantasy Horror Mystery Romance Sci-Fi Thriller].freeze

  # C3 W2 content-based filtering: one training row per (rating, movie genre),
  # user rows carry per-genre average ratings, item rows one genre one-hot.
  def content
    rng = NumpyRandom::RandomState.new(111)
    movies = Array.new(64) do |i|
      genres = GENRES.each_index.to_a
      rng.shuffle!(genres)
      { id: 4000 + 37 * i, year: 2001 + (i * 7) % 18, genres: genres.first(1 + (rng.random_sample * 2.6).floor).sort, quality: gauss(rng, 3.5, 0.4) }
    end
    user_ids = [2, 36] + Array.new(28) { |i| 40 + 13 * i }
    ratings = user_ids.map do |id|
      taste = Array.new(GENRES.size) { gauss(rng, 0.0, 0.7) }
      order = movies.each_index.to_a
      rng.shuffle!(order)
      seen = order.first(8 + (rng.random_sample * 10).floor)
      [id, seen.map do |m|
        movie = movies[m]
        score = movie[:quality] + movie[:genres].sum { |g| taste[g] } / movie[:genres].size + gauss(rng, 0.0, 0.45)
        [m, ((score * 2).round / 2.0).clamp(0.5, 5.0)]
      end]
    end
    movie_average = Hash.new { |h, k| h[k] = [] }
    ratings.each { |_id, rated| rated.each { |m, value| movie_average[m] << value } }
    item_row = lambda do |m, genre|
      average = movie_average[m].empty? ? movies[m][:quality].round(2) : movie_average[m].sum / movie_average[m].size
      [movies[m][:id], movies[m][:year], average] + GENRES.each_index.map { |g| g == genre ? 1 : 0 }
    end
    users, items, targets = [], [], []
    ratings.each do |id, rated|
      genre_averages = GENRES.each_index.map do |g|
        values = rated.select { |m, _v| movies[m][:genres].include?(g) }.map(&:last)
        values.empty? ? 0.0 : values.sum / values.size
      end
      user = [id, rated.size, rated.sum(&:last) / rated.size] + genre_averages
      rated.each do |m, value|
        movies[m][:genres].each do |genre|
          users << user
          items << item_row.call(m, genre)
          targets << [value]
        end
      end
    end
    plain = :to_s.to_proc
    File.write(path("content", "data/content_user_train.csv"), users.map { |u| u.map { |v| v.to_f.to_s }.join(",") }.join("\n") + "\n")
    File.write(path("content", "data/content_item_train.csv"), items.map { |row| row.map(&plain).join(",") }.join("\n") + "\n")
    File.write(path("content", "data/content_y_train.csv"), targets.map { |(v)| format("%.1f", v) }.join("\n") + "\n")
    vectors = movies.each_index.flat_map { |m| movies[m][:genres].map { |g| item_row.call(m, g) } }
    File.write(path("content", "data/content_item_vecs.csv"), vectors.map { |row| row.map(&plain).join(",") }.join("\n") + "\n")
    list = ["movieId,title,genres"] + movies.each_with_index.map { |movie, i| [movie[:id], csv_field("Generated Feature #{format('%02d', i)} (#{movie[:year]})"), movie[:genres].map { |g| GENRES[g] }.join("|")].join(",") }
    File.write(path("content", "data/content_movie_list.csv"), list.join("\n") + "\n")
    File.write(path("content", "data/content_user_train_header.txt"), (["user id", "rating count", "rating ave"] + GENRES).join(",") + "\n")
    File.write(path("content", "data/content_item_train_header.txt"), (["movie id", "year", "ave rating"] + GENRES).join(",") + "\n")
  end

  def all
    %i[restaurants admissions microchips houses evaluation clusters bird anomalies digits collaborative content].each do |name|
      started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      send(name)
      puts format("%-14s %.1fs", name, Process.clock_gettime(Process::CLOCK_MONOTONIC) - started)
    end
  end
end

DatasetGenerator.all if __FILE__ == $PROGRAM_NAME
