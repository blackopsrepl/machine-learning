# frozen_string_literal: true

# Original K-means image-compression assignment: bird_small.png, K=16,
# ten assignment/mean updates and a reconstructed 128×128 RGB image.
# Ruby's PNG reader preserves the source pixels without an image-library proxy.
require_relative "k_means"
if __FILE__ == $PROGRAM_NAME
  require_relative "../lib/course_data"
  require_relative "../lib/png_image"
  include Style
  banner("Image Palette Compression", "original bird_small.png · 16 colors · ten iterations")
  pixels, width, height = PNGImage.read(CourseData.path(:kmeans, "bird_small.png"))
  palette, indices = KMeans.fit(pixels, 16, iterations: 10, rng: Random.new(42)) do |iteration, _centers, _ids, cost|
    step("iteration #{iteration}: pixel distortion #{cost}")
  end
  recovered = indices.map { |i| palette[i] }
  Dir.mkdir("output") unless Dir.exist?("output")
  PNGImage.write("output/bird_compressed.png", recovered, width, height)
  decoded, w, h = PNGImage.read("output/bird_compressed.png")
  raise "Image reconstruction shape changed" unless w == width && h == height && decoded.size == pixels.size
  raise "Palette exceeds 16 colors" unless decoded.uniq.size <= 16

  step("#{width}×#{height}, #{pixels.size} actual pixels → #{palette.size} colors")
  step("reconstructed image: output/bird_compressed.png")
  ok("the original bird image is quantized and written as a real PNG")
end
