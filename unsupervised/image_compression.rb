# frozen_string_literal: true

# Image color quantization — each RGB pixel is a point, each centroid a color.
# Store an index per pixel plus a palette instead of a full RGB triple per pixel.
# A tiny inline 2×4 image keeps every assignment inspectable; no image library.
# Run me: ruby unsupervised/image_compression.rb
require_relative "k_means"
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Image Palette Compression", "K-means on RGB pixels · two-color reconstruction")
  pixels=[[1.0,0.0,0.0],[0.9,0.1,0.0],[1.0,0.1,0.0],[0.8,0.0,0.1],[0.0,0.0,1.0],[0.0,0.1,0.9],[0.1,0.0,1.0],[0.0,0.1,0.8]]
  palette,indices=KMeans.fit(pixels,2,initial: [pixels.first,pixels.last])
  palette.each_with_index { |color,i| step("palette #{i}: #{vector(color)}") }
  pixels.each_index { |i| note("pixel #{i}: #{vector(pixels[i])} → index #{indices[i]} → #{vector(palette[indices[i]])}") }
  original_bits=pixels.size*24
  palette_bits=palette.size*24+pixels.size
  step("conceptual 8-bit RGB storage: #{original_bits} bits → #{palette_bits} bits (excluding file headers)")
  raise "quantization failed" unless indices.uniq.size==2 && KMeans.objective(pixels,palette,indices)<0.03 && palette_bits<original_bits
  ok("the reconstructed image uses two colors instead of eight RGB triples")
end
