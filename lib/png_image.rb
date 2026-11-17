# frozen_string_literal: true

require "zlib"

# Lossless RGB/RGBA PNG reading/writing for the original bird_small image.
module PNGImage
  module_function

  def read(path)
    bytes = File.binread(path)
    raise "Not PNG" unless bytes.start_with?("\x89PNG\r\n\x1a\n".b)

    offset, compressed = 8, "".b
    width = height = channels = nil
    while offset < bytes.bytesize
      size = bytes.byteslice(offset, 4).unpack1("N")
      type = bytes.byteslice(offset + 4, 4)
      data = bytes.byteslice(offset + 8, size)
      if type == "IHDR"
        width, height, depth, color, _, _, interlace = data.unpack("NNC5")
        raise "Only non-interlaced 8-bit RGB/RGBA PNG is supported" unless depth == 8 && [2, 6].include?(color) && interlace.zero?

        channels = color == 2 ? 3 : 4
      elsif type == "IDAT"
        compressed << data
      end
      offset += 12 + size
    end
    raw = Zlib::Inflate.inflate(compressed).bytes
    stride = width * channels
    previous = Array.new(stride, 0)
    pixels = []
    height.times do |y|
      filter = raw[y * (stride + 1)]
      row = raw.slice(y * (stride + 1) + 1, stride)
      stride.times do |j|
        left = j >= channels ? row[j - channels] : 0
        above = previous[j]
        corner = j >= channels ? previous[j - channels] : 0
        predictor = case filter
                    when 0 then 0
                    when 1 then left
                    when 2 then above
                    when 3 then (left + above) / 2
                    when 4
                      p = left + above - corner
                      distances = [(p - left).abs, (p - above).abs, (p - corner).abs]
                      [left, above, corner][distances.index(distances.min)]
                    else raise "Unknown PNG filter #{filter}"
                    end
        row[j] = (row[j] + predictor) & 255
      end
      pixels.concat(row.each_slice(channels).map { |pixel| pixel.first(3).map { |v| v / 255.0 } })
      previous = row
    end
    [pixels, width, height]
  end

  def write(path, pixels, width, height)
    chunk = lambda { |type, data| [data.bytesize].pack("N") + type + data + [Zlib.crc32(type + data)].pack("N") }
    raw = pixels.each_slice(width).map { |row| "\0".b + row.flatten.map { |v| (v * 255).round.clamp(0, 255) }.pack("C*") }.join
    header = [width, height, 8, 2, 0, 0, 0].pack("NNC5")
    File.binwrite(path, "\x89PNG\r\n\x1a\n".b + chunk.call("IHDR", header) + chunk.call("IDAT", Zlib::Deflate.deflate(raw)) + chunk.call("IEND", "".b))
  end
end
