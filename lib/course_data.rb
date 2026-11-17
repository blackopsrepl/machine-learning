# frozen_string_literal: true

require "json"

# Read original course files. The binary assignment uses the first 1000 images,
# exactly as its source loader does; other application datasets remain complete.
# ML_COURSE_ROOT may point at another copy of the original course directory.
module CourseData
  ROOT = ENV.fetch("ML_COURSE_ROOT", "/srv/res/computing/DataEng/MachineLearningSpecialist")
  DIRECTORIES = JSON.parse(File.read(File.expand_path("../data/sources.json", __dir__)))

  module_function

  def path(dataset, file = nil)
    directory = File.join(ROOT, DIRECTORIES.fetch(dataset.to_s))
    target = file ? File.join(directory, file) : directory
    raise "Course source missing: #{target}; set ML_COURSE_ROOT to the course directory" unless File.exist?(target)

    target
  end

  def table(dataset, file)
    File.readlines(path(dataset, file)).filter_map do |line|
      next if line.strip.empty? || line.lstrip.start_with?("#")

      line.strip.split(/[\s,]+/).map { |value| Float(value) }
    end
  end

  def supervised(dataset, file)
    rows = table(dataset, file)
    [rows.map { |row| row[0...-1] }, rows.map(&:last)]
  end

  # NumPy's documented .npy format: numeric arrays, no pickle/object execution.
  def npy(dataset, file)
    bytes = File.binread(path(dataset, file))
    raise "Not a NumPy array" unless bytes.start_with?("\x93NUMPY".b)

    major = bytes.getbyte(6)
    header_size = major == 1 ? 2 : 4
    length = bytes.byteslice(8, header_size).unpack1(major == 1 ? "v" : "V")
    header = bytes.byteslice(8 + header_size, length)
    descriptor = header[/['"]descr['"]:\s*['"]([^'"]+)/, 1]
    shape = header[/['"]shape['"]:\s*\(([^)]*)/, 1].split(",").filter_map { |s| Integer(s.strip) unless s.strip.empty? }
    big = descriptor.start_with?(">")
    kind, width = descriptor[1], descriptor[2..].to_i
    format = case [kind, width]
             when ["f", 8] then big ? "G*" : "E*"
             when ["f", 4] then big ? "g*" : "e*"
             when ["i", 8] then big ? "q>*" : "q<*"
             when ["i", 4] then big ? "l>*" : "l<*"
             when ["u", 1] then "C*"
             else raise "Unsupported NumPy dtype #{descriptor}"
             end
    data = bytes.byteslice(8 + header_size + length..).unpack(format)
    raise "NumPy payload size differs from header" unless data.size == shape.inject(1, :*)
    return data if shape.size == 1
    raise "Only vectors/matrices supported, got #{shape.inspect}" unless shape.size == 2

    if header.match?(/['"]fortran_order['"]:\s*True/)
      Array.new(shape[0]) { |i| Array.new(shape[1]) { |j| data[j * shape[0] + i] } }
    else
      data.each_slice(shape[1]).to_a
    end
  end

  def digits(kind)
    x = npy(kind, "data/X.npy")
    y = npy(kind, "data/y.npy").flatten.map(&:to_i)
    kind.to_s == "binary_digits" ? [x.first(1000), y.first(1000)] : [x, y]
  end
end
