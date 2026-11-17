# frozen_string_literal: true

# Course datasets live under data/<dataset>/ with the course's own file names.
# ML_DATA points at another tree with the same layout, e.g. the originals
# imported by `ruby data/import_course.rb /path/to/course`.
module CourseData
  ROOT = File.expand_path(ENV.fetch("ML_DATA", File.expand_path("../data", __dir__)))

  module_function

  def path(dataset, file)
    target = File.join(ROOT, dataset.to_s, file)
    raise "Course data missing: #{target}" unless File.exist?(target)

    target
  end

  def table(dataset, file, skip: 0)
    File.readlines(path(dataset, file), chomp: true, encoding: "UTF-8").drop(skip).filter_map do |line|
      next if line.strip.empty? || line.lstrip.start_with?("#")

      line.strip.split(/[\s,]+/).map { |value| Float(value) }
    end
  end

  def supervised(dataset, file, skip: 0)
    rows = table(dataset, file, skip: skip)
    [rows.map { |row| row[0...-1] }, rows.map(&:last)]
  end

  # Quoted-field CSV reader for the movie metadata; returns header and rows.
  def csv(dataset, file)
    lines = File.readlines(path(dataset, file), chomp: true, encoding: "UTF-8").reject(&:empty?)
    rows = lines.map { |line| fields(line) }
    [rows.first, rows.drop(1)]
  end

  def fields(line)
    out, field, quoted, i = [], +"", false, 0
    while i < line.size
      char = line[i]
      if quoted && char == '"' && line[i + 1] == '"'
        field << '"'
        i += 1
      elsif char == '"'
        quoted = !quoted
      elsif char == "," && !quoted
        out << field
        field = +""
      else
        field << char
      end
      i += 1
    end
    out << field
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

  def write_npy(target, rows, dtype: "<f8", fortran: false)
    matrix = rows.first.is_a?(Array)
    shape = matrix ? "(#{rows.size}, #{rows.first.size})" : "(#{rows.size},)"
    values = matrix && fortran ? rows.first.each_index.flat_map { |j| rows.map { |row| row[j] } } : rows.flatten
    header = "{'descr': '#{dtype}', 'fortran_order': #{matrix && fortran ? 'True' : 'False'}, 'shape': #{shape}, }"
    header += " " * (64 - (10 + header.size + 1) % 64) + "\n"
    payload = dtype == "|u1" ? values.pack("C*") : values.pack("E*")
    File.binwrite(target, "\x93NUMPY\x01\x00".b + [header.size].pack("v") + header + payload)
  end

  # The course's digit file holds 500 images per class in label order. The
  # binary assignment loads its first 1000 rows: exactly the zeros and ones,
  # which is how they are selected here so other row orders still work.
  def digits(kind)
    x = npy(:digits, "data/X.npy")
    y = npy(:digits, "data/y.npy").flatten.map(&:to_i)
    return [x, y] if kind == :multiclass
    raise ArgumentError, "digits kind is :binary or :multiclass" unless kind == :binary

    keep = y.each_index.select { |i| y[i] <= 1 }
    [keep.map { |i| x[i] }, keep.map { |i| y[i] }]
  end
end
