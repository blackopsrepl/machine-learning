# frozen_string_literal: true

# Copy the original course files into data/'s layout: the examples then run
# on the course's own measurements instead of the shipped stand-ins.
#   ruby data/import_course.rb /path/to/MachineLearningSpecialist [target]
require "fileutils"
require "json"

source, target = ARGV
abort "usage: ruby data/import_course.rb COURSE_ROOT [TARGET=data]" unless source
target ||= __dir__
mapping = JSON.parse(File.read(File.join(__dir__, "sources.json"), encoding: "UTF-8"))
mapping.each do |dataset, entry|
  entry.fetch("files").each do |file|
    from = File.join(source, entry.fetch("directory"), file)
    abort "missing #{from}" unless File.exist?(from)

    to = File.join(target, dataset, file)
    FileUtils.mkdir_p(File.dirname(to))
    FileUtils.cp(from, to)
    puts "#{dataset}/#{file}"
  end
end
