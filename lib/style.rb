# frozen_string_literal: true

# The same small ANSI vocabulary as heuristics: no gems, no terminal framework.
module Style
  module_function

  def bold(text) = wrap("1", text)
  def dim(text) = wrap("2", text)
  def gold(text) = fg("220", text)
  def azure(text) = fg("75", text)
  def violet(text) = fg("141", text)
  def sage(text) = fg("114", text)
  def coral(text) = fg("210", text)
  def gray(text) = fg("245", text)
  def fg(code, text) = "\e[38;5;#{code}m#{text}\e[0m"
  def wrap(code, text) = "\e[#{code}m#{text}\e[0m"

  def banner(title, subtitle = nil)
    puts
    puts azure("──── ") + bold(violet(title)) + (subtitle ? gray("  ·  #{subtitle}") : "")
  end

  def step(text) = puts("   #{gray('·')} #{text}")
  def note(text) = puts("       #{dim(text)}")
  def ok(text) = puts("   #{sage('✓')} #{text}")
  def vector(values) = "[#{values.map { |v| format('% .4f', v) }.join(' ')}]"
end
