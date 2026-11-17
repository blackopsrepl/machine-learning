# frozen_string_literal: true

# NumPy's seeded random streams, reproduced bit for bit in Ruby.
#
# The course seeds NumPy (np.random.seed, RandomState via scikit-learn's
# random_state, and Gym's PCG64 Generator). Reproducing those streams lets the
# Ruby examples regenerate the course's exact synthetic datasets, splits and
# Lunar Lander terrain instead of approximating them.
module NumpyRandom
  MASK32 = 0xffff_ffff
  MASK64 = 0xffff_ffff_ffff_ffff
  MASK128 = (1 << 128) - 1

  # Legacy RandomState: MT19937 seeded with init_genrand(seed), as NumPy does
  # for an integer seed. Doubles use the 53-bit res53 construction.
  class RandomState
    N = 624
    M = 397

    def initialize(seed)
      raise ArgumentError, "seed must be in 0...2**32" unless seed.is_a?(Integer) && seed.between?(0, MASK32)

      @mt = Array.new(N)
      @mt[0] = seed
      (1...N).each { |i| @mt[i] = (1_812_433_253 * (@mt[i - 1] ^ (@mt[i - 1] >> 30)) + i) & MASK32 }
      @index = N
      @gauss = nil
    end

    def next_uint32
      generate if @index >= N
      y = @mt[@index]
      @index += 1
      y ^= y >> 11
      y ^= (y << 7) & 0x9d2c_5680
      y ^= (y << 15) & 0xefc6_0000
      y ^ (y >> 18)
    end

    def random_sample
      a = next_uint32 >> 5
      b = next_uint32 >> 6
      (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
    end

    def uniform(low = 0.0, high = 1.0) = low + (high - low) * random_sample

    # Polar Box–Muller with NumPy's cached second deviate.
    def standard_normal
      if @gauss
        value, @gauss = @gauss, nil
        return value
      end
      loop do
        x1 = 2.0 * random_sample - 1.0
        x2 = 2.0 * random_sample - 1.0
        r2 = x1 * x1 + x2 * x2
        next if r2 >= 1.0 || r2 == 0.0

        f = Math.sqrt(-2.0 * Math.log(r2) / r2)
        @gauss = f * x1
        return f * x2
      end
    end

    def normal(loc = 0.0, scale = 1.0) = loc + scale * standard_normal

    # Uniform integer in 0..max by masked rejection (NumPy random_interval).
    def interval(max)
      return 0 if max.zero?

      mask = max
      [1, 2, 4, 8, 16, 32].each { |shift| mask |= mask >> shift }
      if max <= MASK32
        loop do
          value = next_uint32 & mask
          return value if value <= max
        end
      else
        loop do
          value = ((next_uint32 << 32) | next_uint32) & mask
          return value if value <= max
        end
      end
    end

    # In-place Fisher–Yates from the end, as RandomState.shuffle.
    def shuffle!(array)
      (array.size - 1).downto(1) do |i|
        j = interval(i)
        array[i], array[j] = array[j], array[i]
      end
      array
    end

    def permutation(n) = shuffle!((0...n).to_a)

    private

    def generate
      N.times do |i|
        y = (@mt[i] & 0x8000_0000) | (@mt[(i + 1) % N] & 0x7fff_ffff)
        @mt[i] = @mt[(i + M) % N] ^ (y >> 1) ^ (y.odd? ? 0x9908_b0df : 0)
      end
      @index = 0
    end
  end

  # CPython's random module: MT19937 seeded by init_by_array over the seed's
  # 32-bit words; random() is res53; integers come from getrandbits with
  # rejection (_randbelow), as in choice() and sample().
  class PythonRandom < RandomState
    def initialize(seed)
      super(19_650_218)
      key = []
      value = seed.abs
      loop do
        key << (value & MASK32)
        value >>= 32
        break if value.zero?
      end
      init_by_array(key)
    end

    def random = random_sample

    def getrandbits(k)
      return next_uint32 >> (32 - k) if k <= 32

      words = (k + 31) / 32
      value = 0
      words.times do |i|
        bits = next_uint32
        bits >>= (32 * words - k) if i == words - 1
        value |= bits << (32 * i)
      end
      value
    end

    def randbelow(n)
      k = n.bit_length
      loop do
        r = getrandbits(k)
        return r if r < n
      end
    end

    def choice(array) = array[randbelow(array.size)]

    # random.sample(population, k) for sequences, including its two strategies.
    def sample(population, k)
      n = population.size
      raise ArgumentError, "sample larger than population" if k > n

      setsize = 21
      setsize += 4**Math.log(3 * k, 4).ceil if k > 5
      if n <= setsize
        pool = Array.new(n) { |i| population[i] }
        Array.new(k) do |i|
          j = randbelow(n - i)
          chosen = pool[j]
          pool[j] = pool[n - i - 1]
          chosen
        end
      else
        selected = {}
        Array.new(k) do
          j = randbelow(n)
          j = randbelow(n) while selected.key?(j)
          selected[j] = true
          population[j]
        end
      end
    end

    private

    def init_by_array(key)
      i, j = 1, 0
      [N, key.size].max.times do
        @mt[i] = ((@mt[i] ^ ((@mt[i - 1] ^ (@mt[i - 1] >> 30)) * 1_664_525)) + key[j] + j) & MASK32
        i += 1
        j += 1
        if i >= N
          @mt[0] = @mt[N - 1]
          i = 1
        end
        j = 0 if j >= key.size
      end
      (N - 1).times do
        @mt[i] = ((@mt[i] ^ ((@mt[i - 1] ^ (@mt[i - 1] >> 30)) * 1_566_083_941)) - i) & MASK32
        i += 1
        if i >= N
          @mt[0] = @mt[N - 1]
          i = 1
        end
      end
      @mt[0] = 0x8000_0000
      @index = N
    end
  end

  # SeedSequence entropy pool (pool size 4) and state generation.
  module SeedSequence
    INIT_A = 0x43b0_d7e5
    MULT_A = 0x931e_8875
    INIT_B = 0x8b51_f9dd
    MULT_B = 0x58f3_8ded
    MIX_MULT_L = 0xca01_f9dd
    MIX_MULT_R = 0x4973_f715
    POOL = 4

    module_function

    def words(entropy)
      return [0] if entropy.zero?

      out = []
      while entropy.positive?
        out << (entropy & MASK32)
        entropy >>= 32
      end
      out
    end

    def pool(entropy)
      input = words(entropy)
      constant = INIT_A
      hashmix = lambda do |value|
        value = (value ^ constant) & MASK32
        constant = (constant * MULT_A) & MASK32
        value = (value * constant) & MASK32
        value ^ (value >> 16)
      end
      mix = lambda do |x, y|
        result = (MIX_MULT_L * x - MIX_MULT_R * y) & MASK32
        result ^ (result >> 16)
      end
      mixer = Array.new(POOL) { |i| hashmix.call(input[i] || 0) }
      POOL.times do |source|
        POOL.times do |target|
          mixer[target] = mix.call(mixer[target], hashmix.call(mixer[source])) unless source == target
        end
      end
      (POOL...input.size).each do |source|
        POOL.times { |target| mixer[target] = mix.call(mixer[target], hashmix.call(input[source])) }
      end
      mixer
    end

    def state64(entropy, count)
      mixer = pool(entropy)
      constant = INIT_B
      words = Array.new(count * 2) do |i|
        value = (mixer[i % POOL] ^ constant) & MASK32
        constant = (constant * MULT_B) & MASK32
        value = (value * constant) & MASK32
        value ^ (value >> 16)
      end
      words.each_slice(2).map { |low, high| low | (high << 32) }
    end
  end

  # PCG64 (XSL-RR 128/64) behind numpy.random.Generator, seeded via SeedSequence.
  class Generator
    MULTIPLIER = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645

    def initialize(seed)
      raise ArgumentError, "nonnegative integer seed required" unless seed.is_a?(Integer) && seed >= 0

      words = SeedSequence.state64(seed, 4)
      initial_state = (words[0] << 64) | words[1]
      sequence = (words[2] << 64) | words[3]
      @increment = ((sequence << 1) | 1) & MASK128
      @state = 0
      advance
      @state = (@state + initial_state) & MASK128
      advance
    end

    def next_uint64
      advance
      rotation = @state >> 122
      value = ((@state >> 64) ^ @state) & MASK64
      ((value >> rotation) | (value << ((64 - rotation) & 63))) & MASK64
    end

    def random = (next_uint64 >> 11) * (1.0 / 9_007_199_254_740_992.0)
    def uniform(low = 0.0, high = 1.0) = low + (high - low) * random

    private

    def advance
      @state = (@state * MULTIPLIER + @increment) & MASK128
    end
  end
end
