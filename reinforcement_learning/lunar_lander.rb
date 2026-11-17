# frozen_string_literal: true

# Gym 0.26 LunarLander-v2, ported to Ruby on the Box2D port in lib/box2d.rb.
# State (8): x, y relative to the pad, vx, vy (scaled), angle, angular speed,
# left/right leg contact. Actions (4): nothing, left engine, main engine,
# right engine. Reward: change in shaping −100·distance −100·speed
# −100·|angle| +10 per leg contact, −0.3 per main-engine frame, −0.03 per
# side-engine frame; −100 on crash or leaving the screen, +100 when the
# lander comes to rest; episodes are cut at 1000 steps. Terrain, the initial
# push and engine dispersion draw from NumPy's PCG64 stream seeded like Gym.
# Run me: ruby reinforcement_learning/lunar_lander.rb
require_relative "../lib/style"
require_relative "../lib/box2d"
require_relative "../lib/numpy_random"
class LunarLander
  FPS = 50
  SCALE = 30.0
  MAIN_ENGINE_POWER = 13.0
  SIDE_ENGINE_POWER = 0.6
  INITIAL_RANDOM = 1000.0
  LANDER_POLY = [[-14, 17], [-17, 0], [-17, -10], [17, -10], [17, 0], [14, 17]].freeze
  LEG_AWAY = 20
  LEG_DOWN = 18
  LEG_W = 2
  LEG_H = 8
  LEG_SPRING_TORQUE = 40
  SIDE_ENGINE_HEIGHT = 14.0
  SIDE_ENGINE_AWAY = 12.0
  VIEWPORT_W = 600
  VIEWPORT_H = 400
  MAX_EPISODE_STEPS = 1000
  OBSERVATIONS = 8
  ACTIONS = 4

  attr_reader :lander, :legs, :terrain, :helipad_y, :steps

  # BeginContact: the hull touching ground ends the game; legs report contact.
  class ContactDetector
    def initialize(env) = @env = env

    def begin_contact(contact)
      bodies = [contact.fixture_a.body, contact.fixture_b.body]
      @env.game_over = true if bodies.include?(@env.lander)
      @env.legs.each { |leg| leg.user_data[:ground_contact] = true if bodies.include?(leg) }
    end

    def end_contact(contact)
      bodies = [contact.fixture_a.body, contact.fixture_b.body]
      @env.legs.each { |leg| leg.user_data[:ground_contact] = false if bodies.include?(leg) }
    end
  end

  attr_accessor :game_over

  def initialize(seed: nil)
    @rng = NumpyRandom::Generator.new(seed || Random.new_seed % 2**64)
  end

  def reset(seed: nil)
    @rng = NumpyRandom::Generator.new(seed) if seed
    @world = Box2D::World.new(gravity: [0.0, -10.0])
    @world.listener = ContactDetector.new(self)
    @game_over = false
    @prev_shaping = nil
    @steps = 0
    w = VIEWPORT_W / SCALE
    h = VIEWPORT_H / SCALE
    chunks = 11
    height = Array.new(chunks + 1) { @rng.uniform(0, h / 2) }
    chunk_x = Array.new(chunks) { |i| w / (chunks - 1) * i }
    @helipad_x1, @helipad_x2 = chunk_x[chunks / 2 - 1], chunk_x[chunks / 2 + 1]
    @helipad_y = h / 4
    (-2..2).each { |offset| height[chunks / 2 + offset] = @helipad_y }
    smooth = Array.new(chunks) { |i| 0.33 * (height[i - 1] + height[i] + height[i + 1]) }
    moon = @world.create_body(:static)
    moon.add_fixture(Box2D::Edge.new([0.0, 0.0], [w, 0.0]))
    @terrain = (0...chunks - 1).map do |i|
      edge = [[chunk_x[i], smooth[i]], [chunk_x[i + 1], smooth[i + 1]]]
      moon.add_fixture(Box2D::Edge.new(*edge), density: 0.0, friction: 0.1)
      edge
    end
    initial_y = VIEWPORT_H / SCALE
    @lander = @world.create_body(:dynamic, position: [VIEWPORT_W / SCALE / 2, initial_y])
    @lander.add_fixture(Box2D::Polygon.new(LANDER_POLY.map { |x, y| [x / SCALE, y / SCALE] }), density: 5.0, friction: 0.1, category: 0x0010, mask: 0x001, restitution: 0.0)
    @lander.apply_force_to_center([@rng.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), @rng.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)])
    @legs = [-1, 1].map do |i|
      leg = @world.create_body(:dynamic, position: [VIEWPORT_W / SCALE / 2 - i * LEG_AWAY / SCALE, initial_y], angle: i * 0.05)
      leg.add_fixture(Box2D::Polygon.box(LEG_W / SCALE, LEG_H / SCALE), density: 1.0, restitution: 0.0, category: 0x0020, mask: 0x001)
      leg.user_data = { ground_contact: false }
      lower, upper = i == -1 ? [0.9 - 0.5, 0.9] : [-0.9, -0.9 + 0.5]
      @world.create_joint(Box2D::RevoluteJoint.new(@lander, leg, anchor_a: [0, 0], anchor_b: [i * LEG_AWAY / SCALE, LEG_DOWN / SCALE], lower: lower, upper: upper, motor_speed: 0.3 * i, max_motor_torque: LEG_SPRING_TORQUE))
      leg
    end
    @steps = -1
    step(0).first
  end

  def float32(v) = [v].pack("e").unpack1("e")

  # Returns [state, reward, terminated, truncated].
  def step(action)
    raise ArgumentError, "action must be 0..3" unless (0...ACTIONS).cover?(action)

    tip = [Math.sin(@lander.angle), Math.cos(@lander.angle)]
    side = [-tip[1], tip[0]]
    dispersion = Array.new(2) { @rng.uniform(-1.0, 1.0) / SCALE }
    m_power = 0.0
    if action == 2
      m_power = 1.0
      ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
      oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
      at = [@lander.position[0] + ox, @lander.position[1] + oy]
      @lander.apply_linear_impulse([-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power], at)
    end
    s_power = 0.0
    if [1, 3].include?(action)
      direction = action - 2
      s_power = 1.0
      ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
      oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
      at = [@lander.position[0] + ox - tip[0] * 17 / SCALE, @lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE]
      @lander.apply_linear_impulse([-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power], at)
    end
    @world.step(1.0 / FPS, 6 * 30, 2 * 30)
    @steps += 1
    pos, vel = @lander.position, @lander.linear_velocity
    w, h = VIEWPORT_W / SCALE, VIEWPORT_H / SCALE
    state = [
      (pos[0] - w / 2) / (w / 2),
      (pos[1] - (@helipad_y + LEG_DOWN / SCALE)) / (h / 2),
      vel[0] * (w / 2) / FPS,
      vel[1] * (h / 2) / FPS,
      @lander.angle,
      20.0 * @lander.angular_velocity / FPS,
      @legs[0].user_data[:ground_contact] ? 1.0 : 0.0,
      @legs[1].user_data[:ground_contact] ? 1.0 : 0.0
    ].map { |v| float32(v) }
    shaping = -100 * Math.sqrt(state[0]**2 + state[1]**2) - 100 * Math.sqrt(state[2]**2 + state[3]**2) - 100 * state[4].abs + 10 * state[6] + 10 * state[7]
    reward = @prev_shaping ? shaping - @prev_shaping : 0.0
    @prev_shaping = shaping
    reward -= m_power * 0.30
    reward -= s_power * 0.03
    terminated = false
    if @game_over || state[0].abs >= 1.0
      terminated = true
      reward = -100.0
    end
    unless @lander.awake
      terminated = true
      reward = 100.0
    end
    [state, reward, terminated, @steps >= MAX_EPISODE_STEPS]
  end

  # Gym's hand-written controller for this environment (heuristic()).
  def self.heuristic(s)
    angle_targ = (s[0] * 0.5 + s[2] * 1.0).clamp(-0.4, 0.4)
    hover_targ = 0.55 * s[0].abs
    angle_todo = (angle_targ - s[4]) * 0.5 - s[5] * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - s[3] * 0.5
    if s[6].positive? || s[7].positive?
      angle_todo = 0
      hover_todo = -s[3] * 0.5
    end
    if hover_todo > angle_todo.abs && hover_todo > 0.05 then 2
    elsif angle_todo < -0.05 then 3
    elsif angle_todo > 0.05 then 1
    else 0
    end
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Lunar Lander", "Gym's LunarLander-v2 in Ruby · Box2D contacts, joints and sleeping")
  env = LunarLander.new
  state = env.reset(seed: 0)
  step("terrain heights #{vector(env.terrain.map { |(a, _b)| a[1] })}; pad at y=#{env.helipad_y.round(3)}")
  step("initial state #{vector(state)}")
  results = [0, 1, 2].map do |seed|
    state = env.reset(seed: seed)
    total = 0.0
    loop do
      state, reward, terminated, truncated = env.step(LunarLander.heuristic(state))
      total += reward
      break if terminated || truncated
    end
    outcome = env.lander.awake ? (env.game_over ? "crashed" : "time limit or off screen") : "landed and came to rest"
    step("seed #{seed}: Gym heuristic controller, #{env.steps} steps, return #{total.round(2)}, #{outcome}; legs #{state[6].to_i}/#{state[7].to_i}")
    total
  end
  raise "the heuristic controller should land on most seeds" unless results.count { |r| r > 200 } >= 2

  ok("the hand-written Gym controller lands the simulated lander")
end
