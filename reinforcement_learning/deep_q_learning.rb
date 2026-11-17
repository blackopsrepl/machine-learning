# frozen_string_literal: true

# Course C3 W3 assignment: deep Q-learning on Lunar Lander.
# Q-network and target network 8→64→64→4 (ReLU, ReLU, linear), Adam(1e-3).
# Each environment step stores (s, a, r, s', done) in a 100,000-entry replay
# memory; every 4 steps (once it holds more than 64) a random minibatch of 64
# trains the Q-network on y = r + γ·max_a' Q_target(s', a')·(1−done), γ=0.995,
# by mean squared error on the taken action, then the target network moves
# τ=0.001 toward it. ε-greedy actions; ε decays ×0.995 per episode to 0.01.
# Up to 2000 episodes × 1000 steps; solved when the last-100 average ≥ 200.
# Random choices use Python's random stream seeded 0, as the course's utils.
# Run me: ruby reinforcement_learning/deep_q_learning.rb [--episodes N] [--steps N] [--evaluate FILE]
require "json"
require "optparse"
require_relative "lunar_lander"
require_relative "../neural_networks/sequential"
module DeepQLearning
  MEMORY_SIZE = 100_000
  GAMMA = 0.995
  ALPHA = 1e-3
  NUM_STEPS_FOR_UPDATE = 4
  MINIBATCH_SIZE = 64
  TAU = 1e-3
  E_DECAY = 0.995
  E_MIN = 0.01
  SEED = 0
  Experience = Struct.new(:state, :action, :reward, :next_state, :done)

  module_function

  def network(seed) = Sequential.new(LunarLander::OBSERVATIONS, [[64, :relu], [64, :relu], [LunarLander::ACTIONS, :linear]], rng: Random.new(seed))

  class Agent
    attr_reader :q_network, :target, :memory, :updates

    def initialize(seed: SEED)
      @random = NumpyRandom::PythonRandom.new(seed)
      @q_network = DeepQLearning.network(seed)
      @target = DeepQLearning.network(seed + 1)
      @target.weights = @q_network.weights
      @optimizer = Adam.new(learning_rate: ALPHA)
      @memory = []
      @head = 0
      @updates = 0
    end

    def action(state, epsilon)
      if @random.random > epsilon
        q = @q_network.logits([state]).first
        q.index(q.max)
      else
        @random.choice([0, 1, 2, 3])
      end
    end

    def remember(experience)
      if @memory.size < MEMORY_SIZE
        @memory << experience
      else
        @memory[@head] = experience
        @head = (@head + 1) % MEMORY_SIZE
      end
    end

    # Oldest-first view of the ring buffer, indexed like the course's deque.
    Window = Struct.new(:items, :head) do
      def size = items.size
      def [](i) = items[(head + i) % items.size]
    end

    def ordered = Window.new(@memory, @head)

    def compute_loss(batch)
      next_q = @target.logits(batch.map(&:next_state))
      targets = batch.each_with_index.map { |e, i| e.reward + GAMMA * next_q[i].max * (e.done ? 0.0 : 1.0) }
      activations, logits = @q_network.forward(batch.map(&:state))
      q = logits.last
      width = LunarLander::ACTIONS
      delta = Array.new(q.size, 0.0)
      loss = 0.0
      batch.each_with_index do |e, i|
        error = q[i * width + e.action] - targets[i]
        loss += error * error / batch.size
        delta[i * width + e.action] = 2.0 * error / batch.size
      end
      [loss, activations, logits, delta]
    end

    def learn
      batch = @random.sample(ordered, MINIBATCH_SIZE)
      loss, activations, logits, delta = compute_loss(batch)
      gradients, = @q_network.backward(activations, logits, delta)
      @optimizer.apply(@q_network.tensors, gradients)
      @target.tensors.zip(@q_network.tensors).each do |target, online|
        target.each_index { |i| target[i] = TAU * online[i] + (1.0 - TAU) * target[i] }
      end
      @updates += 1
      loss
    end

    def train(env, episodes: 2000, max_steps: 1000, solved_at: 200.0)
      history = []
      epsilon = 1.0
      episodes.times do |episode|
        state = episode.zero? ? env.reset(seed: SEED) : env.reset
        total = 0.0
        loss = nil
        max_steps.times do |t|
          action = action(state, epsilon)
          next_state, reward, terminated, truncated = env.step(action)
          done = terminated || truncated
          remember(Experience.new(state, action, reward, next_state, done))
          loss = learn if ((t + 1) % NUM_STEPS_FOR_UPDATE).zero? && @memory.size > MINIBATCH_SIZE
          state = next_state
          total += reward
          break if done
        end
        history << total
        average = history.last(100).sum / history.last(100).size
        epsilon = [E_MIN, E_DECAY * epsilon].max
        yield episode + 1, total, average, epsilon, loss if block_given?
        break if average >= solved_at
      end
      history
    end
  end

  def save(model, path)
    File.write(path, JSON.generate(model.weights))
  end

  def load(path)
    DeepQLearning.network(0).tap { |model| model.weights = JSON.parse(File.read(path)) }
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  options = { episodes: 2000, steps: 1000, evaluate: nil }
  OptionParser.new do |parser|
    parser.on("--episodes N", Integer) { |n| options[:episodes] = n }
    parser.on("--steps N", Integer) { |n| options[:steps] = n }
    parser.on("--evaluate FILE") { |path| options[:evaluate] = path }
  end.parse!
  raise "positive episode and step counts required" unless options[:episodes].positive? && options[:steps].positive?

  banner("Deep Q-Learning — Lunar Lander", "replay memory · target network · ε-greedy · Adam")
  env = LunarLander.new(seed: DeepQLearning::SEED)
  if options[:evaluate]
    model = DeepQLearning.load(options[:evaluate])
    returns = Array.new(options[:episodes]) do |episode|
      state = env.reset(seed: episode)
      total = 0.0
      options[:steps].times do
        q = model.logits([state]).first
        state, reward, terminated, truncated = env.step(q.index(q.max))
        total += reward
        break if terminated || truncated
      end
      step("greedy episode #{episode + 1}: return=#{total.round(2)}")
      total
    end
    step("mean return=#{(returns.sum / returns.size).round(2)}")
  else
    agent = DeepQLearning::Agent.new
    step("Q-network #{agent.q_network.count_params} parameters; target network synchronized")
    started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    history = agent.train(env, episodes: options[:episodes], max_steps: options[:steps]) do |episode, total, average, epsilon, loss|
      step("episode #{episode}: return=#{total.round(2)}; last-100 average=#{average.round(2)}; ε=#{epsilon.round(4)}; memory=#{agent.memory.size}; updates=#{agent.updates}; loss=#{loss&.round(4).inspect}")
    end
    minutes = (Process.clock_gettime(Process::CLOCK_MONOTONIC) - started) / 60
    average = history.last(100).sum / history.last(100).size
    step(average >= 200 ? "environment solved in #{history.size} episodes (#{minutes.round(1)} min)" : "stopped after #{history.size} episodes at last-100 average #{average.round(2)} (#{minutes.round(1)} min)")
    Dir.mkdir("output") unless Dir.exist?("output")
    DeepQLearning.save(agent.q_network, "output/lunar_lander.json")
    step("Q-network weights: output/lunar_lander.json")
    raise "training produced invalid returns" unless history.all?(&:finite?)
  end
  ok("Q-learning with replay and a soft-updated target network on the simulated lander")
end
