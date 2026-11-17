# frozen_string_literal: true

# Deep Q-learning — online/target networks, replay, epsilon-greedy exploration.
# Target y=r for terminal transitions, otherwise r+γ max_a Q_target(s',a).
# Differentiate ONLY the selected action's squared error; softly update target
# weights after each minibatch. All network gradients and Adam updates are Ruby.
# The course's LunarLander environment is replaced with a tiny continuous-state
# corridor: the same DQN algorithm, not a claim to reproduce Box2D physics.
# Run me: ruby reinforcement_learning/deep_q_learning.rb
require_relative "../neural_networks/backpropagation"
module DeepQLearning
  Experience = Struct.new(:state, :action, :reward, :next_state, :terminal, keyword_init: true)
  class Corridor
    def reset(rng: Random.new(42), position: nil)
      @position = position || 0.1 + 0.8 * rng.rand
      [@position]
    end

    def step(action)
      @position += action.zero? ? -0.2 : 0.2
      terminal = @position <= 0 || @position >= 1
      reward = terminal ? (@position <= 0 ? 1.0 : 0.4) : -0.02
      [[@position], reward, terminal]
    end
  end

  class Agent
    attr_reader :online, :target, :replay, :updates

    def initialize(rng: Random.new(42), gamma: 0.7, capacity: 1000)
      @rng, @gamma, @capacity = rng, gamma, capacity
      @online = Backpropagation::Network.new([1, 12, 12, 2], activations: [:relu, :relu, :linear], rng: rng)
      @target = Backpropagation::Network.new([1, 12, 12, 2], activations: [:relu, :relu, :linear], rng: rng)
      @target.parameters = @online.parameters
      @optimizer = Adam.new(rate: 0.005)
      @replay = []
      @updates = 0
    end

    def action(state, epsilon: 0.0)
      return @rng.rand(2) if @rng.rand < epsilon

      q = @online.predict(state)
      q.each_index.max_by { |a| q[a] }
    end

    def remember(experience)
      @replay.shift if @replay.size >= @capacity
      @replay << experience
    end

    def bellman(experience)
      experience.reward + (experience.terminal ? 0.0 : @gamma * @target.predict(experience.next_state).max)
    end

    def learn(batch_size: 16, tau: 0.05)
      return nil if @replay.size < batch_size

      batch = @replay.sample(batch_size, random: @rng)
      rows = batch.map(&:state)
      targets = batch.map do |experience|
        values = @online.predict(experience.state)
        values[experience.action] = bellman(experience)
        values
      end
      # Other action targets equal their current predictions: their deltas are zero.
      loss, gradient = @online.gradients(rows, targets, loss: :mse)
      @online.parameters = @optimizer.step(@online.parameters, gradient)
      @target.parameters = @target.parameters.zip(@online.parameters).map { |old, new_value| (1 - tau) * old + tau * new_value }
      @updates += 1
      [loss, batch.first, bellman(batch.first)]
    end

    def train(episodes: 600)
      env = Corridor.new
      steps = 0
      episodes.times do |episode|
        state = env.reset(rng: @rng)
        epsilon = [1.0 * 0.99**episode, 0.05].max
        reward_sum = 0.0
        loss = nil
        30.times do
          selected = action(state, epsilon: epsilon)
          following, reward, terminal = env.step(selected)
          remember(Experience.new(state: state, action: selected, reward: reward, next_state: following, terminal: terminal))
          steps += 1
          learned = learn if (steps % 4).zero?
          loss = learned[0] if learned
          reward_sum += reward
          state = following
          break if terminal
          # A time limit truncates a rollout, not the MDP: stored terminal remains false.
        end
        yield episode + 1, epsilon, reward_sum, loss if block_given?
      end
      self
    end
  end

  module_function

  def rollout(agent, start, gamma: 0.7)
    env = Corridor.new
    state = env.reset(position: start)
    total = 0.0
    trace = []
    30.times do |t|
      q = agent.online.predict(state)
      action = q.each_index.max_by { |a| q[a] }
      following, reward, terminal = env.step(action)
      total += gamma**t * reward
      trace << [state[0], q, action, reward, terminal]
      return [total, trace] if terminal

      state = following
    end
    [total, trace]
  end

  # Independent exact return for each monotone path; any detour wastes reward.
  def oracle(start, gamma: 0.7)
    left = (start / 0.2).ceil
    right = ((1 - start) / 0.2).ceil
    [[left, 1.0], [right, 0.4]].map { |n, reward| gamma**(n - 1) * reward - 0.02 * (1 - gamma**(n - 1)) / (1 - gamma) }.max
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Deep Q-learning", "continuous corridor · replay + target network · seed 42")
  note("a tiny navigation environment replaces LunarLander; both actions and all Bellman updates stay visible")
  agent = DeepQLearning::Agent.new
  terminal = DeepQLearning::Experience.new(state: [0.1], action: 0, reward: 1.0, next_state: [-0.1], terminal: true)
  raise "terminal bootstrapping" unless agent.bellman(terminal) == 1.0

  initial = [0.1, 0.3, 0.5, 0.7, 0.9].map { |x| DeepQLearning.rollout(agent, x)[0] }
  agent.train do |episode, epsilon, reward, loss|
    step(format("episode %-3d ε=%.3f return=% .3f replay=%d updates=%d batch loss=%s", episode, epsilon, reward, agent.replay.size, agent.updates, loss&.round(6).inspect)) if episode <= 2 || (episode % 100).zero?
  end
  results = [0.1, 0.3, 0.5, 0.7, 0.9].map do |start|
    total, trace = DeepQLearning.rollout(agent, start)
    step("start #{start} discounted return=#{total.round(5)} exact best=#{DeepQLearning.oracle(start).round(5)}")
    trace.each { |position, q, a, r, done| note("x=#{position.round(3)} Q=#{vector(q)} #{a.zero? ? 'left' : 'right'} reward=#{r} terminal=#{done}") }
    [total, DeepQLearning.oracle(start)]
  end
  learned = NumericTools.mean(results.map(&:first))
  optimal = NumericTools.mean(results.map(&:last))
  step("mean discounted return: before=#{NumericTools.mean(initial).round(5)} trained=#{learned.round(5)} exact=#{optimal.round(5)}")
  raise "DQN did not learn near-optimal navigation" unless learned >= 0.95 * optimal && agent.updates > 0
  raise "replay exceeds capacity" if agent.replay.size > 1000

  ok("a learned Q function chooses the rewarding terminal; terminal targets never bootstrap")
end
