# frozen_string_literal: true

# C3 W3: the original LunarLander-v2 Box2D environment, not a toy replacement.
# Eight observations, four actions, 8→64→64→4 online/target networks; replay
# 100000, minibatches 64, gamma=.995, Adam(.001), update every four steps,
# soft target tau=.001, epsilon decay .995/min .01; default 2000×1000 budget.
require "optparse"
require_relative "../lib/style"
require_relative "../lib/reference"
module DeepQLearning
  Experience = Struct.new(:state, :action, :reward, :next_state, :done, keyword_init: true)
  class Agent
    attr_reader :online, :target, :updates, :memory

    def initialize(seed: 0)
      @rng = Random.new(seed)
      Reference.seed(seed)
      tf = Reference.tensorflow
      build = lambda do
        tf.keras.Sequential.new([
                                  tf.keras.Input(shape: [8]),
                                  tf.keras.layers.Dense.new(64, activation: "relu"),
                                  tf.keras.layers.Dense.new(64, activation: "relu"),
                                  tf.keras.layers.Dense.new(4, activation: "linear")
                                ])
      end
      @online, @target = build.call, build.call
      @target.set_weights(@online.get_weights)
      @optimizer = tf.keras.optimizers.Adam.new(learning_rate: 0.001)
      @memory, @cursor, @updates = [], 0, 0
    end

    def action(state, epsilon: 0.0)
      return @rng.rand(4) unless @rng.rand > epsilon

      values = @online.__call__(Reference.tensor([state]))
      Reference.numpy.argmax(values.numpy).item
    end

    def remember(experience)
      if @memory.size < 100_000
        @memory << experience
      else
        @memory[@cursor] = experience
        @cursor = (@cursor + 1) % 100_000
      end
    end

    def loss(experiences)
      tf = Reference.tensorflow
      states = Reference.tensor(experiences.map(&:state))
      following = Reference.tensor(experiences.map(&:next_state))
      actions = Reference.array(experiences.map(&:action)).astype("int32")
      rewards = Reference.tensor(experiences.map(&:reward))
      done = Reference.tensor(experiences.map { |e| e.done ? 1.0 : 0.0 })
      future = tf.reduce_max(@target.__call__(following), axis: 1)
      targets = tf.add(rewards, tf.multiply(0.995, tf.multiply(future, tf.subtract(1.0, done))))
      values = @online.__call__(states)
      selected = tf.gather(values, actions, axis: 1, batch_dims: 1)
      tf.keras.losses.mean_squared_error(targets, selected)
    end

    def learn
      tf = Reference.tensorflow
      batch = @memory.sample(64, random: @rng)
      tape = tf.GradientTape.new
      tape.__enter__
      begin
        objective = loss(batch)
      ensure
        tape.__exit__(nil, nil, nil)
      end
      variables = @online.trainable_variables.to_a
      gradients = tape.gradient(objective, variables).to_a
      @optimizer.apply_gradients(gradients.zip(variables).map { |g, v| PyCall.tuple([g, v]) })
      @target.weights.to_a.zip(@online.weights.to_a).each do |target_weight, weight|
        target_weight.assign(tf.add(tf.multiply(0.001, weight), tf.multiply(0.999, target_weight)))
      end
      @updates += 1
      objective.numpy.item
    end

    def train(environment, episodes: 2000, max_steps: 1000)
      epsilon = 1.0
      returns = []
      episodes.times do |episode|
        state = Reference.list(environment.reset.to_a.first)
        total, last_loss = 0.0, nil
        max_steps.times do |t|
          chosen = action(state, epsilon: epsilon)
          observation, reward, terminated, truncated, _info = environment.step(chosen).to_a
          reward = Reference.scalar(reward)
          terminated, truncated = Reference.scalar(terminated), Reference.scalar(truncated)
          following = Reference.list(observation)
          # Gym 0.26 split the old four-result done flag; combine it to retain
          # the notebook's original replay and episode-termination semantics.
          done = terminated || truncated
          remember(Experience.new(state: state, action: chosen, reward: reward, next_state: following, done: done))
          last_loss = learn if ((t + 1) % 4).zero? && @memory.size > 64
          state = following
          total += reward
          break if done
        end
        returns << total
        epsilon = [0.01, 0.995 * epsilon].max
        average = returns.last(100).sum / returns.last(100).size
        yield episode + 1, total, average, epsilon, last_loss if block_given?
        break if average >= 200
      end
      returns
    end
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  options = { episodes: 2000, steps: 1000, evaluate: nil }
  OptionParser.new do |parser|
    parser.on("--episodes N", Integer) { |n| options[:episodes] = n }
    parser.on("--steps N", Integer) { |n| options[:steps] = n }
    parser.on("--evaluate MODEL") { |path| options[:evaluate] = path }
  end.parse!
  raise "positive training budgets required" unless options[:episodes] > 0 && options[:steps] > 0

  banner("Deep Q-Learning — Lunar Lander", "original LunarLander-v2 · Box2D · TensorFlow/Keras")
  environment = PyCall.import_module("gym").make("LunarLander-v2")
  begin
    environment.reset(seed: 0)
    agent = DeepQLearning::Agent.new
    step("observation shape=#{environment.observation_space.shape}; actions=#{environment.action_space.n}; model=#{agent.online.count_params} parameters")
    if options[:evaluate]
      agent.online.set_weights(Reference.tensorflow.keras.models.load_model(File.expand_path(options[:evaluate]), compile: false).get_weights)
      options[:episodes].times do |episode|
        state = Reference.list(environment.reset.to_a.first)
        total = 0.0
        options[:steps].times do
          observation, reward, terminated, truncated, _info = environment.step(agent.action(state)).to_a
          reward = Reference.scalar(reward)
          terminated, truncated = Reference.scalar(terminated), Reference.scalar(truncated)
          state = Reference.list(observation)
          total += reward
          break if terminated || truncated
        end
        step("evaluation episode #{episode + 1}: return=#{total}")
      end
    else
      returns = agent.train(environment, episodes: options[:episodes], max_steps: options[:steps]) do |episode, total, average, epsilon, loss|
        step("episode #{episode}: return=#{total}; last-100 average=#{average}; epsilon=#{epsilon}; replay=#{agent.memory.size}; gradient updates=#{agent.updates}; batch MSE=#{loss.inspect}")
      end
      raise "Lunar Lander produced invalid returns" unless returns.all?(&:finite?)

      Dir.mkdir("output") unless Dir.exist?("output")
      agent.online.save("output/lunar_lander.keras")
      step("trained weights: output/lunar_lander.keras")
    end
    raise "Original environment dimensions changed" unless environment.action_space.n == 4 && environment.observation_space.shape.to_a == [8]

    ok("the real Lunar Lander simulator and original DQN training configuration are active")
  ensure
    environment.close
  end
end
