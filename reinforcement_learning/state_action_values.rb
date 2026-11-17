# frozen_string_literal: true

# Tabular policy iteration — the Python course's six-state left/right world.
# Q(s,a)=R(s)+γΣP(s'|s,a)V(s'); evaluate a policy, greedily improve, repeat.
# Terminal states have their terminal reward and no outgoing probability mass.
# This is model-based dynamic programming, not sampled Q-learning.
# Run me: ruby reinforcement_learning/state_action_values.rb
require_relative "../lib/style"
module StateActionValues
  module_function

  def transitions(states: 6, misstep: 0.0)
    Array.new(states) do |s|
      Array.new(2) do |action|
        p = Array.new(states, 0.0)
        unless s.zero? || s == states - 1
          intended = action.zero? ? s - 1 : s + 1
          other = action.zero? ? s + 1 : s - 1
          p[intended] = 1 - misstep
          p[other] = misstep
        end
        p
      end
    end
  end

  def q(s, a, rewards, p, values, gamma)
    rewards[s] + gamma * p[s][a].each_index.sum { |j| p[s][a][j] * values[j] }
  end

  def evaluate(policy, rewards, p, gamma: 0.5)
    values = Array.new(rewards.size, 0.0)
    10_000.times do
      next_values = values.each_index.map { |s| q(s, policy[s], rewards, p, values, gamma) }
      delta = values.zip(next_values).map { |a, b| (a - b).abs }.max
      values = next_values
      return values if delta < 1e-10
    end
    raise "policy evaluation did not converge"
  end

  def fit(rewards, p, gamma: 0.5)
    policy = Array.new(rewards.size, 0)
    100.times do |iteration|
      values = evaluate(policy, rewards, p, gamma: gamma)
      improved = policy.each_index.map { |s| [0, 1].max_by { |a| q(s, a, rewards, p, values, gamma) } }
      yield iteration + 1, improved.dup, values.dup if block_given?
      return [policy, values] if improved == policy

      policy = improved
    end
    raise "policy iteration did not stabilize"
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("State–Action Values", "six states · terminal rewards 100 and 40 · γ=0.5")
  rewards = [100.0, 0.0, 0.0, 0.0, 0.0, 40.0]
  [0.0, 0.4].each do |misstep|
    p = StateActionValues.transitions(misstep: misstep)
    policy, values = StateActionValues.fit(rewards, p) do |iteration, actions, v|
      step("misstep=#{misstep} policy iteration #{iteration}: #{actions.inspect}; V=#{vector(v)}")
    end
    values.each_index do |s|
      q = [0, 1].map { |a| StateActionValues.q(s, a, rewards, p, values, 0.5) }
      note("state #{s} Q(left,right)=#{vector(q)} action=#{policy[s].zero? ? 'left' : 'right'}")
      raise "Bellman optimality mismatch" unless (values[s] - q.max).abs < 1e-8
    end
    raise "terminal convention changed" unless values[0] == 100 && values[-1] == 40 && p[0].flatten.sum == 0
  end
  ok("policy improvement and Bellman values agree, including stochastic missteps")
end
