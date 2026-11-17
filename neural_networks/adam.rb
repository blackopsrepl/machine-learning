# frozen_string_literal: true

# Adam — exponential averages of gradients and squared gradients, bias corrected.
# m_hat=m/(1-β1^t), v_hat=v/(1-β2^t); θ -= α*m_hat/(sqrt(v_hat)+ε).
# The optimizer is explicit, not a library hiding behind a custom training loop.
# Run me: ruby neural_networks/adam.rb
require_relative "../lib/style"
class Adam
  def initialize(rate: 0.03,beta1: 0.9,beta2: 0.999,epsilon: 1e-8)
    @rate,@beta1,@beta2,@epsilon=rate,beta1,beta2,epsilon
    @m,@v,@t=nil,nil,0
  end
  def step(parameters,gradient)
    raise ArgumentError,"gradient shape differs" unless parameters.size==gradient.size
    @m ||= Array.new(parameters.size,0.0); @v ||= Array.new(parameters.size,0.0)
    raise ArgumentError,"reuse optimizer only for same parameter shape" unless @m.size==parameters.size
    @t+=1
    parameters.each_index.map do |i|
      @m[i]=@beta1*@m[i]+(1-@beta1)*gradient[i]
      @v[i]=@beta2*@v[i]+(1-@beta2)*gradient[i]**2
      mh=@m[i]/(1-@beta1**@t); vh=@v[i]/(1-@beta2**@t)
      updated=parameters[i]-@rate*mh/(Math.sqrt(vh)+@epsilon)
      yield @t,i,gradient[i],mh,vh,updated if block_given?
      updated
    end
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Adam", "first and second moments · bias correction")
  theta=[0.0]; optimizer=Adam.new(rate: 0.1)
  200.times do |i|
    theta=optimizer.step(theta,[2*(theta[0]-3)]) do |t,_j,g,m,v,value|
      step(format("t=%-3d g=% .4f m_hat=% .4f v_hat=%.4f θ=%.5f",t,g,m,v,value)) if i<3 || ((i+1)%50).zero?
    end
  end
  raise "Adam failed" unless (theta[0]-3).abs<0.001
  ok("moment estimates minimize (θ−3)² without autodiff")
end
