# frozen_string_literal: true

# Binary logistic regression — p=σ(w·x+b), dJ/dz=p-y.
# Stable loss softplus(z)-y*z avoids log(0); bias is excluded from L2.
# A threshold acts on probability, not on the training loss.
# Run me: ruby supervised/logistic_regression.rb
require_relative "../lib/style"
require_relative "../lib/problem"
require_relative "../lib/numeric"
module LogisticRegression
  module_function
  def sigmoid(z)
    z >= 0 ? 1.0/(1+Math.exp(-z)) : Math.exp(z)/(1+Math.exp(z))
  end
  def softplus(z) = [z,0.0].max + Math.log(1+Math.exp(-z.abs))
  def probability(row,w,b) = sigmoid(NumericTools.dot(row,w)+b)
  def loss(x,y,w,b,l2: 0.0)
    total = x.zip(y).sum { |row,target| z=NumericTools.dot(row,w)+b; softplus(z)-target*z }
    (total+l2*w.sum { |v| v*v }/2).to_f/x.size
  end
  def gradients(x,y,w,b,l2: 0.0)
    errors = x.zip(y).map { |row,target| probability(row,w,b)-target }
    dw = w.each_index.map { |j| (x.each_index.sum { |i| errors[i]*x[i][j] }+l2*w[j]).to_f/x.size }
    [dw,NumericTools.mean(errors)]
  end
  def fit(x,y,rate: 0.2,iterations: 800,l2: 0.0)
    w,b = Array.new(x[0].size,0.0),0.0
    iterations.times do |i|
      dw,db=gradients(x,y,w,b,l2: l2)
      w=w.zip(dw).map { |v,g| v-rate*g }; b-=rate*db
      yield i+1,w.dup,b,loss(x,y,w,b,l2: l2) if block_given?
    end
    [w,b]
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Logistic Regression", "stable cross-entropy · explicit gradients")
  x,y=Problem.binary,Problem.binary_targets
  w,b=LogisticRegression.fit(x,y) { |i,weights,bias,cost| step(format("it %-3d w=%.4f b=%.4f J=%.6f",i,weights[0],bias,cost)) if i<=3 || (i%200).zero? }
  predictions=x.map do |row|
    p=LogisticRegression.probability(row,w,b)
    step(format("x=% .0f logit=% .4f p=%.5f class=%d",row[0],NumericTools.dot(row,w)+b,p,p>=0.5 ? 1 : 0))
    p>=0.5 ? 1 : 0
  end
  raise "classification failed" unless predictions==y
  raise "unstable extreme logits" unless [-1000.0,1000.0].all? { |z| LogisticRegression.softplus(z).finite? && LogisticRegression.sigmoid(z).finite? }
  ok("the learned boundary separates the classes; extreme logits remain finite")
end
