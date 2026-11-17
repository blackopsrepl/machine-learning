# frozen_string_literal: true

# L2 regularization — J_data + λ||w||²/(2m); gradient adds λw/m, never λb/m.
# Compare training fit with shrinkage. Smaller training error is not by itself
# better generalization; held-out model selection is a separate example.
# Run me: ruby supervised/regularization.rb
require_relative "normal_equation"
require_relative "logistic_regression"
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Regularization", "penalize weights, not the intercept")
  norms=[0.0,20.0].map do |lambda|
    w,b=NormalEquation.fit(Problem.line,Problem.line_targets,l2: lambda)
    step(format("linear λ=%4.1f w=% .4f b=% .4f data J=%.5f",lambda,w[0],b,LinearRegression.loss(Problem.line,Problem.line_targets,w,b)))
    w.sum { |v| v*v }
  end
  raise "L2 did not shrink" unless norms[1]<norms[0]
  [0.0,10.0].each do |lambda|
    w,b=LogisticRegression.fit(Problem.binary,Problem.binary_targets,l2: lambda)
    step(format("logistic λ=%4.1f w=% .4f b=% .4f",lambda,w[0],b))
  end
  w,b=NormalEquation.fit([[0.0],[0.0]],[7.0,7.0],l2: 10)
  raise "intercept penalized" unless b==7.0 && w==[0.0]
  ok("L2 shrinks the slope; an unpenalized constant prediction remains seven")
end
