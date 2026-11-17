# frozen_string_literal: true

# Model selection — fit on train, select degree/λ on validation, test once.
# Degree controls capacity; λ controls shrinkage. Never fit scaling on test rows.
# A small alternating-noise quadratic makes interpolation worse than the right bias.
# Run me: ruby evaluation/model_selection.rb
require_relative "../supervised/polynomial_regression"
module ModelSelection
  module_function
  def split(rows,targets,train: 0.6,validation: 0.2,rng: Random.new(42))
    raise ArgumentError,"invalid split fractions" unless train>0 && validation>0 && train+validation<1
    ids=rows.each_index.to_a.shuffle(random: rng)
    n=(rows.size*train).floor; v=(rows.size*validation).floor
    [ids.first(n),ids.slice(n,v),ids.drop(n+v)].map { |part| [part.map { |i| rows[i] },part.map { |i| targets[i] }] }
  end
  def fit(values,targets,degree:,l2: 0.0)
    features=PolynomialRegression.features(values,degree)
    scaler=NumericTools.fit_scaler(features)
    weights,bias=NormalEquation.fit(NumericTools.transform(features,scaler),targets,l2: l2)
    {degree: degree,scaler: scaler,weights: weights,bias: bias}
  end
  def predict(model,values)
    scaled=NumericTools.transform(PolynomialRegression.features(values,model[:degree]),model[:scaler])
    scaled.map { |row| LinearRegression.predict(row,model[:weights],model[:bias]) }
  end
end
if __FILE__ == $PROGRAM_NAME
  include Style
  banner("Model Selection", "capacity and regularization · validation chooses, test reports")
  x=[-1.0,-0.6,-0.2,0.2,0.6,1.0]
  y=x.each_with_index.map { |v,i| v*v+(i.even? ? 0.1 : -0.1) }
  validation=[-0.8,-0.4,0.0,0.4,0.8]; truth=validation.map { |v| v*v }
  candidates=[1,2,5].product([0.0,0.01,1.0]).map do |degree,lambda|
    model=ModelSelection.fit(x,y,degree: degree,l2: lambda)
    train_error=NumericTools.mse(y,ModelSelection.predict(model,x))
    cv_error=NumericTools.mse(truth,ModelSelection.predict(model,validation))
    step(format("degree=%d λ=%4.2f training MSE=%.6f validation MSE=%.6f",degree,lambda,train_error,cv_error))
    [cv_error,model,lambda]
  end
  _error,best,lambda=candidates.min_by(&:first)
  test=[-0.9,-0.3,0.3,0.9]
  test_error=NumericTools.mse(test.map { |v| v*v },ModelSelection.predict(best,test))
  step("selected degree #{best[:degree]}, λ=#{lambda}; untouched test MSE=#{test_error.round(6)}")
  raise "wrong capacity selected" unless best[:degree]==2 && test_error<0.02
  partitions=ModelSelection.split((0...20).map { |i| [i] },(0...20).to_a)
  raise "unaligned split" unless partitions.all? { |rows,labels| rows.map(&:first)==labels } && partitions.flat_map(&:last).sort==(0...20).to_a
  ok("validation rejects both an underfit line and noisy interpolation")
end
