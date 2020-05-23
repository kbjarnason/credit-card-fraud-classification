#%% md
**FIN580 HW3**
By Kristian Bjarnason

#%%
#Relevant packages
using Pkg, DataFrames, CSV, LinearAlgebra, Dates, Statistics, MLJ, MLJBase, Plots, Flux, EvalCurves
using Flux:outdims, activations

#%%md
*1. Data Preparation*
Divide your sample into two equal sub-samples. You must keep the proportion of frauds the same in each sub-sample (246 frauds in each). You are going to use one sub-sample to estimate (train) your models and the second one to evaluate the out-of-sample performance of each model.
#%%
using Pkg, DataFrames, CSV, LinearAlgebra, Dates, Statistics, MLJ, MLJBase, Plots, Flux, EvalCurves
using Flux:outdims, activations

#Set drive and import data
cd("/Users/kristianbjarnason/Documents/Princeton/Courses/Spring 2020/FIN580 - Quantitative Data Analysis in Finance/HW/HW3")
data = CSV.read("credicard.csv")

data_fraud = filter(row -> row[:Class] == 1, data)
data_notfraud = filter(row -> row[:Class] == 0, data)

#split into training and test data, training with 1 extra row due to odd numbber of non-fraudulent claims and each with even number of fraudulent claims
data_train = vcat(data_fraud[1:round(Int, nrow(data_fraud)/2),:], data_notfraud[1:round(Int, nrow(data_notfraud)/2),:])
sort!(data_train, :Time)
data_test = vcat(data_fraud[round(Int, nrow(data_fraud)/2)+1:nrow(data_fraud),:], data_notfraud[round(Int, nrow(data_notfraud)/2)+1:nrow(data_notfraud),:])
sort!(data_test, :Time)

#Setup train and test arrays/vectors
X_train = DataFrames.select(data_train, Not(:Class))
X_test = DataFrames.select(data_test, Not(:Class))
y_train = categorical(data_train.Class)
y_train_int = data_train.Class
y_test = categorical(data_test.Class)
y_test_int = data_test.Class

#%%md
*2. Estimation of models*
You should estimate three different models: (1) logit; (2) support vector machines; (3) neural network. For each model, you have to justify the choices of the hyper-parameters (for example, for NN models, you should comments on the choice of the number of layers and units in each layer)
#%%md
Logit
#%%
model_logit = @load LinearBinaryClassifier

logit = machine(model_logit, X_train, y_train)

r = range(model_logit, :cost, lower=0.001, upper=1.0, scale=:log)
self_tuning_svm_model = TunedModel(model=model_svm,
                                                  resampling = CV(nfolds=3),
                                                  tuning = Grid(resolution=5),
                                                  range = r,
                                                  measure = misclassification_rate)
self_tuning_logit = machine(self_tuning_logit_model, X_train, y_train)
fit!(self_tuning_logit)
MLJ.save("tuned_logit.jlso", self_tuning_svm)

yhat_logit_tuned_p = predict(self_tuning_logit, X_test)
yhat_logit_tuned = predict_mode(self_tuning_logit, X_test)

misclassification_rate(yhat__tuned_logit, y_test)

#%%
#OG untuned logit
model_logit = @load LinearBinaryClassifier
logit = machine(model_logit, X_train, y_train)
evaluate!(logit,resampling=CV(nfolds=5),measure=[cross_entropy], verbosity=0)
fit!(logit)

yhat_logit_p = predict(logit, X_test)
yhat_logit = predict_mode(logit, X_test)

CSV.write("yhat_logit_p.csv", yhat_logit_p)
CSV.write("yhat_logit.csv", yhat_logit)

misclassification_rate(yhat_logit, y_test)

#%%md
Support Vector Machine
#%%
#tuned kernel svm mk 2
model_svm = @load SVC
svc = machine(model_svm, X_train, y_train)
r = range(model_svm, :cost, lower=0.001, upper=1.0, scale=:log)
self_tuning_svm_model = TunedModel(model=model_svm,
                                                  resampling = CV(nfolds=3),
                                                  tuning = Grid(resolution=3),
                                                  range = r,
                                                  measure = misclassification_rate)
self_tuning_svm = machine(self_tuning_svm_model, X_train, y_train)
fit!(self_tuning_svm)
MLJ.save("tuned_svm.jlso", self_tuning_svm)

MLJBase.params(self_tuning_svm)

yhat_tuned_svm2 = predict(self_tuning_svm, X_test)
misclassification_rate(yhat_tuned_svm, y_test)

CSV.write("yhat_tuned_svm2.csv",yhat_tuned_svm2)

#%%
#tuned kernel svm mk1 (gave same as og...)
model_svm = @load SVC
svc = machine(model_svm, X_train, y_train)
r = range(model_svm, :cost, lower=0.001, upper=1.0, scale=:log)
self_tuning_svm_model = TunedModel(model=model_svm,
                                                  resampling = CV(nfolds=2),
                                                  tuning = Grid(resolution=3),
                                                  range = r,
                                                  measure = misclassification_rate)
self_tuning_svm = machine(self_tuning_svm_model, X_train, y_train)
fit!(self_tuning_svm)
MLJ.save("tuned_svm.jlso", self_tuning_svm)

MLJBase.params(self_tuning_svm)

yhat_tuned_svm = predict(self_tuning_svm, X_test)
misclassification_rate(yhat_tuned_svm, y_test)

CSV.write("yhat_tuned_svm.csv",yhat_tuned_svm)

#%%
#initial testing, no tuning
model_svm = @load SVC
svc = machine("svc_full.jlso")
yhat_svm = predict(svc, X_test)
misclassification_rate(yhat_svm, y_test)
yhat_svm
CSV.write("yhat_svm.csv",yhat_svm)

#%%md
Neural Network
#%%
using CSV
using StatsBase
using Statistics
using Flux
using DataFrames
using UrlDownload
# using AUC # add git@github.com:paulstey/AUC.jl.git
using MLBase
using StatsBase

# Download the data
cd("/Users/kristianbjarnason/Documents/Princeton/Courses/Spring 2020/FIN580 - Quantitative Data Analysis in Finance/HW/HW3")
data = CSV.read("credicard.csv")

data[:Amount] = log.(data[:Amount] .+ 1e-6)

# The scale of the columns is quite variable -- we need to normalize each column
# Looking at the tutorial: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
# The steps are to log the Amount and then standard scale everything
# Obviously we only do this using the training data
n_rows = size(data, 1)
train_idx = sample(1:n_rows, Int(floor(n_rows * .7)))
remainder = setdiff(1:n_rows, train_idx)
validation_idx = sample(remainder, Int(floor(size(remainder, 1) * .2)))
test_idx = setdiff(remainder, validation_idx)

# Remove time column (index 1) and log amount column (index 30)
train = data[train_idx,2:31]
test = data[test_idx,2:31]
valid = data[validation_idx,2:31]

# train = [(tuple(x[2:29]..., log(x[30] + 1e-6)), x[31]) for x in data[train_idx, :]]
# test = [(tuple(x[2:29]..., log(x[30] + 1e-6)), x[31]) for x in data[test_idx, :]]
# valid = [(tuple(x[2:29]..., log(x[30] + 1e-6)), x[31]) for x in data[validation_idx, :]]

# Now let's scale
means_train, sd_train =  Array{Float32}(undef, 29),  Array{Float32}(undef, 29)
means = colwise(mean, train)
sd = colwise(std, train)

training = train
testing = test
validation = valid
for i in 1:29
  training[i] = (training[:,i] .- means[i]) ./ sd[i]
  testing[i] = (testing[:,i] .- means[i]) ./ sd[i]
  validation[i] = (validation[:,i] .- means[i]) ./ sd[i]
end

# training = [(collect((v - means[i]) / sd[i] for (i, v) in enumerate(x[1])), x[2]) for x in train]
# testing = [(collect((v - means[i]) / sd[i] for (i, v) in enumerate(x[1])), x[2]) for x in test]
# validation = [(collect((v - means[i]) / sd[i] for (i, v) in enumerate(x[1])), x[2]) for x in valid]

m = Chain(
  Dense(29, 16, relu),
  Dense(16, 1, σ)
)

loss(x, y) = Flux.binarycrossentropy(m(x), y)
valid_X = Array(validation[:,1:29])'
valid_y = validation[:Class]

# valid_X = [x[1] for x in validation]
# valid_y = [x[2] for x in validation]

progress() = sum(loss.(valid_X, valid_y))
ps = Flux.params(m)
opt = ADAM()
# Custom training loop

for batch in 1:400
    batch_data = sample(Array(training[1:29])', 32)
    Flux.train!(loss, ps, batch_data, opt)
    println(progress())
end

# Evaluate on test data
# I'm not familiar with confusmat...
y_pred = [x[1] for x in m.([d[1] for d in testing])]
y_test = [d[2] for d in testing]

y_thresh = (y_pred[:] .> .5) .+ 1
confusmat(2, y_test[:] .+ 1, y_thresh[:])

#%%
# x = Array(X_train[1:50000,:])'
# y = y_train_int[1:50000]

x = Array(X_train)'
y = y_train_int

data = [(x, y)]

m = Chain(
  Dense(30, 16, relu),
  Dense(16, 1, σ))

loss(x, y) = sum(Flux.crossentropy(m(x), y))
ps = Flux.params(m)
# opt = Momentum(0.01)
opt = ADAM(1e-3)

Flux.train!(loss, ps, data, opt)
m(Array(X_test[1:50000,:])')
sum(m(Array(X_test[1:50000,:])'))

# yhat_nn = categorical(vec(m(Array(X_test)')))
# misclassification_rate(yhat_nn, y_test)

#%%md
*3. OOS results*
For each estimated model, report the following out-of-sample results:
#%%md
(a) Confusion matrix
#%%
cm_logit = confusion_matrix(yhat_logit,y_test)
cm_svm = confusion_matrix(yhat_pred_svm,y_test)
cm_nn = confusion_matrix(yhat_pred_nn,y_test)

#%%md
(b) ROC curves
#%%
plot(roc_curve(yhat_logit_p,y_test))
plot(roc_curve(yhat_svm_p,y_test))
plot(roc_curve(yhat_nn_p,y_test))

#%%md
(c) Precision-Recall curve
A precision-Recall curve is a plot of Precision versus Recall,
where Precision = True Positives/(TruePositives + FalsePositives) Recall = TruePositives/(TruePositives + FalseNegatives)
#%%
plot(prcurve(pdf.(yhat_logit_p,1), y_test_int))
plot(prcurve(pdf.(yhat_svm_p,1), y_test_int))
plot(prcurve(pdf.(yhat_nn_p,1), y_test_int))

#%%md
*4. Discussion of Results*
Comment on your results:
#%%md
*(a) Which model performs the best?*


#%%md
*(b) What are the main differences of each model?*
Logit:


SVM:


Neural Network:
