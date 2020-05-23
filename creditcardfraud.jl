#%% md
By Kristian Bjarnason

#%%
#Relevant packages
using Pkg, DataFrames, CSV, LinearAlgebra, Dates, Statistics, MLJ, MLJBase, Plots, Flux, EvalCurves
using Flux:outdims, activations

#%%md
*Data Preparation*
Divide the sample into two equal sub-samples. Keep the proportion of frauds the same in each sub-sample (246 frauds in each). Use one sub-sample to estimate (train) your models and the second one to evaluate the out-of-sample performance of each model.
#%%
using Pkg, DataFrames, CSV, LinearAlgebra, Dates, Statistics, MLJ, MLJBase, MLJLinearModels, Plots, Flux, EvalCurves
using Flux:outdims, activations

#Set drive and import data
cd("/Users/kristianbjarnason/Documents/Programming/Julia/creditcard")
data = CSV.read("credicard.csv")

data_fraud = filter(row -> row[:Class] == 1, data)
data_notfraud = filter(row -> row[:Class] == 0, data)

#split into training and test data, training wi th 1 extra row due to odd numbber of non-fraudulent claims and each with even number of fraudulent claims
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
*Estimation of models*
Estimate three different models: (1) logit; (2) support vector machines; (3) neural network.
#%%md
Logit
#%%
#initial logit classification with lambda = 0.2
X_train = DataFrames.select(data_train, Not(:Class))
X_test = DataFrames.select(data_test, Not(:Class))
y_train = categorical(data_train.Class)
y_train_int = data_train.Class
y_test = categorical(data_test.Class)
y_test_int = data_test.Class

@load LogisticClassifier pkg=MLJLinearModels
model_logit = LogisticClassifier(lambda=0.2)

logit = machine(model_logit, X_train, y_train)

fit!(logit)

yhat_logit_p = predict(logit, X_test)
yhat_logit = predict_mode(logit, X_test)

misclassification_rate(yhat_logit_p, y_test)

#%%

#%%
#tuned svm
X_train = DataFrames.select(data_train, Not(:Class))
X_test = DataFrames.select(data_test, Not(:Class))
y_train = categorical(data_train.Class)
y_train_int = data_train.Class
y_test = categorical(data_test.Class)
y_test_int = data_test.Class

model_logit = @load LogisticClassifier pkg=MLJLinearModels
r = range(model_logit, :lambda, lower=0.001, upper=1.0, scale=:linear)

self_tuning_logit_model = TunedModel(model=model_logit,
                                                  resampling = CV(nfolds=5),
                                                  tuning = Grid(resolution=5),
                                                  range = r,
                                                  measure = cross_entropy)

self_tuning_logit = machine(self_tuning_logit_model, X_train, y_train)

fit!(self_tuning_logit)

yhat_logit_tuned_p = predict(self_tuning_logit, X_test)
yhat_logit_tuned = predict_mode(self_tuning_logit, X_test)

misclassification_rate(yhat_logit_tuned, y_test)

#%%md
Support Vector Machine
#%%
X_train = DataFrames.select(data_train[1:5000,:], Not(:Class))
X_test = DataFrames.select(data_test[1:5000,:], Not(:Class))
y_train = categorical(data_train[1:5000,:].Class)
y_train_int = data_train[1:5000,:].Class
y_test = categorical(data_test[1:5000,:].Class)
y_test_int = data_test[1:5000,:].Class

#tuning was giving problems so manually trying different costs
@load SVC
model_svm1 = SVC(cost=1.0)
svc1 = machine(model_svm1, X_train, y_train)
fit!(svc1)

yhat_svm1 = predict(svc1, X_test)
misclassification_rate(yhat_svm1, y_test)
yhat_svm1

#%%
@load SVC
model_svm2 = SVC(cost=0.5)
svc2 = machine(model_svm2, X_train, y_train)
fit!(svc2)

yhat_svm2 = predict(svc2, X_test)
misclassification_rate(yhat_svm2, y_test)
yhat_svm2

#%%
@load SVC
model_svm3 = SVC(cost=0.1)
svc3 = machine(model_svm3, X_train, y_train)
fit!(svc3)

yhat_svm3 = predict(svc3, X_test)
misclassification_rate(yhat_svm3, y_test)
yhat_svm3

#%%md
Neural Network
#%%
#Adapted from the tutorial: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
#thanks to Michael Griffiths for his help putting this together too. https://pastebin.com/iwtCFN3F
using CSV
using StatsBase
using Statistics
using Flux
using DataFrames
using UrlDownload
# using AUC # add git@github.com:paulstey/AUC.jl.git
using MLBase
using StatsBase

cd("/Users/kristianbjarnason/Documents/Programming/Julia/creditcard")
data = CSV.read("credicard.csv")
# data = urldownload("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv")

data[:Amount] = log.(data[:Amount] .+ 1e-6)

# The scale of the columns is quite variable -- we need to normalize each column
# The steps are to log the Amount and then standard scale everything (only use training data for this)
n_rows = size(data, 1)
train_idx = sample(1:n_rows, Int(floor(n_rows * .7)))
remainder = setdiff(1:n_rows, train_idx)
validation_idx = sample(remainder, Int(floor(size(remainder, 1) * .2)))
test_idx = setdiff(remainder, validation_idx)

# Remove time column (index 1) and log amount column (index 30)
train = data[train_idx,2:31]
test = data[test_idx,2:31]
valid = data[validation_idx,2:31]

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

m = Chain(
  Dense(29, 16, relu),
  Dense(16, 1, σ)
)
# try crossentropy??
loss(x, y) = Flux.binarycrossentropy(m(x), y)
valid_X = Array(validation[:,1:29])'
valid_y = validation[:Class]

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
y_pred = [x[1] for x in m.([d[1] for d in testing])]
y_test = [d[2] for d in testing]

y_thresh = (y_pred[:] .> .5) .+ 1
confusmat(2, y_test[:] .+ 1, y_thresh[:])

#%%md
#50/50 train/test split version
#%%



#%%md
*OOS results*
#%%md
Confusion matrix
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
Precision-Recall curve
#%%
plot(prcurve(pdf.(yhat_logit_p,1), y_test_int))
plot(prcurve(pdf.(yhat_svm_p,1), y_test_int))
plot(prcurve(pdf.(yhat_nn_p,1), y_test_int))

#%%md
*Discussion of Results*
Comment on your results:
#%%md
*Which model performs the best?*


#%%md
*What are the main differences of each model?*
Logit:


SVM:


Neural Network:
