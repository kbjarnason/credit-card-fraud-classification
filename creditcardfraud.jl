#%% md
By Kristian Bjarnason

#%%
#Relevant packages
using Pkg, Revise, DataFrames, CSV, LinearAlgebra, Dates, Statistics, MLJ, MLJBase, MLJModels, MLJLinearModels, Plots, Flux, EvalCurves, UrlDownload, MLBase, StatsBase
using Flux:outdims, activations, @epochs
using Flux.Data
# using AUC # add git@github.com:paulstey/AUC.jl.git

#%%md
*Data Preparation*
Divide the sample into two equal sub-samples. Keep the proportion of frauds the same in each sub-sample (246 frauds in each). Use one sub-sample to estimate (train) your models and the second one to evaluate the out-of-sample performance of each model.
#%%
#Set drive and import data
# cd("/Users/kristianbjarnason/Documents/Programming/Julia/creditcard")
# data = CSV.read("credicard.csv")
data = DataFrame(urldownload("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"))

#delete time column
select!(data, Not(:Time))

#log data column as it covers a large range. Also add 1e-6 so no values are 0 prior to being logged.
data[!,:Amount] = log.(data[!,:Amount] .+ 1e-6)

data_fraud = filter(row -> row[:Class] == 1, data)
data_notfraud = filter(row -> row[:Class] == 0, data)

#split into training and test data, training with 1 extra row due to odd numbber of non-fraudulent claims and each with even number of fraudulent claims
data_train = vcat(data_fraud[1:round(Int, nrow(data_fraud)/2),:], data_notfraud[1:round(Int, nrow(data_notfraud)/2),:])
data_test = vcat(data_fraud[round(Int, nrow(data_fraud)/2)+1:nrow(data_fraud),:], data_notfraud[round(Int, nrow(data_notfraud)/2)+1:nrow(data_notfraud),:])

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
#initial logit classification with lambda = 1.0
@load LogisticClassifier pkg=MLJLinearModels
model_logit = LogisticClassifier(lambda=1.0)

logit = machine(model_logit, X_train, y_train)

fit!(logit)

yhat_logit_p = MLJBase.predict(logit, X_test)
yhat_logit = MLJBase.predict_mode(logit, X_test)

misclassification_rate(yhat_logit, y_test)

CSV.write("yhat_logit.csv", yhat_logit)
CSV.write("yhat_logit_p.csv", yhat_logit_p)

#%%
#tuned logit
model_logit = @load LogisticClassifier pkg=MLJLinearModels
r = range(model_logit, :lambda, lower=1e-6, upper=100, scale=:log)

self_tuning_logit_model = TunedModel(model=model_logit,
                                                  resampling = CV(nfolds=3),
                                                  tuning = Grid(resolution=10),
                                                  range = r,
                                                  measure = cross_entropy)

self_tuning_logit = machine(self_tuning_logit_model, X_train, y_train)

fit!(self_tuning_logit)

yhat_logit_tuned_p = MLJBase.predict(self_tuning_logit, X_test)
yhat_logit_tuned = categorical(mode.(yhat_logit_tuned_p))

misclassification_rate(yhat_logit_tuned, y_test)

CSV.write("yhat_logit_tuned.csv", yhat_logit_tuned)
CSV.write("yhat_logit_tuned_p.csv", yhat_logit_tuned_p)

#%%md
Support Vector Machine
#%%
#standardise data for SVM
stand_model = Standardizer()

X_train_std = MLJModels.transform(fit!(machine(stand_model, X_train)), X_train) # not used
X_test_std = MLJModels.transform(fit!(machine(stand_model, X_test)), X_test)

#%%
#initial svm classification with cost = 1.0
@load SVC
model_svm = @pipeline Std_SVC(std_model = Standardizer(),
                                      svc = SVC())

svc = machine(model_svm, X_train, y_train)

fit!(svc)

yhat_svm = MLJBase.predict(svc, X_test_std)

misclassification_rate(yhat_svm, y_test)
cm_svm = confusion_matrix(yhat_svm, y_test)

CSV.write("yhat_svm.csv", yhat_svm)

#%%
@load SVC

model_svm = @pipeline Std_SVC(std_model = Standardizer(),
                              svc = SVC())

r = range(model_svm, :(svc.cost), lower=0.0, upper=5.0, scale=:linear)
iterator(r,6)
svc = machine(model_svm, X_train, y_train)

self_tuning_svm_model = TunedModel(model=model_svm,
                                   resampling = CV(nfolds=3),
                                   tuning = Grid(resolution=6),
                                   range = r,
                                   measure = misclassification_rate)

self_tuning_svm = machine(self_tuning_svm_model, X_train, y_train)

fit!(self_tuning_svm)

fitted_params(self_tuning_svm).best_model

report(self_tuning_svm)

yhat_svm_tuned = MLJBase.predict(self_tuning_svm, X_test_std)

misclassification_rate(yhat_svm_tuned, y_test)
cm_svm = confusion_matrix(yhat_svm_tuned, y_test)

CSV.write("yhat_svm_tuned.csv", yhat_svm_tuned)

#%%md
Neural Network
#%%
#NN implementation
data1 = DataLoader(Array(X_train)', y_train_int, batchsize=128)

n_inputs = ncol(X_train)
n_outputs = 1
n_hidden1 = 16

m = Chain(
          Dense(n_inputs, n_hidden1, relu),
          Dropout(0.5),
          Dense(n_hidden1, n_outputs, σ)
          )

loss(x, y) = Flux.crossentropy(m(x), y)
ps = Flux.params(m)
opt = ADAM()

@epochs 20 Flux.train!(loss, ps, data1, opt) #giving highly variable/not always good results?

yhat_nn_p = vec(m(Array(X_test)'))
yhat_nn = categorical(Int.(yhat_nn_p .<= 0.5))

misclassification_rate(yhat_nn, y_test)

#%%md
*OOS results*
#%%
#if needed can reload from here
yhat_logit_tuned = CSV.read("yhat_logit_tuned.csv")
yhat_svm_tuned = CSV.read("yhat_svm_tuned.csv")
yhat_nn = CSV.read("yhat_nn.csv")

#%%md
Misclassification rate
#%%
misclassification_rate(yhat_logit_tuned, y_test)
misclassification_rate(yhat_svm_tuned, y_test)
misclassification_rate(yhat_nn, y_test)

#%%md
Confusion matrix
#%%
cm_logit = confusion_matrix(yhat_logit_tuned,y_test)
cm_svm = confusion_matrix(yhat_svm_tuned ,y_test)
cm_nn = confusion_matrix(yhat_nn,y_test)

#%%md
ROC curves
#%%
plot(roc_curve(yhat_logit_tuned_p,y_test))
# don't have score vectors for SVM
# plot(roc_curve(yhat_svm_p,y_test))
plot(roc_curve(yhat_nn_p,y_test))

#%%md
Precision-Recall curve
#%%
plot(prcurve(pdf.(yhat_logit_tuned_p,1), y_test_int))
# don't have score vectors for SVM
# plot(prcurve(pdf.(yhat_svm_p,1), y_test_int))
plot(prcurve(yhat_nn_p, y_test_int))

#%%md
*Discussion of Results*
Comment on your results:
#%%md
#TODO fix ordering
*Which model performs the best?*

Misclassification Ranking
1. NN
2. Logit
3. SVM

Confusion Matrices
1. NN
2. Logit
3. SVM

ROC Curves
1. NN
2. Logit

Precision-Recall curve
1. NN
2. Logit

#%%
#Couldn't get working
#Adapted from the tutorial: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
#thanks to Michael Griffiths for his help putting this together too. https://pastebin.com/iwtCFN3F

# The scale of the columns is quite variable -- we need to normalize each column
# The steps are to log the Amount and then standard scale everything (only use training data for this)
n_rows = size(data, 1)
train_idx = sample(1:n_rows, Int(floor(n_rows * .7)))
remainder = setdiff(1:n_rows, train_idx)
validation_idx = sample(remainder, Int(floor(size(remainder, 1) * .2)))
test_idx = setdiff(remainder, validation_idx)

# Remove time column (index 1) and log amount column (index 30)
train = data[train_idx,:]
test = data[test_idx,:]
valid = data[validation_idx,:]

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
valid_y = validation[:,:Class]

progress() = sum(loss.(valid_X, valid_y))
ps = Flux.params(m)
opt = ADAM()

# Custom training loop
for batch in 1:400
    batch_data = sample(Array(training[1:29])', 32)
    Flux.train!(loss, ps, batch_data, opt) #TODO crashes here...
    println(progress())
end

# Evaluate on test data
y_pred = [x[1] for x in m.([d[1] for d in testing])]
y_test = [d[2] for d in testing]

y_thresh = (y_pred[:] .> .5) .+ 1
confusmat(2, y_test[:] .+ 1, y_thresh[:])
