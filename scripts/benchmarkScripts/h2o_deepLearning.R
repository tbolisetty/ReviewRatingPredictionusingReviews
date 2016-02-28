library(h2o)

#dataset processing 
data<-read.csv(file="checkdata.csv",header=TRUE,sep=",")
#summary(data)
#last record filled as NA
#data<-data[-125974,]

#feature normalization
y<-data$V253
preProcessModel <- preProcess(data[,-c(1,2)], method = c("center", "scale"))
data<-predict(preProcessModel,data[,-c(1,2)])
data$V253<-y

features<-colnames(data)[1:250]

set.seed(999)
## Start cluster with all available threads
h2o.init(nthreads=-1,max_mem_size='4G')
## Load data into cluster from R
trainHex<-as.h2o(data,destination_frame = "trainHex")

deep_learn<-h2o.deeplearning(x=features,y="V253",training_frame = trainHex,model_id="DL_v1",
                             hidden = c(10,10),nfolds=5,epochs = 10)

summary(deep_learn)
# checking the error produced by the deeplearning  
deep_learn@model$cross_validation_metrics