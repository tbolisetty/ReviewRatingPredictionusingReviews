library(caret)
library(h2o)
#reading the dataset
data<-read.table(file="features_250.txt",header=FALSE,sep=",")
#summary(data)
#last record filled as NA
#data<-data[-125974,]

#checking 
y=data$X253.col
featureNames<-colnames(data)[-c(1,2,253)]

h2o.init(nthreads=-1,max_mem_size='6G')
## Load data into cluster from R
trainHex<-as.h2o(data,destination_frame ="trainHex" )
rfHex <- h2o.randomForest(x=featureNames,
                          y="X253.col", 
                          model_id="rf",
                          ntrees = 30,
                          mtries = 70,
                          #max_depth = 30,
                          #nbins_cats = 1115, ## allow it to fit store ID
                          training_frame=trainHex,
                          nfolds=5)

#checking the crossvalidation metrics
rfHex@model$cross_validation_metrics

h2o.shutdown()

system.time()