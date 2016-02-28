#Loading Xgboost
library(xgboost)

#reading the dataset
data<-read.csv(file="checkdata.csv",header=TRUE,sep=",")
#summary(data)
#last record filled as NA
#data<-data[-125974,]

#Preparing the target variable for xgboost classifier
y=data$X253.col
y<-y-1
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 5,
              "eta"=0.1,
              "max.depth"=10,
              "nthread" = 2,
              "min_child_weight"=1,
              "colsample_bytree"=0.8,
              "subsample"=0.8
)
xgb.cv(params=param,nrounds=10,data=as.matrix(data[,-c(1,2,253)]),nfold=5,label=y,
       metrics={'mlogloss'},
       verbose=TRUE,showsd=FALSE,
       maximize=TRUE)





