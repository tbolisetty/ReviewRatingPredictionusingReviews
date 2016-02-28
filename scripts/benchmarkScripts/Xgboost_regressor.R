#Xgboost Regressor
#Loading Xgboost
library(xgboost)



#reading the dataset
#data<-read.table(file="features_250.txt",header=FALSE,sep=",",)
data<-read.csv(file="C:\D\BigData\Project\frame\frame1.csv",header=FALSE)
summary(data)
#last record filled as NA
#data<-data[-125974,]

#Preparing the target variable for xgboost classifier
y=data$X253.col
param <- list("objective" = "reg:linear",
              "booster" = "gbtree",
              "eta"=0.05,
              "max.depth"=10,
              "nthread" = 2,
              "min_child_weight"=1,
              "colsample_bytree"=0.8,
              "subsample"=0.8
)
xgb.cv(params=param,nrounds=100,data=as.matrix(data[,-c(1,2,253)]),nfold=5,label=y,
       metrics={'rmse'},
       verbose=TRUE,showsd=FALSE,
       maximize=TRUE)