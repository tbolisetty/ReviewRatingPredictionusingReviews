#h2o usage for RosmannSales
#after feature Engineering
#Xgboost for Rossman Sales

library(data.table)  
library(xgboost)
library(plyr)

cat("reading the train and test data (with data.table) \n")
train <- fread("train.csv",stringsAsFactors = T)
test  <- fread("test.csv",stringsAsFactors = T)
store <- fread("store.csv",stringsAsFactors = T)

## more care should be taken to ensure the dates of test can be projected from train
## decision trees do not project well, so you will want to have some strategy here, if using the dates
train[,Date:=as.Date(Date)]
test[,Date:=as.Date(Date)]

# seperating out the elements of the date column for the train set
train[,month:=as.integer(format(Date, "%m"))]
train[,year:=as.integer(format(Date, "%y"))]

test[,month:=as.integer(format(Date, "%m"))]
test[,year:=as.integer(format(Date, "%y"))]


# impute Competition Values 
store$CompetitionOpenSinceYear[is.na(store$CompetitionOpenSinceYear)] <- 1990 # Dealing with NA and outlayers
store$CompetitionOpenSinceMonth[is.na(store$CompetitionOpenSinceMonth)] <- 1 # Dealing with NA
store$CompetitionDistance[is.na(store$CompetitionDistance)] <- 75000 # Dealing with NA

store$CompetitionStrength <- cut(store$CompetitionDistance, breaks=c(0, 1500, 6000, 12000, 20000, Inf), labels=FALSE) # 15 min, 1/2/3 hours (or walking and 10/20/30 min driving)

store$SundayStore <- as.numeric(store$Store %in% unique(train$Store[train$DayOfWeek==7 & train$Open==1])) #is this a Sunday-store

store$Promo2SinceWeek[is.na(store$Promo2SinceWeek)]<-0
store$Promo2SinceYear[is.na(store$Promo2SinceYear)]<-0


store[,StoreType:=as.factor(StoreType)]
store[,Assortment:=as.factor(Assortment)]

#Expanding the promoInterval
sample_matrix<-getPromoMatrix(store$PromoInterval)
store<-cbind(store,as.data.frame(sample_matrix))

#store_matrix<-model.matrix(~.-1,data=store)
#Generating store features depending upon the mean sales
custdata<-ddply(train[Open==1 & Sales!=0,],c('Store'),function(x)
  c(mean_customers=mean(x$Customers),
    mean_Sales=mean(x$Sales))
)
store<-merge(store,custdata,by="Store")


train <- merge(train,store,by="Store")
test <- merge(test,store,by="Store")


#considering only the stores from test set
#storeList<-unique(test$Store)
#rowids<-which(train$Store %in% storeList)
#train<-train[rowids,]

#taking only the needed records
train <- train[Sales > 0,]  ## We are not judged on 0 sales records in test set
## See Scripts discussion from 10/8 for more explanation.

# impute 11 days open for one store in the test dataset
test$Open[is.na(test$Open)] <- 1


#changing the sales parameter to log
train[,logSales:=log(Sales)]

#Converting the other features too
train$StateHoliday<-as.integer(train$StateHoliday)
test$StateHoliday<-as.integer(test$StateHoliday)

train$SchoolHoliday<-as.integer(train$SchoolHoliday)
test$SchoolHoliday<-as.integer(test$SchoolHoliday)

train$StoreType<-as.integer(train$StoreType)
test$StoreType<-as.integer(test$StoreType)

train$Assortment<-as.integer(train$Assortment)
test$Assortment<-as.integer(test$Assortment)

train$PromoInterval<-as.integer(train$PromoInterval)
test$PromoInterval<-as.integer(test$PromoInterval)

train$weekid<-as.integer(format(as.POSIXct(train$Date), "%U"))
test$weekid<-as.integer(format(as.POSIXct(test$Date), "%U"))

firstNday<-as.integer(format(train$Date,"%d"))
firstNday[firstNday<=3]<-1
firstNday[firstNday!=1]<-0
train$firstNday<-firstNday

firstNday<-as.integer(format(test$Date,"%d"))
firstNday[firstNday<=3]<-1
firstNday[firstNday!=1]<-0
test$firstNday<-firstNday

lastNday<-as.integer(format(train$Date,"%d"))
lastNday[lastNday>=28]<-1
lastNday[lastNday!=1]<-0
train$lastNday<-lastNday

lastNday<-as.integer(format(test$Date,"%d"))
lastNday[lastNday>=28]<-1
lastNday[lastNday!=1]<-0
test$lastNday<-lastNday

#Columns to include as features
features<-c(1,2,6:37,40,41)
response<-train[[38]]
featureNames<-colnames(train)[features]

#Identifying the potential periods that nearlt represent the test set
#working with dates in R
start_2014<-as.Date("2014-08-01", format="%Y-%m-%d")
end_2014<-as.Date("2014-09-17",format="%Y-%m-%d")

start_2013<-as.Date("2013-08-01", format="%Y-%m-%d")
end_2013<-as.Date("2013-09-17",format="%Y-%m-%d")

set1<-which(train$Date>=start_2014&train$Date<=end_2014)
set2<-which(train$Date>=start_2013&train$Date<=end_2013)
trainset<-c(set1,set2)
##########################################################
#seed setting
set.seed(999)
isolated_set<-setdiff(c(1:nrow(train)),trainset)
h1<-sample(isolated_set,size=0.5*length(isolated_set))
#Using stratified sampling for generating the samples
#     sample_train<-as.data.frame(train[isolated_set,c(1,2,26),with=FALSE])
#     sample_train<-stratified(sample_train,group=c("Store","DayOfWeek"),size=0.5)
#h1<-sample_train$rowid


h2<-sample(trainset,size=0.8*length(trainset))

h<-c(h1,h2)
#h<-c(h1,trainset)
#values that act as predicted set
h3<-setdiff(trainset,h2)

#Initializing h2o
set.seed(999)
## Start cluster with all available threads
h2o.init(nthreads=-1,max_mem_size='6G')
## Load data into cluster from R
trainHex<-as.h2o(train,destination_frame ="trainHex" )

trainHex[,1]<-as.numeric(trainHex[,1])
## Train a random forest using all default parameters
rfHex <- h2o.randomForest(x=featureNames,
                          y="logSales", 
                          model_id="rf",
                          ntrees = 150,
                          mtries = 10,
                          #max_depth = 30,
                          #nbins_cats = 1115, ## allow it to fit store ID
                          training_frame=trainHex[h,],
                          validation_frame = trainHex[h3,])

#checking the crossvalidation metrics
rfHex@model$validation_metrics

#Need to load the testset
cat("Predicting Sales\n")
## Load test data into cluster from R
testHex<-as.h2o(test[,featureNames,with=FALSE],destination_frame ="testHex" )
## Get predictions out; predicts in H2O, as.data.frame gets them into R
predictions<-as.data.frame(h2o.predict(rfHex,testHex))
## Return the predictions to the original scale of the Sales data
pred <- exp(predictions[,1])

#Understanding the summary ofpredictions
summary(pred)
submission <- data.frame(Id=test$Id, Sales=pred)
write.csv(submission, "h2o_rf_dependencytest.csv",row.names=F)



