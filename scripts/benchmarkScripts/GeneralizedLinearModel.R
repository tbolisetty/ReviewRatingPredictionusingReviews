#Running GLM(generalizedLinearModel)

library(caret)
#reading the dataset
data<-read.table(file="features_250.txt",header=FALSE,sep=",")
summary(data)
#last record filled as NA
#data<-data[-125974,]

y<-data$V253
preProcessModel <- preProcess(data[,-c(1,2)], method = c("center", "scale"))
data<-predict(preProcessModel,data[,-c(1,2)])
data$V253<-y
                            
fitControl <- trainControl(method = "cv",
                           ## Estimate class probabilities
                           classProbs = FALSE,
                           number=5,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = defaultSummary)
glmFit<-train(V253~ ., data = data,
            method = "glm",
            trControl = fitControl,
            
            ## Specify which metric to optimize
            metric ="RMSE")

print(glmFit)
