wine=read.csv('redquality.csv')

wine$QS<-as.factor(wine$QS>=7)
levels(wine$QS)<-c("No","Yes")
colnames(wine)[12]="GoodQS"

set.seed(123)
train_index <- sample(1:nrow(wine), 0.8 * nrow(wine))

# Split data into training and testing set 
trainData <- wine[train_index,]
testData <- wine[-train_index,]

# Check the data definition for classification
contrasts(testData$GoodQS)

#Function to calculate statistical accuracy metrics of binary classification
#INPUTS: y=test data "No" or "Yes"
#prob=predicted probability of a "Yes" for each row of test data
#cutoff=cutoff probability for predicting the classification
# e.g. predicted outcome = Yes if prob>cutoff
#showplot=TRUE/FALSE indicating whether a plot of the ROC curve should be produced
labelAccuracy <- function(y,prob,cutoff,showplot){
  #Hit and miss table
  HM <- matrix(0,2,2)
  colnames(HM) <- c("Actual y='Yes'", "Actual y='No'")
  rownames(HM) <- c("Predicted y='Yes'", "Predicted y='No'")
  HM[1,1] <- mean(prob>cutoff & y=="Yes", na.rm=TRUE)
  HM[1,2] <- mean(prob>cutoff & y=="No", na.rm=TRUE)
  HM[2,1] <- mean(prob<=cutoff & y=="Yes", na.rm=TRUE)
  HM[2,2] <- mean(prob<=cutoff & y=="No", na.rm=TRUE)
  
  #Storage for various accuracy measures
  metrics=matrix(0,1,6)
  colnames(metrics)=c("Overall","Precision","Sensitivity",
                      "Specificity","AUC","Gini")
  metrics[1,"Overall"]=sum(diag(HM))
  metrics[1,"Precision"]=HM[1,1]/sum(HM[1,])
  metrics[1,"Sensitivity"]=HM[1,1]/sum(HM[,1])
  metrics[1,"Specificity"]=HM[2,2]/sum(HM[,2])
  
  #Construct ROC to calculate AUC and Gini
  #Show plot only if input showplot==TRUE
  #These metrics do not vary with the cutoff level
  # ROC curve
  fit.ROC <- roc(y~prob, plot=showplot, print.auc=TRUE,quiet=TRUE)
  metrics[1,"AUC"]=fit.ROC$auc
  metrics[1,"Gini"]=2*fit.ROC$auc-1
  
  UserFunc=(HM[2,1]*15+HM[1,2]*2)*100
  
  
  #Function returns a list, including the hit and miss table
  #the summary metrics and the ROC curve object
  return(list(HitMiss=HM,Metrics=metrics,ROC=fit.ROC,UDF=UserFunc))
}


#The Logistic model
#Train the logistic regression using the glm() function
fit1 <- glm(GoodQS~., data = trainData, family=binomial)
#A summary of the fit
summary(fit1)
#Summarizing the variable importance graphically
vip(fit1)

# Predicting y for the test set
pfit1 <- predict(fit1, newdata = testData, type="response")


# Assessing model accuracy with cutoff threshold of 0.5
fit1Acc=labelAccuracy(testData$GoodQS,pfit1,0.5,TRUE)
fit1Acc$Metrics

##Decision tree
library(rpart)
library(rpart.plot)

# Building the decision tree for classification
fit2 <- rpart(GoodQS ~ ., data = trainData, method  = "class")

# Visualizing the tree
rpart.plot(fit2)
vip(fit2)

# Predicting y for the test set
pfit2 <- predict(fit2, newdata=testData, type="prob")

# Assessing model accuracy
fit2Acc=labelAccuracy(testData$GoodQS,pfit2[,"Yes"],0.5,TRUE)
fit2Acc$Metrics

wine_scaled <- wine
for (i in 1:11){
  wine_scaled[,i] <- as.numeric(scale(wine_scaled[,i]))
}

# Split data
trainData_scaled <- wine_scaled[train_index,]
testData_scaled <- wine_scaled[-train_index,]

##Neural network
library(neuralnet)
library(NeuralNetTools)
#Set the random seed and train the neural network
set.seed(123)
fit3 <- neuralnet(GoodQS~., data=trainData_scaled, hidden=4,
                  linear.output = FALSE)

#Olden's variable importance
fit3olden=olden(fit3,out_var='Yes')
fit3olden$data

# Predicting y for the test set
pfit3 <- predict(fit3, newdata=testData_scaled)
# Assessing model accuracy
fit3Acc=labelAccuracy(testData$GoodQS,pfit3[,2],0.5,TRUE)
fit3Acc$Metrics
fit3Acc$HitMiss

##K-Nearest Neighbour classification
#Train the k-NN classification here. 
#The knn3 function is from the caret library, and it returns predicted probabilities for all classes within the dataset.
fit4 <- knn3(GoodQS~., data=trainData_scaled,
             k=as.integer(sqrt(nrow(trainData_scaled))))

# Predicting y for the test set
pfit4 <- predict(fit4, newdata=testData_scaled, type="prob")
# Assessing model accuracy
fit4Acc=labelAccuracy(testData$GoodQS,pfit4[,"Yes"],0.5,TRUE)
fit4Acc$Metrics

##Bagging
#Set the random seed and train the bagging algorithm
set.seed(123)
fit5 <- train(GoodQS~., data = trainData, method='treebag',
              metric="Accuracy")
print(fit5)
vip(fit5)

# Predicting y for the test set
pfit5 <- predict(fit5, newdata=testData, type="prob")
# Assessing model accuracy
fit5Acc=labelAccuracy(testData$GoodQS,pfit5[,"Yes"],0.5,TRUE)
fit5Acc$Metrics

##Random Forest
library(randomForest)
#Set the random seed and train the random forest algorithm
set.seed(123)
fit6 <- randomForest(GoodQS~., data=trainData, importance=TRUE,
                     ntrees=50, mtry=2)
print(fit6)
vip(fit6)

# Predicting y for the test set
pfit6 <- predict(fit6, newdata=testData, type="prob")
# Assessing model accuracy
fit6Acc=labelAccuracy(testData$GoodQS,pfit6[,"Yes"],0.5,TRUE)
fit6Acc$Metrics

##Boosting
library(mboost)
#Set random seed and train he goosing algorithm
#Base model is the logistic regression
set.seed(123)
fit7 <- glmboost(GoodQS~., data=trainData_scaled, family=Binomial(),
                 control=boost_control(nu=0.1),center = TRUE)
summary(fit7)
plot(varimp(fit7))

# Predicting y for the test set
pfit7 <- predict(fit7, newdata=testData_scaled, type="response")
# Assessing model accuracy
fit7Acc=labelAccuracy(testData$GoodQS,pfit7[,1],0.5,TRUE)
fit7Acc$Metrics

##Support Vector Machine
library(e1071)
fit8 <- svm(GoodQS~., data=trainData_scaled,
            type="C-classification", kernel="linear",
            probability=TRUE,
            scale=FALSE) # use scale false to turn off scaling as dummy variable shouldn't be scaled and use manual scaled data instead
summary(fit8)
#Predicting for the test set and extracting predicted probabilities
pfit8 <- predict(fit8, newdata=testData_scaled, probability=TRUE)
pfit8 <- attr(pfit8, "probabilities")
# Assessing model accuracy
fit8Acc=labelAccuracy(testData_scaled$GoodQS,pfit8[,"Yes"],0.5,TRUE)
fit8Acc$Metrics



#Set up a grid value of the cutoff
cutoff=seq(from=0.01,to=0.99,by=0.01)

logRegLoss=vector(mode="numeric",length(cutoff))
for(i in 1:length(cutoff)){
  acc=labelAccuracy(testData$GoodQS,pfit1,cutoff[i],FALSE)
  logRegLoss[i]=acc$UDF
}
plot(x=cutoff,y=logRegLoss,xlab="Cutoff Probability", ylab="Log Reg Dollar Loss",
     main="Dollar Loss per Hundred Bottles vs Cutoff probabilities")
cutoff[which.min(logRegLoss)]
min(logRegLoss)
acc=labelAccuracy(testData$GoodQS,pfit1,cutoff[which.min(logRegLoss)],FALSE)
acc$Metrics

decisionTreeLoss=vector(mode="numeric",length(cutoff))
for(i in 1:length(cutoff)){
  acc=labelAccuracy(testData$GoodQS,pfit2[,"Yes"],cutoff[i],FALSE)
  decisionTreeLoss[i]=acc$UDF
}
plot(x=cutoff,y=decisionTreeLoss,xlab="Cutoff Probability", ylab="Decision Tree Dollar Loss",
     main="Dollar Loss per Hundred Bottles vs Cutoff probabilities")
cutoff[which.min(decisionTreeLoss)]
min(decisionTreeLoss)
acc=labelAccuracy(testData$GoodQS,pfit2[,"Yes"],cutoff[which.min(decisionTreeLoss)],FALSE)
acc$Metrics

neuralNetLoss=vector(mode="numeric",length(cutoff))
for(i in 1:length(cutoff)){
  acc=labelAccuracy(testData$GoodQS,pfit3[,2],cutoff[i],FALSE)
  neuralNetLoss[i]=acc$UDF
}
plot(x=cutoff,y=neuralNetLoss,xlab="Cutoff Probability", ylab="Neural Net Dollar Loss",
     main="Dollar Loss per Hundred Bottles vs Cutoff probabilities")
acc=labelAccuracy(testData$GoodQS,pfit3[,2],cutoff[which.min(neuralNetLoss)],FALSE)
c(acc$Metrics,cutoff[which.min(neuralNetLoss)],min(neuralNetLoss))

kNNLoss=vector(mode="numeric",length(cutoff))
for(i in 1:length(cutoff)){
  acc=labelAccuracy(testData$GoodQS,pfit4[,"Yes"],cutoff[i],FALSE)
  kNNLoss[i]=acc$UDF
}
plot(x=cutoff,y=kNNLoss,xlab="Cutoff Probability", ylab="kNN Dollar Loss",
     main="Dollar Loss per Hundred Bottles vs Cutoff probabilities")
acc=labelAccuracy(testData$GoodQS,pfit4[,"Yes"],cutoff[which.min(kNNLoss)],FALSE)
c(acc$Metrics,cutoff[which.min(kNNLoss)],min(kNNLoss))


baggingLoss=vector(mode="numeric",length(cutoff))
for(i in 1:length(cutoff)){
  acc=labelAccuracy(testData$GoodQS,pfit5[,"Yes"],cutoff[i],FALSE)
  baggingLoss[i]=acc$UDF
}
plot(x=cutoff,y=baggingLoss,xlab="Cutoff Probability", ylab="Bagging Dollar Loss",
     main="Dollar Loss per Hundred Bottles vs Cutoff probabilities")
acc=labelAccuracy(testData$GoodQS,pfit5[,"Yes"],cutoff[which.min(baggingLoss)],FALSE)
c(acc$Metrics,cutoff[which.min(baggingLoss)],min(baggingLoss))


RFLoss=vector(mode="numeric",length(cutoff))
for(i in 1:length(cutoff)){
  acc=labelAccuracy(testData$GoodQS,pfit6[,"Yes"],cutoff[i],FALSE)
  RFLoss[i]=acc$UDF
}
plot(x=cutoff,y=RFLoss,xlab="Cutoff Probability", ylab="RF Dollar Loss",
     main="Dollar Loss per Hundred Bottles vs Cutoff probabilities")
acc=labelAccuracy(testData$GoodQS,pfit6[,"Yes"],cutoff[which.min(RFLoss)],FALSE)
c(acc$Metrics,cutoff[which.min(RFLoss)],min(RFLoss))

boostingLoss=vector(mode="numeric",length(cutoff))
for(i in 1:length(cutoff)){
  acc=labelAccuracy(testData$GoodQS,pfit7[,1],cutoff[i],FALSE)
  boostingLoss[i]=acc$UDF
}
plot(x=cutoff,y=boostingLoss,xlab="Cutoff Probability", ylab="Boosting Dollar Loss",
     main="Dollar Loss per Hundred Bottles vs Cutoff probabilities")
acc=labelAccuracy(testData$GoodQS,pfit7[,1],cutoff[which.min(boostingLoss)],FALSE)
c(acc$Metrics,cutoff[which.min(boostingLoss)],min(boostingLoss))

svmLoss=vector(mode="numeric",length(cutoff))
for(i in 1:length(cutoff)){
  acc=labelAccuracy(testData$GoodQS,pfit8[,"Yes"],cutoff[i],FALSE)
  svmLoss[i]=acc$UDF
}
plot(x=cutoff,y=svmLoss,xlab="Cutoff Probability", ylab="SVM Dollar Loss",
     main="Dollar Loss per Hundred Bottles vs Cutoff probabilities")
acc=labelAccuracy(testData$GoodQS,pfit8[,"Yes"],cutoff[which.min(svmLoss)],FALSE)
c(acc$Metrics,cutoff[which.min(svmLoss)],min(svmLoss))

##############Run regression using same config#############
wine4Reg=read.csv('redquality.csv')

wine4Reg_scaled <- wine4Reg
for (i in 1:12){
  wine4Reg_scaled[,i] <- as.numeric(scale(wine4Reg_scaled[,i]))
}

# Split data into training and testing set 
trainData4Reg <- wine4Reg[train_index,]
testData4Reg <- wine4Reg[-train_index,]


# Split data
trainData4Reg_scaled <- wine4Reg_scaled[train_index,]
testData4Reg_scaled <- wine4Reg_scaled[-train_index,]

labelAccuracy4Reg <- function(y,predicted,showplot){
  
  y=as.factor(y>=7)
  levels(y)<-c("No","Yes")
  predicted=as.factor(predicted>=7)
  levels(predicted)<-c("No","Yes")
  #Hit and miss table
  HM <- matrix(0,2,2)
  colnames(HM) <- c("Actual y='Yes'", "Actual y='No'")
  rownames(HM) <- c("Predicted y='Yes'", "Predicted y='No'")
  HM[1,1] <- mean(predicted=="Yes" & y=="Yes", na.rm=TRUE)
  HM[1,2] <- mean(predicted=="Yes" & y=="No", na.rm=TRUE)
  HM[2,1] <- mean(predicted=="No" & y=="Yes", na.rm=TRUE)
  HM[2,2] <- mean(predicted=="No" & y=="No", na.rm=TRUE)
  
  #Storage for various accuracy measures
  metrics=matrix(0,1,4)
  colnames(metrics)=c("Overall","Precision","Sensitivity",
                      "Specificity")
  metrics[1,"Overall"]=sum(diag(HM))
  metrics[1,"Precision"]=HM[1,1]/sum(HM[1,])
  metrics[1,"Sensitivity"]=HM[1,1]/sum(HM[,1])
  metrics[1,"Specificity"]=HM[2,2]/sum(HM[,2])
  
  
  UserFunc=(HM[2,1]*15+HM[1,2]*2)*100
  

  #Function returns a list, including the hit and miss table
  #the summary metrics 
  return(list(HitMiss=HM,Metrics=metrics,UDF=UserFunc))
}

########Neural Net#########
set.seed(123)
NN <- neuralnet(formula = QS ~., data=trainData4Reg_scaled, hidden = 4)
y_NN <- predict(NN, newdata = testData4Reg_scaled)
head(y_NN)
# convert predicted y into  original unit
y_NN <- y_NN*sd(trainData4Reg[,"QS"]) + mean(trainData4Reg[,"QS"])
head(y_NN)

acc=labelAccuracy4Reg(testData4Reg[,"QS"],y_NN,FALSE)
c(acc$Metrics,acc$UDF)

#########kNN#########
#Let's look at how well the knn regression fits the training data - use scaled data here
#Remember you need to remove the output variable to train k-nn (unsupervised)
knn_fit <- knnreg(x = trainData4Reg_scaled[, -12] , y = trainData4Reg_scaled[,"QS"] , 
                  k = as.integer(sqrt(nrow(trainData4Reg_scaled))))

#Don't forget to convert the predictions back to raw data unit when obtaining predictions
y_knnm <- predict(knn_fit,testData4Reg_scaled[,-12])
y_knnm<-y_knnm*sd(trainData4Reg[,"QS"])+mean(trainData4Reg[,"QS"])
head(y_knnm)

acc=labelAccuracy4Reg(testData4Reg[,"QS"],y_knnm,FALSE)
c(acc$Metrics,acc$UDF)

########Random Forest#########
set.seed(123)
RF_50_2_5 <- randomForest(QS~., data=trainData4Reg, importance=TRUE,
                     ntrees=50, mtry=2)
y_RF_50_2_5 <- predict(RF, newdata=testData4Reg)
acc=labelAccuracy4Reg(testData4Reg[,"QS"],y_RF_50_2_5,FALSE)
c(acc$Metrics,acc$UDF)

set.seed(123)
RF_200_9_5 <- randomForest(QS~., data=trainData4Reg, importance=TRUE,
                     ntrees=200, mtry=9)
y_RF_200_9_5 <- predict(RF_200_9_5, newdata=testData4Reg)
acc=labelAccuracy4Reg(testData4Reg[,"QS"],y_RF_200_9_5,FALSE)
c(acc$Metrics,acc$UDF)

set.seed(123)
RF_300_9_5 <- randomForest(QS~., data=trainData4Reg, importance=TRUE,
                     ntrees=300, mtry=9)
y_RF_300_9_5 <- predict(RF_300_9_5, newdata=testData4Reg)
acc=labelAccuracy4Reg(testData4Reg[,"QS"],y_RF_300_9_5,FALSE)
c(acc$Metrics,acc$UDF)

########Boosting#########
set.seed(123)
boost <- glmboost(QS~., data = trainData4Reg,
                  control=boost_control(nu=0.98))
y_boost <- predict(boost, newdata = testData4Reg)

acc=labelAccuracy4Reg(testData4Reg[,"QS"],y_boost,FALSE)
c(acc$Metrics,acc$UDF)



save.image("WineWorkspaceSession5.RData")
