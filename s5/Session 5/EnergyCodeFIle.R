#########################################
#Session 5 - Predicting Classification
#Energy consumer sentiment work through - code file
#Predictive Analytics, Term 4, 2020
#Associate Professor Ole Maneesoonthorn
#########################################


##Loading library
library(tidyverse)
library(caret)
# for ROC curve
library(pROC)
# for variable importance
library(vip)

##Loading data file
load("energyData.RData")

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
  
  #Function returns a list, including the hit and miss table
  #the summary metrics and the ROC curve object
  return(list(HitMiss=HM,Metrics=metrics,ROC=fit.ROC))
}

##Predictive assessment using December 2019 data
energy=dec19

set.seed(123)
train_index <- sample(1:nrow(energy), 0.8 * nrow(energy))

# Split data into training and testing set 
trainData <- energy[train_index,]
testData <- energy[-train_index,]

# Check the data definition for classification
contrasts(testData$Switch)

#Initial data transformation, etc..
energy_scaled <- energy

# change categorical variables to dummy
catVar <- model.matrix(~Employment+DecRole+
                         Gender+Age+AHInc+Educ+
                         DepChildren+OwnOrRent+BillLargerThanExp+
                         BillPressure+HouseType, data = energy_scaled)[,-1]

# scale continuous variables
for (i in 9:13){
  energy_scaled[,i] <- as.numeric(scale(energy_scaled[,i]))
}

# combining dummy and scaled variables
energy_scaled <- cbind(energy_scaled[, 9:13], catVar,
                       energy_scaled[,"Switch"])

# rename variables as the existing name is too long
names(energy_scaled)[] <- c("PeopleInHH", "VoMSat_Elec", "OverallCSE6mth",
                            "BillSatElec", "BlackoutSatElec", "EmpFT", "EmpPT", 
                            "EmpHome", "EmpRetired",
                            "EmpUnemployed", "EmpStudent", "JointDecMaker",
                            "Female","Age25_34","Age35_44", "Age45_54",
                            "Age55_64", "Age65_74","Age75", "AgeNoAns", "Income20k",
                            "Income40k", "Income60k", "Income80k", 
                            "Income100k", "Income120k", "Income150k",
                            "EducYear12", "EducTAFE", "EducDiploma", 
                            "EducUniversity", "DepChildrenY",
                            "OwnHomeY", "BillLargerThanExp", "BillPressureY",
                            "Townhouse","Unit","OtherHouseType",
                            "Switch")

# Split data
trainData_scaled <- energy_scaled[train_index,]
testData_scaled <- energy_scaled[-train_index,]

##Modelling probability of "Switch"

#The naive prediction....
pnaive=sum(trainData$Switch=='Yes')/nrow(trainData)

# Assessing model accuracy
naiveAcc=labelAccuracy(testData$Switch,replicate(nrow(testData),pnaive),0.5,TRUE)
naiveAcc$Metrics

#The Logistic model
#Train the logistic regression using the glm() function
fit1 <- glm(Switch~., data = trainData, family=binomial)
#A summary of the fit
summary(fit1)
#Summarizing the variable importance graphically
vip(fit1)

# Predicting y for the test set
pfit1 <- predict(fit1, newdata = testData, type="response")

# Assessing model accuracy with cutoff threshold of 0.5
fit1Acc=labelAccuracy(testData$Switch,pfit1,0.5,TRUE)
fit1Acc$Metrics


##Decision tree
library(rpart)
library(rpart.plot)

# Building the decision tree for classification
fit2 <- rpart(Switch ~ ., data = trainData, method  = "class")

# Visualizing the tree
rpart.plot(fit2)
vip(fit2)

# Predicting y for the test set
pfit2 <- predict(fit2, newdata=testData, type="prob")

# Assessing model accuracy
fit2Acc=labelAccuracy(testData$Switch,pfit2[,"Yes"],0.5,TRUE)
fit2Acc$Metrics

##Neural network
library(neuralnet)
library(NeuralNetTools)
#Set the random seed and train the neural network
set.seed(123)
fit3 <- neuralnet(Switch~., data=trainData_scaled, hidden=4,
                  linear.output = FALSE)

#Olden's variable importance
fit3olden=olden(fit3,out_var='Yes')
fit3olden$data

# Predicting y for the test set
pfit3 <- predict(fit3, newdata=testData_scaled)
# Assessing model accuracy
fit3Acc=labelAccuracy(testData$Switch,pfit3[,2],0.5,TRUE)
fit3Acc$Metrics

##K-Nearest Neighbour classification
#Train the k-NN classification here. 
#The knn3 function is from the caret library, and it returns predicted probabilities for all classes within the dataset.
fit4 <- knn3(Switch~., data=trainData_scaled,
             k=as.integer(sqrt(nrow(trainData))))

# Predicting y for the test set
pfit4 <- predict(fit4, newdata=testData_scaled, type="prob")
# Assessing model accuracy
fit4Acc=labelAccuracy(testData$Switch,pfit4[,"Yes"],0.5,TRUE)
fit4Acc$Metrics

##Bagging
#Set the random seed and train the bagging algorithm
set.seed(123)
fit5 <- train(Switch~., data = trainData, method='treebag',
              metric="Accuracy")
print(fit5)
vip(fit5)

# Predicting y for the test set
pfit5 <- predict(fit5, newdata=testData, type="prob")
# Assessing model accuracy
fit5Acc=labelAccuracy(testData$Switch,pfit5[,"Yes"],0.5,TRUE)
fit5Acc$Metrics

##Random Forest
library(randomForest)
#Set the random seed and train the random forest algorithm
set.seed(123)
fit6 <- randomForest(Switch~., data=trainData, importance=TRUE,
                     ntrees=50, mtry=2)
print(fit6)
vip(fit6)

# Predicting y for the test set
pfit6 <- predict(fit6, newdata=testData, type="prob")
# Assessing model accuracy
fit6Acc=labelAccuracy(testData$Switch,pfit6[,"Yes"],0.5,TRUE)
fit6Acc$Metrics

##Boosting
library(mboost)
#Set random seed and train he goosing algorithm
#Base model is the logistic regression
set.seed(123)
fit7 <- glmboost(Switch~., data=trainData_scaled, family=Binomial(),
                 control=boost_control(nu=0.1),center = TRUE)
summary(fit7)
plot(varimp(fit7))

# Predicting y for the test set
pfit7 <- predict(fit7, newdata=testData_scaled, type="response")
# Assessing model accuracy
fit7Acc=labelAccuracy(testData$Switch,pfit7[,1],0.5,TRUE)
fit7Acc$Metrics

##Support Vector Machine
library(e1071)
fit8 <- svm(Switch~., data=trainData_scaled,
            type="C-classification", kernel="linear",
            probability=TRUE,
            scale=FALSE) # use scale false to turn off scaling as dummy variable shouldn't be scaled and use manual scaled data instead
summary(fit8)

#Predicting for the test set and extracting predicted probabilities
pfit8 <- predict(fit8, newdata=testData_scaled, probability=TRUE)
pfit8 <- attr(pfit8, "probabilities")
# Assessing model accuracy
fit8Acc=labelAccuracy(testData$Switch,pfit8[,"Yes"],0.5,TRUE)
fit8Acc$Metrics

##Summarizing all predictive statistics
allstats=rbind(naiveAcc$Metrics,fit1Acc$Metrics,fit2Acc$Metrics,fit3Acc$Metrics,
               fit4Acc$Metrics,fit5Acc$Metrics,fit6Acc$Metrics,fit7Acc$Metrics,fit8Acc$Metrics)
rownames(allstats)=c("Naive","Logistic","DecisionTree","NeuralNet",
                     "k-NN","Bagging","RandomForest","Boosting","SVM")
allstats

##Assessing the optimal cutoff probability for logistic regression

#Set up a grid value of the cutoff
cutoff=seq(from=0.01,to=0.99,by=0.01)
overall=vector(mode="numeric",length(cutoff))
precision=vector(mode="numeric",length(cutoff))
sensitivity=vector(mode="numeric",length(cutoff))
specificity=vector(mode="numeric",length(cutoff))

for(i in 1:length(cutoff)){
  acc=labelAccuracy(testData$Switch,pfit1,cutoff[i],FALSE)
  overall[i]=acc$Metrics[,"Overall"]
  precision[i]=acc$Metrics[,"Precision"]
  sensitivity[i]=acc$Metrics[,"Sensitivity"]
  specificity[i]=acc$Metrics[,"Specificity"]
}

par(mfrow=c(2,2))
plot(x=cutoff,y=overall,xlab="Cutoff Probability", ylab="Overall Accuracy",
     main="Overall accuracy vs cutoff probabilities")
plot(x=cutoff,y=precision,xlab="Cutoff Probability", ylab="Precision",
     main="Precision vs cutoff probabilities")
plot(x=cutoff,y=sensitivity,xlab="Cutoff Probability", ylab="Sensitivity",
     main="Sensitivity vs cutoff probabilities")
plot(x=cutoff,y=specificity,xlab="Cutoff Probability", ylab="Specificity",
     main="Specificity vs cutoff probabilities")

#optimal cutoff for precision
cutoff[which.max(precision)]
#optimal cutoff for sensitivity
cutoff[which.max(sensitivity)]

##Out-of-time robustness using June 2020 data

#Constructing the scaled version of June 2020 data for testing
jun20_scaled <- jun20

# change categorical variables to dummy
catVar <- model.matrix(~Employment+DecRole+
                         Gender+Age+AHInc+Educ+
                         DepChildren+OwnOrRent+BillLargerThanExp+
                         BillPressure+HouseType, data = jun20_scaled)[,-1]

# scale continuous variables
for (i in 9:13){
  jun20_scaled[,i] <- as.numeric(scale(jun20_scaled[,i]))
}

# combining dummy and scaled variables
jun20_scaled <- cbind(jun20_scaled[, 9:13], catVar,
                      jun20_scaled[,"Switch"])

# rename variables as the existing name is too long
names(jun20_scaled)[] <- c("PeopleInHH", "VoMSat_Elec", "OverallCSE6mth",
                           "BillSatElec", "BlackoutSatElec", "EmpFT", "EmpPT", 
                           "EmpHome", "EmpRetired",
                           "EmpUnemployed", "EmpStudent", "JointDecMaker",
                           "Female","Age25_34","Age35_44", "Age45_54",
                           "Age55_64", "Age65_74","Age75", "AgeNoAns", "Income20k",
                           "Income40k", "Income60k", "Income80k", 
                           "Income100k", "Income120k", "Income150k",
                           "EducYear12", "EducTAFE", "EducDiploma", 
                           "EducUniversity", "DepChildrenY",
                           "OwnHomeY", "BillLargerThanExp", "BillPressureY",
                           "Townhouse","Unit","OtherHouseType",
                           "Switch")

#Retrain the neural network with the entire data set from December 2019
set.seed(456)
fit3Dec19=neuralnet(Switch~., data=energy_scaled, hidden=4,
                    linear.output = FALSE)
pfit3Jun20 <- predict(fit3Dec19, newdata=jun20_scaled)[,2]
# Assessing model accuracy
fit3AccJun20=labelAccuracy(jun20$Switch,pfit3Jun20,0.5,TRUE)

#Retraining the bagging algorithm using the entire data set from December 2019
set.seed(456)
fit5Dec19 <- train(Switch~., data = dec19, method='treebag',
                   metric="Accuracy")
pfit5Jun20 <- predict(fit5Dec19, newdata=jun20, type="prob")[,2]
# Assessing model accuracy
fit5AccJun20=labelAccuracy(jun20$Switch,pfit5Jun20,0.5,TRUE)

#Retraining the random forest using the entire data set from December 2019
set.seed(456)
fit6Dec19 <- randomForest(Switch~., data=dec19, importance=TRUE,
                          ntrees=50, mtry=2)
pfit6Jun20 <- predict(fit6Dec19, newdata=jun20, type="prob")[,2]
# Assessing model accuracy
fit6AccJun20=labelAccuracy(jun20$Switch,pfit6Jun20,0.5,TRUE)

accJun20=rbind(fit3AccJun20$Metrics,fit5AccJun20$Metrics,fit6AccJun20$Metrics)
rownames(accJun20)=c("NeuralNet","Bagging","RandomForest")
accJun20

##What has changed between December 2019 and June 2020
#Retrain the random forest with June 20 data
set.seed(123)
fit6Jun20<- randomForest(Switch~., data=jun20, importance=TRUE,
                         ntrees=50, mtry=2)

#Comparing the variable importance plots from Dec 19 and Jun 20
vip(fit6Dec19)
vip(fit6Jun20)