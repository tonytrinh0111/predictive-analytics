#Load saved workspace from Session 3
load('WineWorkspaceSession3.RData')

###Bagging###
# Load caret & vip library
library(caret)
library(vip)
library(doParallel)
registerDoParallel(cores = detectCores() - 1)

bagStats = rbind(c('RMSE','MAE','MAPE','MASE','WineMakerLoss'))
bagNum = 20
while (bagNum <= 200) {
  # Bagging for regression tree
  set.seed(1234)
  # Training the bagging algorithm
  bag <- train(QS~., data = trainData, method='treebag',
               nbagg=bagNum) #Control the number of bags here
  # Predicting test data using trained bag
  y_bag <- predict(bag, newdata = testData)
  
  # Calculate and compile predictive accuracy metrics
  stats_bag = cbind(csaccuracy(testData[,"QS"],y_bag,mean(trainData[,"QS"])),
                    wine_maker_loss_function(actualQS-y_bag,actualQS))

  rownames(stats_bag)[1] =sprintf("Bag_%d",bagNum)
  
  bagStats=rbind(bagStats,stats_bag)
  bagNum = bagNum + 20
}
bagStats
write.csv(bagStats, file = "bagStats.csv")


###Random Forest###
# Load randomForest library
library(randomForest)
# Set random seet to make sure results remain static
#Correction of typo for "ntree", and addition of "nodesize" input
#You can also input the option of "maxnode" if you wish 
#"maxnode" should be greater than "nodesize"

RFStats = rbind(c('RMSE','MAE','MAPE','MASE','WineMakerLoss'))
treeNum = 100
while (treeNum <= 1000) {
  tryNum = 2
  while (tryNum <= 11) {
    nodeNum = 5
    while(nodeNum <= 20){
      set.seed(1234)
      randFor = randomForest(QS ~ ., data = trainData, importance = TRUE, 
                             ntree = treeNum, mtry = tryNum, nodesize=nodeNum)

      y_rf = predict(randFor, newdata = testData)
      stats_rf = cbind(csaccuracy(testData[,"QS"],y_rf,mean(trainData[,"QS"])),
                       wine_maker_loss_function(actualQS-y_rf,actualQS))
      rownames(stats_rf)[1] =sprintf("RF_%d_%d_%d",treeNum, tryNum,nodeNum)
      
      RFStats=rbind(RFStats,stats_rf)
      nodeNum = nodeNum + 5
    }

    tryNum = tryNum + 1
  }

  treeNum = treeNum + 100
}
RFStats
write.csv(RFStats, file = "RFStats.csv")



###Boosting###
#Load the mboost library
library(mboost)

boostStats = rbind(c('RMSE','MAE','MAPE','MASE','WineMakerLoss'))
nuNum = 0.01
while (nuNum <= 1) {
  #Set random seed and train the boosting algorithm
  set.seed(1234)
  boost <- glmboost(QS~., data = trainData,
                    control=boost_control(nu=nuNum))
  
  # Predicting test data using the trained boosting algorithm
  y_boost <- predict(boost, newdata = testData)
  
  # Calculate and compile predictive accuracy metrics
  stats_boost <- cbind(csaccuracy(testData[,"QS"],y_boost,mean(trainData[,"QS"])),
                       wine_maker_loss_function(actualQS-y_boost[,1],actualQS))
  
  rownames(stats_boost)[1] =sprintf("Boost_%f",nuNum)
  
  boostStats=rbind(boostStats,stats_boost)
  nuNum = nuNum + 0.01
}
boostStats
write.csv(boostStats, file = "boostStats.csv")

set.seed(1234)
boost_005 <- glmboost(QS~., data = trainData,
                  control=boost_control(nu=0.05))
set.seed(1234)
boost_010 <- glmboost(QS~., data = trainData,
                      control=boost_control(nu=0.10))

summary(boost_005)
summary(boost_010)

###Support Vector Machine###
# load the svm library
library(e1071)
svmStats = rbind(c('RMSE','MAE','MAPE','MASE','WineMakerLoss'))

kernelValue <- c("linear","polynomial","radial","sigmoid")
for (val in kernelValue) {
  svmm <- svm(QS~., data=trainData, kernel = val)
  # Predict the test data using trained svm
  y_svmm <- predict(svmm, newdata=testData)
  # Calculate and compile predictive accuracy metrics
  stats_svm <- cbind(csaccuracy(testData[,"QS"],y_svmm,mean(trainData[,"QS"])),
                     wine_maker_loss_function(actualQS-y_svmm,actualQS))
  rownames(stats_svm)[1] =sprintf("SVM_%s",val)
  svmStats=rbind(svmStats,stats_svm)
  
}
svmStats
write.csv(svmStats, file = "svmStats.csv")

#Save the workspace
save.image("WineWorkspaceSession4.RData")
