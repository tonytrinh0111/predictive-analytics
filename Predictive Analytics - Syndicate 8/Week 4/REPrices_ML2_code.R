#######################################################################
#Predictive Analytics - Term 4 2020
#Predictive Modelling with Machine Learning Techniques (2)
#R code ONLY
#######################################################################

#Load saved workspace from Session 3
load('REPriceSession3.RData')

###Bagging###
# Load caret & vip library
library(caret)
library(vip)

# Bagging for regression tree
set.seed(1234)
# Training the bagging algorithm
bag <- train(Price~., data = trainData, method='treebag',
             nbagg=100) #Control the number of bags here
print(bag)
# Plotting variable importance from the algorithm
vip(bag)

# Predicting test data using trained bag
y_bag <- predict(bag, newdata = testData)

# Calculate and compile predictive accuracy metrics
stats_bag <- cbind(csaccuracy(testData[,"Price"],y_bag,mean(trainData[,"Price"])),
                   Banker=banker(testData[,"Price"]-y_bag),
                   REAgent=agent(testData[,"Price"]-y_bag,testData[,"Price"]))

#Gathering predictive accuracy statistics
allstats=rbind(allstats[1:8,],stats_bag)
rownames(allstats)[9]='Bagging(Tree)'
allstats

###Random Forest###
# Load randomForest library
library(randomForest)
# Set random seet to make sure results remain static
set.seed(1234)
#Correction of typo for "ntree", and addition of "nodesize" input
#You can also input the option of "maxnode" if you wish 
#"maxnode" should be greater than "nodesize"
randFor = randomForest(Price ~ ., data = trainData, importance = TRUE, 
                       ntree = 500, mtry=2, nodesize=5)

print(randFor)
# Plot variable importance
vip(randFor)

# Predicting test data using trained forest
y_rf<- predict(randFor, newdata = testData)

# Calculate and compile predictive accuracy metrics
stats_rf <- cbind(csaccuracy(testData[,"Price"],y_rf,mean(trainData[,"Price"])),
                  Banker=banker(testData[,"Price"]-y_rf),
                  REAgent=agent(testData[,"Price"]-y_rf,testData[,"Price"]))

#Gathering predictive accuracy statistics
allstats=rbind(allstats[1:9,],stats_rf)
rownames(allstats)[10]='Random Forest'
allstats

###Boosting###
#Load the mboost library
library(mboost)
#Set random seed and train the boosting algorithm
set.seed(1234)
boost <- glmboost(Price~., data = trainData,
                  control=boost_control(nu=0.05))
summary(boost)
#Variable importance plot - mboost library has its own function
plot(varimp(boost))

# Predicting test data using the trained boosting algorithm
y_boost <- predict(boost, newdata = testData)

# Calculate and compile predictive accuracy metrics
stats_boost <- cbind(csaccuracy(testData[,"Price"],y_boost,mean(trainData[,"Price"])),
                     Banker=banker(testData[,"Price"]-y_boost),
                     REAgent=agent(testData[,"Price"]-y_boost,testData[,"Price"]))

#Gathering predictive accuracy statistics
allstats=rbind(allstats[1:10,],stats_boost)
rownames(allstats)[11]='Boosting'
allstats

###Support Vector Machine###
# load the svm library
library(e1071)
svmm <- svm(Price~., data=trainData, kernel = "linear")
svmm

#plotting svm fitted values against actual data
plot(x=svmm$fitted,y=trainData[,"Price"],
     xlab="Fitted", ylab="Actual", main="SVM: actual vs fitted")

# Predict the test data using trained svm
y_svmm <- predict(svmm, newdata=testData)

# Calculate and compile predictive accuracy metrics
stats_svm <- cbind(csaccuracy(testData[,"Price"],y_svmm,mean(trainData[,"Price"])),
                   Banker=banker(testData[,"Price"]-y_svmm),
                   REAgent=agent(testData[,"Price"]-y_svmm,testData[,"Price"]))

#Gathering predictive accuracy statistics
allstats=rbind(allstats[1:11,],stats_svm)
rownames(allstats)[12]='SVMReg'
allstats

#Save the workspace
save.image('REPriceSession4.RData')