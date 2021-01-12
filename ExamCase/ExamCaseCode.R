###########################################
# Predictive Analytics Exam Case - Code File
# Term 4, 2020
# Associate Professor Ole Maneesoonthorn
###########################################

#Defining a function to compute the all metrics for cross sectional data
#Packages available to compute the metrics, but this function conveniently gathers all metrics
csaccuracy<-function(actual,predicted,naive){
  stats<-matrix(0,1,4)
  colnames(stats)<-c("RMSE","MAE","MAPE","MASE")
  dev<-actual-predicted
  BE=mean(abs(actual-naive))
  q=(dev)/BE
  stats[1]=sqrt(mean(dev^2))
  stats[2]=mean(abs(dev))
  stats[3]=mean(100*abs(dev/actual))
  stats[4]=mean(abs(q))
  return(stats)
}

#Loading in the data and sample split
load("CreditBal.RData")
set.seed(123)
train_index <- sample(1:nrow(Credit), 0.8 * nrow(Credit))
train=Credit[train_index,]
test=Credit[-train_index,]

##Predictive assessment with linear regression
fitlm=lm(Balance~.,data=train)
summary(fitlm)
predlm=predict(fitlm,newdata=test)
stats1=csaccuracy(test$Balance,predlm,mean(train$Balance))
stats1
fitlm

##Regression tree
# Load the libraries
# Note rpart.plot enables us to plot the tree visually
library(rpart)
library(rpart.plot)
library(vip)

# Building model
regtree <- rpart(formula = Balance ~ ., data = train, method  = "anova",
                 control=rpart.control(cp=0.01))

# Visualizing the tree
rpart.plot(regtree)

#Variable importance from the tree algorithm
vip(regtree)

# prediction
y_regtree<- predict(regtree, newdata = test)
stats2=csaccuracy(test$Balance,y_regtree,mean(train$Balance))
stats2
y_regtree
##Neural Network

library(neuralnet)
# Convert categorical variables to dummy variables 0/1 coding
# For the usage of model which do not accept factor variables
catVar <- model.matrix(~Gender+Student+Married, data = Credit)[,-1]
dat_scaled=Credit
dat_scaled[,5:7]=catVar
which=c(1:4,8)
for(i in 1:length(which)){
  dat_scaled[,which[i]]=as.numeric(scale(Credit[,which[i]]))
}
train_scaled=dat_scaled[train_index,]
test_scaled=dat_scaled[-train_index,]
set.seed(321)
NN <- neuralnet(formula = Balance ~., data=train_scaled, hidden = 5)

#Visualization of the neural network
library(NeuralNetTools)
plotnet(NN,cex_val=0.7)

#variable importance
#two different methods to look at them
garson(NN)
olden(NN)

#by clusters - group data into 3 clusters
set.seed(321)
lekprofile(NN,xsel=c('Income','Rating','Age','Education'),group_val=3)
#Reveal the characteristics of the clusters
set.seed(321)
lekprofile(NN,xsel=c('Income','Rating','Age','Education'),group_val=3,group_show=TRUE)


#Predicting with neural network
y_NN <- predict(NN, newdata = test_scaled)

# convert predicted y into  original unit
y_NN <- y_NN*sd(train[,"Balance"]) + mean(train[,"Balance"])
head(y_NN)
stats3=csaccuracy(test$Balance,y_NN,mean(train$Balance))
stats3

##K-Nearest Neighbour
#Loading the library
library(caret)
#Let's look at how well the knn regression fits the training data - use scaled data here
#Remember you need to remove the output variable to train k-nn (unsupervised)
knn_fit <- knnreg(x = train_scaled[, -8] , y = train_scaled[,"Balance"] , 
                  k = as.integer(sqrt(nrow(train))))

#Don't forget to convert the predictions back to raw data unit when obtaining predictions
y_knnm <- predict(knn_fit,test_scaled[,-8])
y_knnm<-y_knnm*sd(train[,"Balance"])+mean(train[,"Balance"])
stats4=csaccuracy(test$Balance,y_knnm,mean(train$Balance))
stats4

## Bagging
set.seed(1234)
# Training the bagging algorithm
bag <- train(Balance~., data = train, method='treebag')
print(bag)
# Plotting variable importance from the algorithm
vip(bag)

# Predicting test data using trained bag
y_bag <- predict(bag, newdata = test)
stats5=csaccuracy(test$Balance,y_bag,mean(train$Balance))
stats5

##Random Forest
# Load randomForest library
library(randomForest)
# Set random seet to make sure results remain static
set.seed(1234)
randFor = randomForest(Balance ~ ., data = train, importance = TRUE, 
                       mtry=3)
print(randFor)
# Plot variable importance
vip(randFor)

# Predicting test data using trained forest
y_rf<- predict(randFor, newdata = test)
stats6=csaccuracy(test$Balance,y_rf,mean(train$Balance))
stats6

##Boosting
#Load the mboost library
library(mboost)
#Set random seed and train the boosting algorithm
set.seed(1234)
boost <- glmboost(Balance~., data = train,
                  control=boost_control(nu=0.1))
summary(boost)
#Variable importance plot - mboost library has its own function
plot(varimp(boost))
varimp(boost)
# Predicting test data using the trained boosting algorithm
y_boost <- predict(boost, newdata = test)
stats7=csaccuracy(test$Balance,y_boost,mean(train$Balance))
stats7

##Support Vector Machine
# load the svm library
library(e1071)
svmm <- svm(Balance~., data=train, kernel = "linear")
svmm

#plotting svm fitted values against actual data
plot(x=svmm$fitted,y=train[,"Balance"],
     xlab="Fitted", ylab="Actual", main="SVM: actual vs fitted")

#plotting svm fitted values against actual data
plot(x=train$Rating,y=svmm$fitted,
     xlab="Ratings", ylab="Fitted", main="SVM: Ratings vs fitted")


# Predict the test data using trained svm
y_svmm <- predict(svmm, newdata=test)
stats8=csaccuracy(test$Balance,y_svmm,mean(train$Balance))
stats8

##Collating all statistics
allstats=rbind(stats1,stats2,stats3,stats4,
               stats5,stats6,stats7,stats8)
rownames(allstats)=c("Linear Regression","RegTree","NeuralNetwork",
                     "k-NNReg","Bagging","RandomForest",
                     "Boosting","SVM")
allstats

save.image('Exam_1st_round.RData')

cor(dat_scaled)
t.test(dat_scaled$Balance)
t.test(Credit$Balance)
