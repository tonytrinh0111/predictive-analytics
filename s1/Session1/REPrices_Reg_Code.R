#######################################################################
#Predictive Analytics - Term 4 2020
#Prediction with Regression - Real Estate Prices
#R code ONLY
#######################################################################

# Upload necessary packages
library(tidyverse)
library(Metrics)

# Reading the data from a .csv file
data <- read.csv("RE.csv", stringsAsFactors = TRUE)
str(data)

# Convert categorical variables to dummy variables 0/1 coding
# For the usage of model which do not accept factor variables such as neural net, boosting
catVar <- model.matrix(~SchoolZone+Auction+New+Subdivided, data = data)[,-1]

# Final data - ready fo ruse
realestate <- cbind(data[1:5], catVar)
str(realestate)

# Set the random seed and sample the index of the training set data.
set.seed(123)
train_index <- sample(1:nrow(realestate), 0.8 * nrow(realestate))

# Split data into training and testing set 
trainData <- realestate[train_index,]
testData <- realestate[-train_index,]

#############################################################################
#Fit the regression model for Price on all available predictors
fit1 <- lm(Price~., data = trainData)
summary(fit1)

# Predicting y with prediction interval using test set -  other values of interval are confidence, neural net cannot give you an interval - even a statiscial insignificance factor can increase model perfomance
pfit1 <- predict(fit1, newdata = testData, interval="prediction")
head(pfit1)

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

# Assessing model - collate the predictive accuracy metrics 
stats1=csaccuracy(testData[,"Price"],pfit1[, "fit"],mean(trainData[,"Price"]))
stats1

#############################################################################
#Load the MASS library and use the stepAIC() function for stepwise regression - AIC, not p-value
library(MASS)
fit2 <- stepAIC(fit1, direction="both", trace = FALSE)
summary(fit2)

# Predicting y with prediction interval using test set
pfit2 <- predict(fit2, newdata = testData, interval = "prediction") 
head(pfit2)
# Assessing model 
stats2=csaccuracy(testData[,"Price"],pfit2[, "fit"],mean(trainData[,"Price"]))
stats2


#############################################################################
#Fit the nonlinear regression model with quadratics and interactions
#Fit the nonlinear regression model with quadratics and interactions
fit3 <- lm(Price~Beds+Baths+Cars+Area+I(Area^2)+SchoolZoneYes+NewYes+AuctionYes
           +SubdividedYes+I(NewYes*Area)+I(NewYes*(Area^2)), data = trainData)
summary(fit3)

# Predicting y with prediction interval using test set
pfit3 <- predict(fit3, newdata = testData, interval = "prediction") 
head(pfit3)
# Assessing model
stats3=csaccuracy(testData[,"Price"],pfit3[, "fit"],mean(trainData[,"Price"]))
stats3

#############################################################################
#Collate summary statistics from the three models
sumstats=rbind(stats1, stats2, stats3)
rownames(sumstats) <- c("Linear","Stepwise","Nonlinear")
sumstats

#############################################################################
#The banker's loss
#First define a function for the banker's loss then calculate the loss for each model
banker<-function(error) 
{
  loss=sum(2*abs(error[(error<=0)]))+sum(abs(error[(error>0)]))
  return(loss/length(error))
}

bankerloss=rbind(Linear=banker(testData[,"Price"]-pfit1[,"fit"]),
                 Stepwise=banker(testData[,"Price"]-pfit2[,"fit"]),
                 Nonlinear=banker(testData[,"Price"]-pfit3[,"fit"]))
bankerloss

#############################################################################
#First define a function for the agent's loss then calculate the loss for each model - prediction error VS model error VS error
agent<-function(error,actual){
  relative=100*(error/actual)
  low=relative[(abs(relative)<=10)]
  med=relative[(abs(relative)>10)&(abs(relative)<=30)]
  hi=relative[(abs(relative)>30)]
  loss=sum(abs(low))+sum(5*abs(med))+sum(10*abs(hi))
  return(loss/length(error))
}
agentloss=rbind(Linear=agent(testData[,"Price"]-pfit1[,"fit"],testData[,"Price"]),
                Stepwise=agent(testData[,"Price"]-pfit2[,"fit"],testData[,"Price"]),
                Nonlinear=agent(testData[,"Price"]-pfit3[,"fit"],testData[,"Price"]))
agentloss

#Let's collate all statistics
allstats=cbind(sumstats,bankerloss,agentloss)
colnames(allstats)[5:6] <- c("Banker","REAgent")
allstats

#Save the workspace for future use
save.image('REWorkspace.RData')
