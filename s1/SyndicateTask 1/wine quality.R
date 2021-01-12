# humna factor: individual or panel 
# data factor: not enough of features being capture & not enough data point?
# algo factor: not flexible enough algo

# Upload necessary packages
library(tidyverse)
library(Metrics)

#import redwine
df=read.csv('redquality.csv',stringsAsFactors = TRUE)

# explore data
head(df)
str(df)

#Question1a avg of each variable
dfsumm=summary(df)
dfsumm
dfmean=dfsumm[4,]
dfmean
# write.csv(dfmean, file = "mean.csv")
#Prepare graphs
par(mfrow=c(1,2))

#Question1b Histogram of the quality score
df_hist=hist(df$QS,breaks=5,main="Histogram of Quality Score", xlab="Quality Score")
text(df_hist$mids,df_hist$counts,labels=df_hist$counts,adj=c(0.5,-0.5))

#check unique
df_uniq=unique(df$QS)
df_uniq

#Question1c Relationship between chemicals
df_cor=cor(df)
df_cor

round(df_cor,digits=3)
cor_qs=df_cor[,"QS"]
cor_qs

write.csv(round(cor_qs,digits=3), file = "correlation.csv")

# Set the random seed and sample the index of the training set data.
set.seed(123)
train_index <- sample(1:nrow(df), 0.8 * nrow(df))

# Split data into training and testing set 
trainData <- df[train_index,]
testData <- df[-train_index,]

#############################################################################
#Fit the regression model for Price on all available predictors
fit1 <- lm(QS~., data = trainData)
fit1Summary = summary(fit1)
fit1Summary

#############################################################################
#Load the MASS library and use the stepAIC() function for stepwise regression
library(MASS)
fit2 <- stepAIC(fit1, direction="both", trace = FALSE)
fit2Summary = summary(fit2)
fit2Summary

par("mar")
par(mar=c(2,2,2,2))
######### Residual FIT 1 #########
par(mfrow=c(3,4))

plot(x=trainData$FA,y=fit1$residuals,xlab="FA",ylab="Residuals",
     main="Residuals vs FA")
abline(a=0,b=0,col="red")

plot(x=trainData$VA,y=fit1$residuals,xlab="VA",ylab="Residuals",
     main="Residuals vs VA")
abline(a=0,b=0,col="red")

plot(x=trainData$CA,y=fit1$residuals,xlab="CA",ylab="Residuals",
     main="Residuals vs CA")
abline(a=0,b=0,col="red")

plot(x=trainData$RS,y=fit1$residuals,xlab="RS",ylab="Residuals",
     main="Residuals vs RS")
abline(a=0,b=0,col="red")

plot(x=trainData$Ch,y=fit1$residuals,xlab="Ch",ylab="Residuals",
     main="Residuals vs Ch")
abline(a=0,b=0,col="red")

plot(x=trainData$FSD,y=fit1$residuals,xlab="FSD",ylab="Residuals",
     main="Residuals vs FSD")
abline(a=0,b=0,col="red")

plot(x=trainData$TSD,y=fit1$residuals,xlab="TSD",ylab="Residuals",
     main="Residuals vs TSD")
abline(a=0,b=0,col="red")

plot(x=trainData$Density,y=fit1$residuals,xlab="Density",ylab="Residuals",
     main="Residuals vs Density")
abline(a=0,b=0,col="red")

plot(x=trainData$pH,y=fit1$residuals,xlab="pH",ylab="Residuals",
     main="Residuals vs pH")
abline(a=0,b=0,col="red")

plot(x=trainData$Sulphates,y=fit1$residuals,xlab="Sulphates",ylab="Residuals",
     main="Residuals vs Sulphates")
abline(a=0,b=0,col="red")

plot(x=trainData$Alc,y=fit1$residuals,xlab="Alc",ylab="Residuals",
     main="Residuals vs Alc")
abline(a=0,b=0,col="red")

plot(x=trainData$QS,y=fit1$fitted.values,xlab="Actual",ylab="Predicted",
     main="Predicted vs Actual")
abline(a=0,b=1,col="red")


######### Residual FIT 2 #########
#Scatter of residuals vs fitted values and residuals vs VA
par(mfrow=c(2,4))

plot(x=trainData$VA,y=fit2$residuals,xlab="VA",ylab="Residuals",
     main="Residuals vs VA")
abline(a=0,b=0,col="red")

plot(x=trainData$Ch,y=fit2$residuals,xlab="Ch",ylab="Residuals",
     main="Residuals vs Ch")
abline(a=0,b=0,col="red")

plot(x=trainData$FSD,y=fit2$residuals,xlab="FSD",ylab="Residuals",
     main="Residuals vs FSD")
abline(a=0,b=0,col="red")

plot(x=trainData$TSD,y=fit2$residuals,xlab="TSD",ylab="Residuals",
     main="Residuals vs TSD")
abline(a=0,b=0,col="red")

plot(x=trainData$pH,y=fit2$residuals,xlab="pH",ylab="Residuals",
     main="Residuals vs pH")
abline(a=0,b=0,col="red")

plot(x=trainData$Sulphates,y=fit2$residuals,xlab="Sulphates",ylab="Residuals",
     main="Residuals vs Sulphates")
abline(a=0,b=0,col="red")

plot(x=trainData$Alc,y=fit2$residuals,xlab="Alc",ylab="Residuals",
     main="Residuals vs Alc")
abline(a=0,b=0,col="red")

plot(x=trainData$QS,y=fit2$fitted.values,xlab="Actual",ylab="Predicted",
     main="Predicted vs Actual")
abline(a=0,b=1,col="red")

#############################################################################
#Fit the nonlinear regression model with quadratics and interactions
#Fit the nonlinear regression model with quadratics and interactions
fit3 <- lm(QS~ VA+ I(VA^2) + I(log(VA))+ Ch + I(Ch^2) + I(log(Ch)) + TSD + I(TSD^2) + I(log(TSD)) + FSD + I(FSD^2) + I(log(FSD)) + pH + Sulphates + I(Sulphates^2) + I(log(Sulphates)) + Alc, data = trainData)
summary(fit3)

fit4 <- stepAIC(fit3, direction="both", trace = FALSE)
summary(fit4)

# logreg <- glm(log(QS)~ VA+ I(VA^2) + I(log(VA))+ Ch + I(Ch^2) + I(log(Ch)) + TSD + I(TSD^2) + I(log(TSD)) + FSD + I(FSD^2) + I(log(FSD)) + pH + Sulphates + I(Sulphates^2) + I(log(Sulphates)) + Alc, data = trainData, family = binomial)
# summary(logreg)


######### Residual FIT 3 #########
par(mfrow=c(2,4))
plot(x=trainData$VA,y=fit3$residuals,xlab="VA",ylab="Residuals",
     main="Residuals vs VA")
abline(a=0,b=0,col="red")

plot(x=trainData$Ch,y=fit3$residuals,xlab="Ch",ylab="Residuals",
     main="Residuals vs Ch")
abline(a=0,b=0,col="red")

plot(x=trainData$FSD,y=fit3$residuals,xlab="FSD",ylab="Residuals",
     main="Residuals vs FSD")
abline(a=0,b=0,col="red")

plot(x=trainData$TSD,y=fit3$residuals,xlab="TSD",ylab="Residuals",
     main="Residuals vs TSD")
abline(a=0,b=0,col="red")

plot(x=trainData$pH,y=fit3$residuals,xlab="pH",ylab="Residuals",
     main="Residuals vs pH")
abline(a=0,b=0,col="red")

plot(x=trainData$Sulphates,y=fit3$residuals,xlab="Sulphates",ylab="Residuals",
     main="Residuals vs Sulphates")
abline(a=0,b=0,col="red")

plot(x=trainData$Alc,y=fit3$residuals,xlab="Alc",ylab="Residuals",
     main="Residuals vs Alc")
abline(a=0,b=0,col="red")

plot(x=trainData$QS,y=fit3$fitted.values,xlab="Actual",ylab="Predicted",
     main="Predicted vs Actual")
abline(a=0,b=1,col="red")


######### Residual FIT 4 #########
par(mfrow=c(2,4))
plot(x=trainData$VA,y=fit4$residuals,xlab="VA",ylab="Residuals",
     main="Residuals vs VA")
abline(a=0,b=0,col="red")

plot(x=trainData$Ch,y=fit4$residuals,xlab="Ch",ylab="Residuals",
     main="Residuals vs Ch")
abline(a=0,b=0,col="red")

plot(x=trainData$FSD,y=fit4$residuals,xlab="FSD",ylab="Residuals",
     main="Residuals vs FSD")
abline(a=0,b=0,col="red")

plot(x=trainData$TSD,y=fit4$residuals,xlab="TSD",ylab="Residuals",
     main="Residuals vs TSD")
abline(a=0,b=0,col="red")

plot(x=trainData$pH,y=fit4$residuals,xlab="pH",ylab="Residuals",
     main="Residuals vs pH")
abline(a=0,b=0,col="red")

plot(x=trainData$Sulphates,y=fit4$residuals,xlab="Sulphates",ylab="Residuals",
     main="Residuals vs Sulphates")
abline(a=0,b=0,col="red")

plot(x=trainData$Alc,y=fit4$residuals,xlab="Alc",ylab="Residuals",
     main="Residuals vs Alc")
abline(a=0,b=0,col="red")

plot(x=trainData$QS,y=fit4$fitted.values,xlab="Actual",ylab="Predicted",
     main="Predicted vs Actual")
abline(a=0,b=1,col="red")




######## Residual  - Non linear ########

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

# Predicting y with prediction interval using test set
pfit1 <- predict(fit1, newdata = testData, interval = "prediction") 
head(pfit1)

# Predicting y with prediction interval using test set
pfit2 <- predict(fit2, newdata = testData, interval = "prediction") 
head(pfit2)

# Predicting y with prediction interval using test set
pfit3 <- predict(fit3, newdata = testData, interval = "prediction") 
head(pfit3)

# Predicting y with prediction interval using test set
pfit4 <- predict(fit4, newdata = testData, interval = "prediction") 
head(pfit4)

# Assessing model - collate the predictive accuracy metrics - Question 3
stats1=csaccuracy(testData[,"QS"],pfit1[, "fit"],mean(trainData[,"QS"]))
stats1

# Assessing model 
stats2=csaccuracy(testData[,"QS"],pfit2[, "fit"],mean(trainData[,"QS"]))
stats2

# Assessing model
stats3=csaccuracy(testData[,"QS"],pfit3[, "fit"],mean(trainData[,"QS"]))
stats3

# Assessing model
stats4=csaccuracy(testData[,"QS"],pfit4[, "fit"],mean(trainData[,"QS"]))
stats4

#############################################################################
#Collate summary statistics from the three models
sumstats=rbind(stats1, stats2, stats3, stats4)
rownames(sumstats) <- c("Linear","Stepwise","Nonlinear", "Nonlinear plus stepwise")
sumstats
write.csv(sumstats, file = "sumstats.csv")
save.image('WineWorkspace.RData')

