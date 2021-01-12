###############Question 1##############
library(forecast)
#Read in the data and classify it as a time series "ts" data type
#You can specify the start date and seasonal frequency of the data
rmo=ts(read.csv("RetailEShopMail.csv")[,2],start=c(1992,1),frequency=12)
#Plot the time series data and modify the title and label accordingly
plot(rmo,main="Retail sales - Mail orders and online shopping",
     ylab="Millions of US Dollars")


#The stl() function for decomposition takes in two inputs
#s.window controls the smoothness of the seasonal component
#t.window controls the smoothness of the trend component
stl_rmo=stl(rmo,s.window=9,t.window=11)
#construct the seasonally adjusted series 
rmo_sadj=seasadj(stl_rmo)
plot(rmo_sadj,main="Seasonally Adjusted Retail Sales - Mail orders and online shopping",
     ylab="Millions of US Dollars")


#Plot the decomposition of the BWL data can also be useful
plot(stl_rmo)
#Plotting the trend of each seasonal component reveal a different angle
monthplot(stl_rmo)

###############Question 2##############
#Data split
rmo_train=window(rmo,end=c(2010,12))
rmo_test=window(rmo,start=c(2011,1))

###############Question 3##############
#Fit the linear regression with trend and seasonality 
#Construct a summary - this assume seasonal factor is fixed and the growthh is fixed, which fail to address the increasement trend of season ==> address this by using log
reg_rmo=tslm(rmo_train~trend+season)
summary(reg_rmo)

#Fit the model with log of the time series
lreg_rmo=tslm(log(rmo_train)~trend+season)
summary(lreg_rmo)

#Fitting the ETS model to training data 
#Specification will be chosen automatically using a designated model selection criteria
ets_rmo=ets(rmo_train)
summary(ets_rmo)

#Fit the ARIMA model to the training data
#The auto.arima() function searches for the best combination of (p,d,q)(P,D,Q) order
arima_rmo=auto.arima(rmo_train)
arima_rmo

###############Long horizon forecast##############
#Construct the long-horizon prediction into the test data set
#This is over 113 periods beyond the test set data
#Plot the prediction along with the test set data
reg_fcast=forecast(reg_rmo,h=113)
plot(reg_fcast,ylim=c(min(rmo),max(rmo)))
lines(rmo_test,col="red")
legend("topleft", legend=c("Actual","Predicted"),
       col=c("red","blue"), lty=1, cex=0.8,text.font=4)


#Predict and plot
lreg_fcast=forecast(lreg_rmo,h=113)
#Remember that you constructed the prediction in log terms!
plot(lreg_fcast)
#So you will need to convert the predictions back using the exp() function
plot(rmo,col="gray",main="Long Horizon Predictions - Log Regression",
     ylab="Millions of US Dollars")
lines(exp(lreg_fcast$mean),col="blue")
lines(rmo_test,col="red")
legend("topleft", legend=c("Actual","Predicted"),
       col=c("red","blue"), lty=1, cex=0.8,text.font=4)

#Prediction still makes use of the forecast() function
ets_fcast=forecast(ets_rmo,h=113)
plot(ets_fcast)
lines(rmo_test,col="red")
legend("topleft", legend=c("Actual","Predicted"),
       col=c("red","blue"), lty=1, cex=0.8,text.font=4)


#The forecast() function can be used to generate ARIMA predictions
arima_fcast=forecast(arima_rmo,h=113)
plot(arima_fcast)
lines(rmo_test,col="red")
legend("topleft", legend=c("Actual","Predicted"),
       col=c("red","blue"), lty=1, cex=0.8,text.font=4)


#Calculate and collate predictive accuracy metrics
stats_long=rbind(tsaccuracy(rmo_test,reg_fcast$mean),
                 tsaccuracy(rmo_test,exp(lreg_fcast$mean)),
                 tsaccuracy(rmo_test,ets_fcast$mean),
                 tsaccuracy(rmo_test,arima_fcast$mean))
rownames(stats_long)=c("Regression", "Log Regression","Exp Smoothing", "ARIMA")
stats_long
write.csv(stats_long, file = "stats_long.csv")


###############Question 4##############
##Real time predictions
#Storage spaces
ntrain=length(rmo_train)
ntest=length(rmo_test)
training=ts(rmo[1:ntrain],start=c(1992,1),frequency=12)
reg_yhat=matrix(0,ntest,1)
lreg_yhat=matrix(0,ntest,1)
ets_yhat=matrix(0,ntest,1)
arima_yhat=matrix(0,ntest,1)
##This loop is a real time updating loop
##It can take a little while to run

for(i in 1:ntest){
  #Estimate the models
  reg_rmo=tslm(training~trend+season)
  lreg_rmo=tslm(log(training)~trend+season)
  ets_rmo=ets(training,model="MAM",damped=TRUE)
  arima_rmo=Arima(training,order=c(2,1,0),seasonal=c(0,1,2))
  
  #Predict one period ahead
  reg_f=forecast(reg_rmo,h=1)
  lreg_f=forecast(lreg_rmo,h=1)
  ets_f=forecast(ets_rmo,h=1)
  arima_f=forecast(arima_rmo,h=1)
  
  #Save the predictions
  reg_yhat[i]=reg_f$mean[1]
  lreg_yhat[i]=exp(lreg_f$mean[1])
  ets_yhat[i]=ets_f$mean[1]
  arima_yhat[i]=arima_f$mean[1]
  
  #Expand the training set for the next iteration
  training=ts(rmo[1:(ntrain+i)],start=c(1992,1),frequency=12)
}


#Function for accuracy metrics in time series data
tsaccuracy <- function (actual,predict) {
  stats<-matrix(0,1,5)
  colnames(stats)<-c("RMSE","MAE","MAPE","MASE","ACF")
  m=frequency(actual)
  actual=as.numeric(actual)
  dev<-actual-predict
  naive=c(rep(NA,m),actual[1:(length(actual)-m)])
  q<-dev/mean(abs(actual-naive),na.rm=TRUE)
  stats[1]=sqrt(mean(dev^2))
  stats[2]=mean(abs(dev))
  stats[3]=mean(100*abs(dev/actual))
  stats[4]=mean(abs(q))
  stats[5]=acf(dev,plot=FALSE)$acf[2]
  return(stats)
}

#Plot the forecasts
par(mfrow=c(2,2))

plot(rmo,col="black",main="Real Time Predictions - Time Series Regression",
     ylab="Millions of US Dollars",cex.main=0.8)
lines(rmo_test,col="red")
lines(ts(reg_yhat,start=c(2011,1),frequency=12),col="blue")
legend("topleft", legend=c("Actual","Predicted"),
       col=c("red","blue"), lty=1, cex=0.8,text.font=4)

plot(rmo,col="black",main="Real Time Predictions - Time Series Log Regression",
     ylab="Millions of US Dollars",cex.main=0.8)
lines(ts(lreg_yhat,start=c(2011,1),frequency=12),col="blue")
lines(rmo_test,col="red")
legend("topleft", legend=c("Actual","Predicted"),
       col=c("red","blue"), lty=1, cex=0.8,text.font=4)

plot(rmo,col="black",main="Real Time Predictions - Exponential Smoothing",
     ylab="Millions of US Dollars",cex.main=0.8)
lines(ts(ets_yhat,start=c(2011,1),frequency=12),col="blue")
lines(rmo_test,col="red")
legend("topleft", legend=c("Actual","Predicted"),
       col=c("red","blue"), lty=1, cex=0.8,text.font=4)


plot(rmo,col="black",main="Real Time Predictions - ARIMA",
     ylab="Millions of US Dollars",cex.main=0.8)
lines(ts(arima_yhat,start=c(2011,1),frequency=12),col="blue")
lines(rmo_test,col="red")
legend("topleft", legend=c("Actual","Predicted"),
       col=c("red","blue"), lty=1, cex=0.8,text.font=4)


#A summary of the real time predictive accuracy metrics
stats_short=rbind(tsaccuracy(rmo_test,reg_yhat),
                  tsaccuracy(rmo_test,lreg_yhat),
                  tsaccuracy(rmo_test,ets_yhat),
                  tsaccuracy(rmo_test,arima_yhat))
rownames(stats_short)=c("Regression", "Log Regression","Exp Smoothing", "ARIMA")
stats_short
write.csv(stats_short, file = "stats_short.csv")



###############Question 5##############
bank_loss<-function(error,actual){
  relative=100*(error/actual)
  underestimate=relative[(relative>=0)]
  overU20=relative[ (relative<0) & (relative >= -20)]
  overTheTop=relative[(relative < -20)]
  loss=sum(5*underestimate) + sum(1*abs(overU20)) + sum(3*abs(overTheTop))
  return(loss/length(error))
}

bankLoss=rbind(Regression=bank_loss(rmo_test-reg_yhat,rmo_test),
                LogRegression=bank_loss(rmo_test-lreg_yhat,rmo_test),
                ExpSmoothing=bank_loss(rmo_test-ets_yhat,rmo_test),
                ARIMA=bank_loss(rmo_test-arima_yhat,rmo_test))

bankLoss

combinedMetrics = cbind(stats_short, bankLoss[,1])
colnames(combinedMetrics)[6] = "BankLoss"

combinedMetrics

write.csv(combinedMetrics, file = "combinedMetrics.csv")

save.image('RMOWorkspace.RData')

# actual = c(100,200,300)
# predicted = c(95,210,370)
# 
# error = actual - predicted
# error
# 
# relative=100*(error/actual)
# relative
# 
# underestimate=relative[(relative>=0)]
# underestimate
# 
# overU20=relative[ (relative<0) & (relative >= -20) ]
# overU20
# 
# overTheTop=relative[(relative<0) & (relative < -20)]
# overTheTop
# 
# loss=sum(5*underestimate)+sum(1*abs(overU20))+sum(3*abs(overTheTop))
# loss
# loss/3








