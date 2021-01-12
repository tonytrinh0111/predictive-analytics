#######################################################################
#Predictive Analytics - Term 4 2020
#Time Series Predictions - Considerations and Methods
#R code ONLY
#######################################################################


library(forecast)
#Read in the data and classify it as a time series "ts" data type
#You can specify the start date and seasonal frequency of the data
bwl=ts(read.csv("RetailBWL.csv")[,2],start=c(1992,1),frequency=12)
#Plot the time series data and modify the title and label accordingly
plot(bwl,main="Retail Sales - Beer, Wine and Liqour",
     ylab="Millions of US Dollars")


#The stl() function for decomposition takes in two inputs
#s.window controls the smoothness of the seasonal component
#t.window controls the smoothness of the trend component
stl_bwl=stl(bwl,s.window=9,t.window=11)
#construct the seasonally adjusted series 
bwl_sadj=seasadj(stl_bwl)
plot(bwl_sadj,main="Seasonally Adjusted Retail Sales - Beer, Wine,Liqour",
     ylab="Millions of US Dollars")


#Plot the decomposition of the BWL data can also be useful
plot(stl_bwl)
#Plotting the trend of each seasonal component reveal a different angle
monthplot(stl_bwl)
plot(bwl_sadj)


#Data split
bwl_train=window(bwl,end=c(2010,12))
bwl_test=window(bwl,start=c(2011,1))

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


#Fit the linear regression with trend and seasonality 
#Construct a summary - this assume seasonal factor is fixed and the growthh is fixed, which fail to address the increasement trend of season ==> address this by using log
reg_bwl=tslm(bwl_train~trend+season)
summary(reg_bwl)
#Construct the long-horizon prediction into the test data set
#This is over 113 periods beyond the test set data
#Plot the prediction along with the test set data
reg_fcast=forecast(reg_bwl,h=113)
plot(reg_fcast,ylim=c(min(bwl),max(bwl)))
lines(bwl_test,col="red")
legend("topleft", legend=c("Actual","Predicted"),
       col=c("red","blue"), lty=1, cex=0.8,text.font=4)


#Fit the model with log of the time series
lreg_bwl=tslm(log(bwl_train)~trend+season)
summary(lreg_bwl)
#Predict and plot
lreg_fcast=forecast(lreg_bwl,h=113)
#Remember that you constructed the prediction in log terms!
plot(lreg_fcast)
#So you will need to convert the predictions back using the exp() function
plot(bwl,col="gray",main="Long Horizon Predictions - Log Regression",
     ylab="Millions of US Dollars")
lines(exp(lreg_fcast$mean),col="blue")
lines(bwl_test,col="red")
legend("topleft", legend=c("Actual","Predicted"),
       col=c("red","blue"), lty=1, cex=0.8,text.font=4)


#Fitting the ETS model to training data 
#Specification will be chosen automatically using a designated model selection criteria
ets_bwl=ets(bwl_train)
summary(ets_bwl)
#Prediction still makes use of the forecast() function
ets_fcast=forecast(ets_bwl,h=113)
plot(ets_fcast)
lines(bwl_test,col="red")
legend("topleft", legend=c("Actual","Predicted"),
       col=c("red","blue"), lty=1, cex=0.8,text.font=4)


#Fit the ARIMA model to the training data
#The auto.arima() function searches for the best combination of (p,d,q)(P,D,Q) order
arima_bwl=auto.arima(bwl_train)
arima_bwl
#The forecast() function can be used to generate ARIMA predictions
arima_fcast=forecast(arima_bwl,h=113)
plot(arima_fcast)
lines(bwl_test,col="red")
legend("topleft", legend=c("Actual","Predicted"),
       col=c("red","blue"), lty=1, cex=0.8,text.font=4)


#Calculate and collate predictive accuracy metrics
stats_long=rbind(tsaccuracy(bwl_test,reg_fcast$mean),
                 tsaccuracy(bwl_test,exp(lreg_fcast$mean)),
                 tsaccuracy(bwl_test,ets_fcast$mean),
                 tsaccuracy(bwl_test,arima_fcast$mean))
rownames(stats_long)=c("Regression", "Log Regression","Exp Smoothing", "ARIMA")
stats_long

##Real time predictions
#Storage spaces
ntrain=length(bwl_train)
ntest=length(bwl_test)
training=ts(bwl[1:ntrain],start=c(1992,1),frequency=12)
reg_yhat=matrix(0,ntest,1)
lreg_yhat=matrix(0,ntest,1)
ets_yhat=matrix(0,ntest,1)
arima_yhat=matrix(0,ntest,1)

##This loop is a real time updating loop
##It can take a little while to run
for(i in 1:ntest){
  #Estimate the models
  reg_bwl=tslm(training~trend+season)
  lreg_bwl=tslm(log(training)~trend+season)
  ets_bwl=ets(training,model="MAM",damped=TRUE)
  arima_bwl=Arima(training,order=c(2,1,0),seasonal=c(0,1,2))
  
  #Predict one period ahead
  reg_f=forecast(reg_bwl,h=1)
  lreg_f=forecast(lreg_bwl,h=1)
  ets_f=forecast(ets_bwl,h=1)
  arima_f=forecast(arima_bwl,h=1)
  
  #Save the predictions
  reg_yhat[i]=reg_f$mean[1]
  lreg_yhat[i]=exp(lreg_f$mean[1])
  ets_yhat[i]=ets_f$mean[1]
  arima_yhat[i]=arima_f$mean[1]
  
  #Expand the training set for the next iteration
  training=ts(bwl[1:(ntrain+i)],start=c(1992,1),frequency=12)
}

#Plot the forecasts
par(mfrow=c(2,2))
plot(bwl,col="gray",main="Real Time Predictions - Time Series Regression",
     ylab="Millions of US Dollars",cex.main=0.8)
lines(ts(reg_yhat,start=c(2011,1),frequency=12),col="blue")
plot(bwl,col="gray",main="Real Time Predictions - Time Series Log Regression",
     ylab="Millions of US Dollars",cex.main=0.8)
lines(ts(lreg_yhat,start=c(2011,1),frequency=12),col="blue")
plot(bwl,col="gray",main="Real Time Predictions - Exponential Smoothing",
     ylab="Millions of US Dollars",cex.main=0.8)
lines(ts(ets_yhat,start=c(2011,1),frequency=12),col="blue")
plot(bwl,col="gray",main="Real Time Predictions - ARIMA",
     ylab="Millions of US Dollars",cex.main=0.8)
lines(ts(arima_yhat,start=c(2011,1),frequency=12),col="blue")

#A summary of the real time predictive accuracy metrics
stats_short=rbind(tsaccuracy(bwl_test,reg_yhat),
                  tsaccuracy(bwl_test,lreg_yhat),
                  tsaccuracy(bwl_test,ets_yhat),
                  tsaccuracy(bwl_test,arima_yhat))
rownames(stats_short)=c("Regression", "Log Regression","Exp Smoothing", "ARIMA")
stats_short