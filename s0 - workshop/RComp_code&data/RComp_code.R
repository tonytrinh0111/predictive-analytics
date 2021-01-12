################################################
#Inference and Linear Regression in R - code file
#Associate Professor Ole Maneesoonthorn
################################################

#Read the data from .csv file.
#Make sure that you have set the working directory to the location of the data file.
# use quotation, single or double for string, filename 
raw_data=read.csv('salary.csv')
#Compute gender specific average salary
mean_M=mean(raw_data[raw_data$Gender=="M",'salary'])
mean_F=mean(raw_data[raw_data$Gender=="F",'salary'])
mean_M
mean_F
#Compute the relative difference from simple average - in percentage
100*(mean_M-mean_F)/mean_F

#Conduct a t-test of mean difference between the two Genders
t.test(salary~Gender,data=raw_data)

#Fit a multiple linear regression
# R auto recognize gender as categorical and bin them; also the baseline will be chosen on alphabet (F in this case)
fit1=lm(salary~age+Expr+education+Gender, data=raw_data)
#Let us look at a summary of the model fit
summary(fit1)
#Compute the model implied differential - in percentage
100*fit1$coefficients["GenderM"]/mean_F

#A scatter plot of actual salary vs fitted values
plot(x=raw_data$salary,y=fit1$fitted.values,
     xlab="Actual",ylab="Fitted",main="Actual vs Fitted")
#Plot a linear line where intercept=0 and slope=1
abline(a=0,b=1,col="red") 

##Residual diagnostics
#Plot of residual histogram
hist(fit1$residuals,25)

#Scatter of residuals vs fitted values and residuals vs Expr
par(mfrow=c(2,2))
plot(x=raw_data$age,y=fit1$residuals,xlab="Age",ylab="Residuals",
     main="Residuals vs Age")
abline(a=0,b=0,col="red")
plot(x=raw_data$Expr,y=fit1$residuals,xlab="Expr",ylab="Residuals",
     main="Residuals vs Expr")
abline(a=0,b=0,col="red")
plot(x=raw_data$education,y=fit1$residuals,xlab="Education",ylab="Residuals",
     main="Residuals vs Education")
abline(a=0,b=0,col="red")

#Improving the model
#Reclassify education as a factor variable - this is ordinal or categorical data
raw_data$education=as.factor(raw_data$education)
#Fit a nonlinear regression model
fit2=lm(salary~age+I(age^2)+Expr+I(Expr^2)+education+Gender,data=raw_data)
#Improved model summary
summary(fit2)

#Plot of residual histogram
hist(fit2$residuals,25)

#Scatter of residuals vs fitted values and residuals vs Expr
par(mfrow=c(2,2))
plot(x=raw_data$age,y=fit2$residuals,xlab="Age",ylab="Residuals",
     main="Residuals vs Age")
abline(a=0,b=0,col="red")
plot(x=raw_data$Expr,y=fit2$residuals,xlab="Expr",ylab="Residuals",
     main="Residuals vs Expr")
abline(a=0,b=0,col="red")
plot(x=raw_data$education,y=fit2$residuals,xlab="Education",ylab="Residuals",
     main="Residuals vs Education")
abline(a=0,b=0,col="red")

#Model implied differential - in percentage
100*fit2$coefficients["GenderM"]/mean_F

#Characteristic profile by gender
#Age and experience profiles are captured by box plots
boxplot(age~Gender,data=raw_data,main="Age profile by gender")
boxplot(Expr~Gender,data=raw_data,main="Experience profile by gender")


#The relative frequency table is used to compare the education profile
FEduc=100*table(raw_data$education[raw_data$Gender=="F"])/sum(raw_data$Gender=="F")
MEduc=100*table(raw_data$education[raw_data$Gender=="M"])/sum(raw_data$Gender=="M")
FEduc
MEduc
