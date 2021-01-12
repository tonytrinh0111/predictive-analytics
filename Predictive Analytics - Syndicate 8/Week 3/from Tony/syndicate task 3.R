#######################################################################
#Predictive Analytics - Term 4 2020
#Predictive Modelling with Machine Learning Techniques (1)
#R code ONLY
#######################################################################

#Load saved workspace
load('WineWorkspace.RData')

##Regression Tree##
# Load the libraries
# Note rpart.plot enables us to plot the tree visually
library(rpart)
library(rpart.plot)
library(vip)

# Building model - anaova means looking at variance of price or variation explained. It simply mean the predictive accuracy is measured by RMSE and the purpose is to minimize it. Use GINI/AUC for classification
regtree <- rpart(formula = QS ~ ., data = trainData, method  = "anova")

# Visualizing the tree
rpart.plot(regtree)
#The summary command gives you detailed information at every level of the tree.
summary(regtree)

#Variable importance from the tree algorithm
vip(regtree)

#Pruning tree - complexity coefficient controls the size of the tree
ptree<-prune.rpart(regtree,cp=0.05)
rpart.plot(ptree)

# prediction
y_regtree<- predict(regtree, newdata = testData)
head(y_regtree)



# Assessing model
stats_regtree=cbind(csaccuracy(testData[,"QS"],y_regtree,mean(trainData[,"QS"])))
stats_regtree

##Neural Network##
library(neuralnet)

# scaled first five numeric data 
# do not scale the dummy variables
wine_scaled <- df
for(i in 1:12){
  wine_scaled[,i]=as.numeric(scale(df[,i]))
}

# creating training and test set using the same random data partition as before
trainData_scaled <- wine_scaled[train_index,]
testData_scaled <- wine_scaled[-train_index,]

# Building model 
#Set the random seed to make the results static
set.seed(321)
NN <- neuralnet(formula = QS ~., data=trainData_scaled, hidden = 4)

#Visualization of the neural network
#neuralnet allows you to plot the network
plot(NN,rep="best")

#... But the plot is difficult to read
#Let us use the toolset from "NeuralNetTools"
library(NeuralNetTools)
plotnet(NN)

#variable importance
#two different methods to look at them
garson(NN)
olden(NN)

#sensitivity analysis
#by default quantiles
lekprofile(NN)
#by clusters - group data into 3 clusters
set.seed(321)
lekprofile(NN,group_val=3)
#Reveal the characteristics of the clusters
set.seed(321)
lekprofile(NN,group_val=3,group_show=TRUE)

#Predicting with neural network
y_NN <- predict(NN, newdata = testData_scaled)
head(y_NN)


# convert predicted y into  original unit
y_NN <- y_NN*sd(trainData[,"QS"]) + mean(trainData[,"QS"])
head(y_NN)

# Assessing model
stats_NN=cbind(csaccuracy(testData[,"QS"],y_NN,mean(trainData[,"QS"])))
stats_NN

#Gathering the predictive accuracy statistics 
allstats = sumstats
allstats=rbind(allstats[1:3,],RegTree=stats_regtree,NeuralNet=stats_NN)
rownames(allstats)[4:5]=c('RegTree','NeuralNet')
allstats

##Unsupervised Learning - K-means##
library(factoextra)
# select optimal number of clusters based 3 methods
#Elbow method - where is the kink? ==> K=7
fviz_nbclust(trainData_scaled[-12], kmeans, method = "wss")
#Gap stat method ==> K=2,3,7.10
fviz_nbclust(trainData_scaled[-12], kmeans, method = "gap_stat")
#Silhouette method ==> K=2,7
fviz_nbclust(trainData_scaled[-12], kmeans, method = "silhouette")

# only select numerical variables
# set the number of cluster at 6 based on the Elbow method discussed above
# The input nstart is the number of random sets to start with when initializing the algorithm
set.seed(123)
km <- kmeans(trainData_scaled[-12], centers = 7, nstart = 25)
#number of obs per cluster
km$size
#cluster membership in percentage
(km$size/nrow(trainData))*100

# visualizing the result
fviz_cluster(km, data = trainData_scaled[-12])

# Mean for each cluster 
cl_stats=aggregate(trainData, by=list(cluster=km$cluster), mean)
cl_stats

# Prediction with sample mean
library(clue)
#First predict the cluster that each row of the test data most likely belong
test_cl=as.integer(cl_predict(km, newdata = testData_scaled[-12]))
#Match the cluster classification with the test set data ID and actual observation
c_km=data.frame(id=as.integer(rownames(testData)),actual=testData$QS,cl=test_cl)
#Look up the mean of the prices for each cluster, and match it to the predicted classification
y_km=merge(c_km, cl_stats[c(1,13)],by.x="cl",by.y="cluster")


#Calculate predictive accuracy metrics
stats_km=cbind(csaccuracy(y_km$actual,y_km$QS,mean(trainData[,"QS"])))
stats_km

#Prediction with cluster specific regression

#A function to fit and calculate the predicted value for each cluster.
#You will need to make sure that any input variables that may cause multicollinearity 
#in the regression in each cluster are removed.
kmWithReg<-function(kmobj,train,test,testcl){
  for(i in 1:max(kmobj$cluster)){
    #Extract training data for each cluster
    dat=train[(kmobj$cluster==i),]
    #Fit regression model for that cluster
    #Change the name of your output variable here. 
    #You can also change the model specification.
    regfit=lm(QS~.,data=dat)
    #Extract the relevant test data for this cluster
    newdat=test[(testcl==i),]
    #Predict using the fitted regression model
    pred=predict(regfit,newdata=newdat)
    #save the outputs
    saved=data.frame(id=as.integer(rownames(newdat)),
                     actual=newdat$QS,predicted=pred)
    if(i==1){
      yhat=saved
    }else{
      
      yhat=rbind(yhat,saved)
    }
  }
  return(yhat)
}

#Reduce the number of clusters to make sure we have adequate observations per cluster
km4reg=kmeans(trainData_scaled[-12], centers = 2, nstart = 25)
fviz_cluster(km4reg, data = trainData_scaled[-12])

# Mean for each cluster
cl_stats2=aggregate(trainData, by=list(cluster=km4reg$cluster), mean)
cl_stats2

# Obtaining the predictive cluster for the test data
test_cl2=as.integer(cl_predict(km4reg, newdata = testData_scaled[-12]))


kmregFull=kmWithReg(km4reg,trainData,testData,test_cl2)

#Remove the variable "CA" from the data set, as it seems to be a perfect discriminator of the clusters, and will cause multicolinearity issues in regression.
kmregWithoutCA=kmWithReg(km4reg,trainData[,-3],testData[,-3],test_cl2)

#Calculating the predictive accuracy statistics
stats_kmregFull=cbind(csaccuracy(kmregFull$actual,kmregFull$predicted,mean(trainData[,"QS"])))

#Calculating the predictive accuracy statistics
stats_kmregWithoutCA=cbind(csaccuracy(kmregWithoutCA$actual,kmregWithoutCA$predicted,mean(trainData[,"QS"])))

#Gathering predictive accuracy statistics
allstats=rbind(allstats[1:5,],stats_km,stats_kmregFull,stats_kmregWithoutCA)
rownames(allstats)[6:8]=c('kMeans','kMeansRegFUll','kmregWithoutCA')
allstats

###Unsupervised Learning - k-nearest neighbour##
#Loading the library
library(caret)
#Let's look at how well the knn regression fits the training data - use scaled data here
#Remember you need to remove the output variable to train k-nn (unsupervised)
knn_fit <- knnreg(x = trainData_scaled[, -12] , y = trainData_scaled[,"QS"] , 
                  k = as.integer(sqrt(nrow(trainData))))

#Don't forget to convert the predictions back to raw data unit when obtaining predictions
y_knnm <- predict(knn_fit,testData_scaled[,-12])
y_knnm<-y_knnm*sd(trainData[,"QS"])+mean(trainData[,"QS"])
#Compute predictive accuracy metrics for k-nn
stats_knn <- cbind(csaccuracy(testData[,"QS"],y_knnm,mean(trainData[,"QS"])))

#Gathering predictive accuracy statistics
allstats=rbind(allstats[1:8,],stats_knn)
rownames(allstats)[9]='knn'
allstats



##################33Question2 ###########Aggregate all predictions and actual QS
actualQS=testData[,"QS"]

Linear_yhat=pfit1[, "fit"]
Stepwise_yhat=pfit2[, "fit"]
Nonlinear_yhat=pfit3[, "fit"]
RegTree_yhat=y_regtree
NeuralNet_yhat=y_NN[,1]
kMeans_yhat=y_km$QS
kMeansRegFUll_yhat=kmregFull$predicted
kmregWithoutCA_yhat=kmregWithoutCA$predicted
knn_yhat=y_knnm


##Specific loss function
# An underestimate of the prediction is weighed by a factor of 5.
# An overestimate of the prediction that is less than 20% deviation from the actual QS is weighed by a factor of 1.
# A severe overestimate of the prediction that is greater than 20% deviation from the actual QS is weighed by a factor of 3.

# error = actualQS-Linear_yhat
# actual = actualQS

wine_maker_loss_function = function(error,actual){
  relative=100*(error/actual)
  lossDF=as.data.frame(t(rbind(relative,actual)))

  highQualityOverestimate               =lossDF[lossDF$actual>=7 & lossDF$relative <= 0, 1]
  highQualityUnderestimateWithin22      =lossDF[lossDF$actual>=7 & lossDF$relative > 0 & lossDF$relative <= 22, 1]
  highQualityUnderestimateWithin22to29  =lossDF[lossDF$actual>=7 & lossDF$relative > 22 & lossDF$relative <= 29, 1]
  highQualityUnderestimateWithin29to36  =lossDF[lossDF$actual>=7 & lossDF$relative > 29 & lossDF$relative <= 36, 1]
  highQualityUnderestimateMoreThan36    =lossDF[lossDF$actual>=7 & lossDF$relative > 36, 1]
  
  
  lowQualityUnderestimate             =lossDF[lossDF$actual<7 & lossDF$relative>=0, 1]
  lowQualityOverestimateWithin22      =lossDF[lossDF$actual<7 & lossDF$relative < 0 & lossDF$relative >= -22, 1]
  lowQualityOverestimateWithin22to29  =lossDF[lossDF$actual<7 & lossDF$relative < -22 & lossDF$relative >= -29, 1]
  lowQualityOverestimateWithin29to36  =lossDF[lossDF$actual<7 & lossDF$relative < -29 & lossDF$relative >= -36, 1]
  lowQualityOverestimateMoreThan36    =lossDF[lossDF$actual<7 & lossDF$relative < -36, 1]
  
  loss= sum(1*abs(highQualityOverestimate)) + 
      sum(8*abs(highQualityUnderestimateWithin22)) + 
      sum(18*abs(highQualityUnderestimateWithin22to29)) + 
      sum(28*abs(highQualityUnderestimateWithin29to36)) + 
      sum(38*abs(highQualityUnderestimateMoreThan36)) + 
      sum(1*abs(lowQualityUnderestimate)) + 
      sum(8*abs(lowQualityOverestimateWithin22)) + 
      sum(18*abs(lowQualityOverestimateWithin22to29)) + 
      sum(28*abs(lowQualityOverestimateWithin29to36)) + 
      sum(38*abs(lowQualityOverestimateMoreThan36))
  
  return(loss/length(error))
}


wine_maker_loss=rbind(Linear=wine_maker_loss_function(actualQS-Linear_yhat,actualQS),
                      Stepwise=wine_maker_loss_function(actualQS-Stepwise_yhat,actualQS),
                      Nonlinear=wine_maker_loss_function(actualQS-Nonlinear_yhat,actualQS),
                      RegTree=wine_maker_loss_function(actualQS-RegTree_yhat,actualQS),
                      NN_1_4_seed321=wine_maker_loss_function(actualQS-y_NN[,1],actualQS),
                      NN_1_4_seed123=wine_maker_loss_function(actualQS-y_NN1[,1],actualQS),
                      NN_1_4_seed888=wine_maker_loss_function(actualQS-y_NN2[,1],actualQS),
                      NN_1_5_seed321=wine_maker_loss_function(actualQS-y_NN2_5[,1],actualQS),
                      NN_1_5_seed123=wine_maker_loss_function(actualQS-y_NN1_2[,1],actualQS),
                      kMeans=wine_maker_loss_function(actualQS-kMeans_yhat,actualQS),
                      kMeansRegFull_2cluster=wine_maker_loss_function(actualQS-kMeansRegFUll_yhat,actualQS),
                      kmregWithoutCA_2cluster=wine_maker_loss_function(actualQS-kmregWithoutCA_yhat,actualQS),
                      kMeansRegFull_7cluster=wine_maker_loss_function(actualQS-kmregfullv2$predicted,actualQS),
                      kmregWithoutCA_7cluster=wine_maker_loss_function(actualQS-kmregv2$predicted,actualQS),
                      knn=wine_maker_loss_function(actualQS-knn_yhat,actualQS))

wine_maker_loss


finalStats = cbind(allstats, wine_maker_loss[,1])
colnames(finalStats)[5] = "WineMakerLoss"
finalStats


#Save the workspace
save.image("WineWorkspaceSession3.RData")

