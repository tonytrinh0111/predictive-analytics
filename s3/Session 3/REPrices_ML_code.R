#######################################################################
#Predictive Analytics - Term 4 2020
#Predictive Modelling with Machine Learning Techniques (1)
#R code ONLY
#######################################################################

#Load saved workspace
load('REWorkspace.RData')

##Regression Tree##
# Load the libraries
# Note rpart.plot enables us to plot the tree visually
library(rpart)
library(rpart.plot)
library(vip)

# Building model - anaova means looking at variance of price or variation explained. It simply mean the predictive accuracy is measured by RMSE
regtree <- rpart(formula = Price ~ ., data    = trainData, method  = "anova")

# Visualizing the tree
rpart.plot(regtree)
#The summary command gives you detailed information at every level of the tree.
#summary(regtree)

#Variable importance from the tree algorithm
vip(regtree)

#Pruning tree - complexity coefficient controls the size of the tree
ptree<-prune.rpart(regtree,cp=0.05)
rpart.plot(ptree)

# prediction
y_regtree<- predict(regtree, newdata = testData)
head(y_regtree)

# Assessing model
stats_regtree=cbind(csaccuracy(testData[,"Price"],y_regtree,mean(trainData[,"Price"])),
                    Banker=banker(testData[,"Price"]-y_regtree),
                    REAgent=agent(testData[,"Price"]-y_regtree,testData[,"Price"]))
stats_regtree

##Neural Network##
library(neuralnet)

# scaled first five numeric data 
# do not scale the dummy variables
realestate_scaled <- realestate
for(i in 1:5){
  realestate_scaled[,i]=as.numeric(scale(realestate[,i]))
}

# creating training and test set using the same random data partition as before
trainData_scaled <- realestate_scaled[train_index,]
testData_scaled <- realestate_scaled[-train_index,]

# Building model 
#Set the random seed to make the results static
set.seed(321)
NN <- neuralnet(formula = Price ~., data=trainData_scaled, hidden = 4)

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
lekprofile(NN,xsel=c('Beds','Baths','Cars','Area'))
#by clusters - group data into 3 clusters
set.seed(321)
lekprofile(NN,xsel=c('Beds','Baths','Cars','Area'),group_val=3)
#Reveal the characteristics of the clusters
set.seed(321)
lekprofile(NN,xsel=c('Beds','Baths','Cars','Area'),group_val=3,group_show=TRUE)

#Predicting with neural network
y_NN <- predict(NN, newdata = testData_scaled)
head(y_NN)

# convert predicted y into  original unit
y_NN <- y_NN*sd(trainData[,"Price"]) + mean(trainData[,"Price"])
head(y_NN)

# Assessing model
stats_NN=cbind(csaccuracy(testData[,"Price"],y_NN,mean(trainData[,"Price"])),
               Banker=banker(testData[,"Price"]-y_NN),
               REAgent=agent(testData[,"Price"]-y_NN,testData[,"Price"]))

#Gathering the predictive accuracy statistics 
allstats=rbind(allstats[1:3,],RegTree=stats_regtree,NeuralNet=stats_NN)
rownames(allstats)[4:5]=c('RegTree','NeuralNet')
allstats

##Unsupervised Learning - K-means##
library(factoextra)
# select optimal number of clusters based 3 methods
#Elbow method - where is the kink? 
fviz_nbclust(trainData_scaled[-1], kmeans, method = "wss")
#Gap stat method
fviz_nbclust(trainData_scaled[-1], kmeans, method = "gap_stat")
#Silhouette method
fviz_nbclust(trainData_scaled[-1], kmeans, method = "silhouette")

# only select numerical variables
# set the number of cluster at 6 based on the Elbow method discussed above
# The input nstart is the number of random sets to start with when initializing the algorithm
set.seed(123)
km <- kmeans(trainData_scaled[-1], centers = 6, nstart = 25)
#number of obs per cluster
km$size
#cluster membership in percentage
(km$size/nrow(trainData))*100

# visualizing the result
fviz_cluster(km, data = trainData_scaled[-1])

# Mean for each cluster 
cl_stats=aggregate(trainData, by=list(cluster=km$cluster), mean)
cl_stats

# Prediction with sample mean
library(clue)
#First predict the cluster that each row of the test data most likely belong
test_cl=as.integer(cl_predict(km, newdata = testData_scaled[-1]))
#Match the cluster classification with the test set data ID and actual observation
c_km=data.frame(id=as.integer(rownames(testData)),actual=testData$Price,cl=test_cl)
#Look up the mean of the prices for each cluster, and match it to the predicted classification
y_km=merge(c_km, cl_stats[1:2],by.x="cl",by.y="cluster")


#Calculate predictive accuracy metrics
stats_km=cbind(csaccuracy(y_km$actual,y_km$Price,mean(trainData[,"Price"])),
               Banker=banker(y_km$actual-y_km$Price),
               REAgent=agent(y_km$actual-y_km$Price,y_km$actual))
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
    regfit=lm(Price~.,data=dat)
    #Extract the relevant test data for this cluster
    newdat=test[(testcl==i),]
    #Predict using the fitted regression model
    pred=predict(regfit,newdata=newdat)
    #save the outputs
    saved=data.frame(id=as.integer(rownames(newdat)),
                     actual=newdat$Price,predicted=pred)
    if(i==1){
      yhat=saved
    }else{
      
      yhat=rbind(yhat,saved)
    }
  }
  return(yhat)
}

#Reduce the number of clusters to make sure we have adequate observations per cluster
km4reg=kmeans(trainData_scaled[-1], centers = 2, nstart = 25)

# Mean for each cluster
cl_stats2=aggregate(trainData, by=list(cluster=km4reg$cluster), mean)
cl_stats2

# Obtaining the predictive cluster for the test data
test_cl2=as.integer(cl_predict(km4reg, newdata = testData_scaled[-1]))

#Remove the variable "Subdivision" from the data set, as it is the perfect discriminator of the clusters, and will cause multicolinearity issues in regression.
kmreg=kmWithReg(km4reg,trainData[,-9],testData[,-9],test_cl2)

#Calculating the predictive accuracy statistics
stats_kmreg=cbind(csaccuracy(kmreg$actual,kmreg$predicted,mean(trainData[,"Price"])),
                  Banker=banker(kmreg$actual-kmreg$predicted),
                  REAgent=agent(kmreg$actual-kmreg$predicted,kmreg$actual))

#Gathering predictive accuracy statistics
allstats=rbind(allstats[1:5,],stats_km,stats_kmreg)
rownames(allstats)[6:7]=c('kMeans','kMeansReg')
allstats

###Unsupervised Learning - k-nearest neighbour##
#Loading the library
library(caret)
#Let's look at how well the knn regression fits the training data - use scaled data here
#Remember you need to remove the output variable to train k-nn (unsupervised)
knn_fit <- knnreg(x = trainData_scaled[, -1] , y = trainData_scaled[,"Price"] , 
                  k = as.integer(sqrt(nrow(trainData))))

#Don't forget to convert the predictions back to raw data unit when obtaining predictions
y_knnm <- predict(knn_fit,testData_scaled[,-1])
y_knnm<-y_knnm*sd(trainData[,"Price"])+mean(trainData[,"Price"])
#Compute predictive accuracy metrics for k-nn
stats_knn <- cbind(csaccuracy(testData[,"Price"],y_knnm,mean(trainData[,"Price"])),
                   Banker=banker(testData[,"Price"]-y_knnm),
                   REAgent=agent(testData[,"Price"]-y_knnm,testData[,"Price"]))

#Gathering predictive accuracy statistics
allstats=rbind(allstats[1:7,],stats_knn)
rownames(allstats)[8]='knn'
allstats

#Save the workspace
save.image("REPriceSession3.RData")
