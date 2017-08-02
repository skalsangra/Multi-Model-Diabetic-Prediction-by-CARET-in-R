library(caret)

#1-Data Acquisition
# accessing data from mass
library(MASS)
data("Pima.te")
# rename the dataset
dataset <- Pima.te
#***************************************************************************
#2-Data Exploration and Analysis
# summarize attribute distributions
summary(dataset)

# split input and output
x <- dataset[,1:7]
y <- dataset[,8]

# barplot for class breakdown
plot(y)

# box and whisker plots for each attribute
featurePlot(x=x, y=y, plot="box")

#****************************************************************************
#3-Data Partition
#lets check Proportions of class in original data
prop.table(table(dataset$type))

# create a list of 80% of the rows in the original dataset we can use for training
test_index <- createDataPartition(dataset$type, p=0.80, list=FALSE)

# select 20% of the data for Testing
test_dataset <- dataset[-test_index,]
# use the remaining 80% of data to training and testing the models
train_dataset <- dataset[test_index,]

# Verify proportions.
prop.table(table(test_dataset$type))
prop.table(table(train_dataset$type))
#****************************************************************************
#4-Model control tunning
# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

#5-Model Fitting
# a) linear algorithms
set.seed(7)
fit.lda <- train(type~., data=train_dataset, method="lda", metric=metric, trControl=control)

set.seed(7)
fit.glm <- train(type~., data=train_dataset, method="glm", metric=metric, trControl=control)


# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(type~., data=train_dataset, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(7)
fit.knn <- train(type~., data=train_dataset, method="knn", metric=metric, trControl=control)

# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(type~., data=train_dataset, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(7)
fit.rf <- train(type~., data=train_dataset, method="rf", metric=metric, trControl=control)

#Naive Bayes
set.seed(7)
fit.naive <- train(type~., data=train_dataset, method="naive_bayes", metric=metric, trControl=control)

#AdaBoost
set.seed(7)
fit.aboost <- train(type~., data=train_dataset, method="adaboost", metric=metric, trControl=control)

#****************************************************************************
#6-Model Evaluation
# summarize accuracy of models
results <- resamples(list(lda=fit.lda, glm=fit.glm, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf, naive=fit.naive, aboost=fit.aboost))
summary(results)

# compare accuracy of models
dotplot(results)

#****************************************************************************
# summarize Best Models
print(fit.rf)
print(fit.aboost)
print(fit.naive)
print(fit.glm)
#****************************************************************************

#6-Prediction
# estimate skill of glm and rf on the test dataset
predictions <- predict(fit.glm, test_dataset)
confusionMatrix(predictions, test_dataset$type)

predictions <- predict(fit.rf, test_dataset)
confusionMatrix(predictions, test_dataset$type)