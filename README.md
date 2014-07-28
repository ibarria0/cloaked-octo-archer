# Predicting exercise manner using activity data

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


##  Libraries and Multiprocessing


```r
library(caret)
require(doMC)
registerDoMC(3)
```

## Load Data

```r
df <- read.csv('pml-training.csv')
```

## Remove problematic columns
These columns are dates and factors; problematic for training.

```r
df <- df[,-c(1:7)]
```

## Remove NearZeroVars
This takes care of all those DIV#0 values and nearZeroVar covariates.

```r
nsv <- nearZeroVar(df)
clean <- df[,-nsv]
```

## Partition Data

```r
inTrain <- createDataPartition(y=clean$classe,p=0.6,list=FALSE)
training <- clean[inTrain,]
testing <- clean[-inTrain,]
```

## Compute preProcessor
Imputing is very important because there are a ton of NAs.

```r
pre <- preProcess(training[,-94],method=c('center','scale','knnImpute','pca'))
```

## Preprocess data sets

```r
trainPC <- predict(pre,training[,-94])
testPC <- predict(pre,testing[,-94])
```

## Build model using Random Forest
This takes a long time....

```r
model <- train(training$classe ~ ., data=trainPC, method='rf')
```

## confusionMatrix against preProccessed testing set

```r
confusionMatrix(testing$classe, predict(model,testPC))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2196   12   16    4    4
##          B   36 1446   26    5    5
##          C    5   25 1321   10    7
##          D    2    6   75 1200    3
##          E    1   16   15   24 1386
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9621          
##                  95% CI : (0.9577, 0.9663)
##     No Information Rate : 0.2855          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9521          
##  Mcnemar's Test P-Value : 5.31e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9804   0.9608   0.9092   0.9654   0.9865
## Specificity            0.9936   0.9886   0.9926   0.9870   0.9913
## Pos Pred Value         0.9839   0.9526   0.9656   0.9331   0.9612
## Neg Pred Value         0.9922   0.9907   0.9796   0.9934   0.9970
## Prevalence             0.2855   0.1918   0.1852   0.1584   0.1791
## Detection Rate         0.2799   0.1843   0.1684   0.1529   0.1767
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9870   0.9747   0.9509   0.9762   0.9889
```
