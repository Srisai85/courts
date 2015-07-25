---
Title: "Predicting US SupremeCourt Decision Making- Motivation and Model Building"
Author: "Srisai Sivakumar"
Date: "Friday, July 17, 2015"
Output:
  html_document:
    pandoc_args: [
      "+RTS", "-K64m",
      "-RTS"
    ]
---
# Predicting US SupremeCourt Decision Making

## By: Srisai Sivakumar

### Part 1: Introduction, description, assumptions, models and tuning

This is the first of the 3 part series on using machine learning algorithms to predict the juedegemt of the cases appearing before the Supreme Court of the USA. If the models give rockstar grade predictions, perhaps, one day, the courts can be run by R codes and excel sheets?

### Motivation

Political scientists and legal academics have long scrutinized the legal entities like Court with the intent to understand what motivates the justices and get insights into predicting the coutcomes of cases.  Lawyers, too, possess expertise that should enable them to forecast legal events with some accuracy. After all, the everyday practice of law requires lawyers to predict court decisions in order to advise clients or determine suitable strategies. Prediction of success in the case is of paramount importance for several reasons. In the course of litigation, lawyers constantly make strategic decisions and/or advise their clients on the basis of these predictions. Attorneys make decisions about future courses of action, such as whether to take on a new client, the value of a case, whether to advise the client to enter into settlement negotiations, and whether to accept a settlement offer or proceed to trial. Thus, these
judgments are influential in shaping the cases and the mechanisms selected to resolve them. Clients' choices and outcomes therefore depend on the abilities of their counsel to make reasonably accurate forecasts
concerning case outcomes. 

My uncle, who is a lawyer in India, says that in civil cases, after depositions of key witnesses or at the close of discovery, the parties reassess the likelihood of success at trial in light of the impact of these events. Ability to foresee or predict the outcomes/verdict is an invaluable insight not just to the client, but also for the possible lawyer, whose professional and financial success, satisfaction of his client is strongly correlated to his ability to predict the outcome of the case. 

This study examines the cases using statistical models that relies on the general case characteristics.

### Assumption

Our principal goal in constructing statistical models capable of predicting the outcome of Supreme Court cases "prospectively"", using only information available prior to the judgement or oral argument. 

A cornerstone of creating such statistical models is an assumption about
the temporal stability in the Justices' behavior. In other words, its assumed that observable patterns in the Justices' past behavior would hold true for their future behavior.  

### Set-up

This study attempts to predict the outcome of cases heard by the US Supreme Court. The Statistical forecasting model is based on information derived from past Supreme Court decisions. The model discerns patterns in the justices' decisions based on observable case characteristics. Bases on these patterns, the model allows the construction of classification trees and other such models to predict outcomes based on these characteristics
of the cases. This correctness or goodness of the model shall be evaluated by allowing it to predict the outcomes of 'settled' cases. Shoult this accuracy be good enough, the model may be used to predict the verdicts of future or unresolved cases.

### Brief description of US judicial setup and summary of dataset

The legal system of the United States operates at the state level and at the federal level. Federal courts hear cases beyond the scope of state law. Federal courts are divided into:

- District Courts: makes initial decision

- Circuit Courts: hears appeals from the district courts

- Supreme Court: highest level - makes final decision 

The Supreme Court of the United States Consists of nine judges ("justices"), appointed by the President. Justices are distinguished judges, professors of law, state and federal attorneys. The final verdict is the collective judgement is taken by a 'bench' of judges. Predictind the collective outcome needs predicting the juegements of all the 3 judges in the bench. It can be thought of predicting the decisions of one judge and using the same process to the other two. In essence, the framework to predict the judgements of one judge can be extended to include multiple.

This study will focus on the judgements of one such judge, Justice Stevens. We examine his judgements in cases from 1994 through 2001, totalling to 567 cases.

### The data and assumptions

The data set has 567 cases. The variables in the data set are:

- Docket: docket number

- Term: year of hearing (e.g., 1999, 2001, etc)

- Circuit: circuit of origin (1st - 11th, DC, FED)

- Issue: issue area of the case.

- Petitioner: type of petitioner  (e.g., the United States, an employer, etc.).

- Respondent: type of respondent (e.g., Americal Indian, Business, etc.).

- LowerCourt:  ideological direction (liberal or conservative) of the lower
court ruling.

- Unconst:  whether the petitioner argued that a law or practice is unconstitutional.

- Reverse:  affirm/reverse judgement of lower court.



```r
setwd("~/working_directory/R/AE/Trees")
# read csv file
stevens1 = read.csv("stevens.csv")
# examining the structure of the data frame
str(stevens1)
```

```
## 'data.frame':	566 obs. of  9 variables:
##  $ Docket    : Factor w/ 566 levels "00-1011","00-1045",..: 63 69 70 145 97 181 242 289 334 436 ...
##  $ Term      : int  1994 1994 1994 1994 1995 1995 1996 1997 1997 1999 ...
##  $ Circuit   : Factor w/ 13 levels "10th","11th",..: 4 11 7 3 9 11 13 11 12 2 ...
##  $ Issue     : Factor w/ 11 levels "Attorneys","CivilRights",..: 5 5 5 5 9 5 5 5 5 3 ...
##  $ Petitioner: Factor w/ 12 levels "AMERICAN.INDIAN",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ Respondent: Factor w/ 12 levels "AMERICAN.INDIAN",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ LowerCourt: Factor w/ 2 levels "conser","liberal": 2 2 2 1 1 1 1 1 1 1 ...
##  $ Unconst   : int  0 0 0 0 0 1 0 1 0 0 ...
##  $ Reverse   : int  1 1 1 1 1 0 1 1 1 1 ...
```

We have assumed termporal stability of the judge. This means that the year of hearing has no bearing on the outcome of the judge. It is also fair to assume that the judge will reverse or affirm the ruling of a lower court if he deems necessary, irrespective of the docket number of the case. So we remove the variables 'Docket' and 'Term' from the data frame.

The dependent variable, 'Reverse' is categorized '0' and '1'. It may be noted from the str output that this variable is of type integer. Its useful to have them as a caregorical variable. So we convery them into a factor, with 'yes' replacing 1, and 'no', 0.


```r
# converting dependent variable (Reverse) to factor.
stevens1$Reverse <- factor(ifelse(stevens1$Reverse == 1, "yes","no"))
# transforming the unconst variable as a factor too.
stevens1$Unconst = as.factor(stevens1$Unconst)
# removing Docket and Term variables from the data frame
stevens = stevens1[,-c(1,2)]
# examining the structure of the new data frame
str(stevens)
```

```
## 'data.frame':	566 obs. of  7 variables:
##  $ Circuit   : Factor w/ 13 levels "10th","11th",..: 4 11 7 3 9 11 13 11 12 2 ...
##  $ Issue     : Factor w/ 11 levels "Attorneys","CivilRights",..: 5 5 5 5 9 5 5 5 5 3 ...
##  $ Petitioner: Factor w/ 12 levels "AMERICAN.INDIAN",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ Respondent: Factor w/ 12 levels "AMERICAN.INDIAN",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ LowerCourt: Factor w/ 2 levels "conser","liberal": 2 2 2 1 1 1 1 1 1 1 ...
##  $ Unconst   : Factor w/ 2 levels "0","1": 1 1 1 1 1 2 1 2 1 1 ...
##  $ Reverse   : Factor w/ 2 levels "no","yes": 2 2 2 2 2 1 2 2 2 2 ...
```

### Training and Test data sets

We now have the data in hand and we can proceed with model building. The primary tools of choice would be the options from the Caret , that provides a unified framework for numerous machine learning algorithms.

Before we split the data into training and test set, its important to give this step a thought. We are trying to build a model that would predict the outcomes of cases in coing years. Currently, the most prevalent method to get the training and test data is to split the data ramdomly into training set and test set. But one must think how predicting the outcome of a case from 1996 would give confidence in the model's ability to predict the outcome of a case in 2002. A possible way around is to see if we can split the data by years or term of the cases' hearing. We can make a training set out of the cases from 1994-1999 and test set from the cases 2000-01. But with this approach, we have to ensure that the test set have any 'new' kind or class of case that the training test doesnt have. This is a hasslenot worth endring, especially with the assumption of temporal stability of the judge. So its fair to assume that the year or term of hearing has no bearing or influence on the precitions of the outcome. So we stick to the widely used random subsetting to get the training and test sets.

As we proceed with the spits, we set seed to enable reproducable results. We split the data into training and test sets, with 70% and 30% of the data respectively.


```r
library(caret)
# set seed - 3000
set.seed(3000)
inTraining <- createDataPartition(stevens$Reverse, p = .7, list = FALSE)
Train <- stevens[ inTraining,]
Test  <- stevens[-inTraining,]
str(Train)
```

```
## 'data.frame':	397 obs. of  7 variables:
##  $ Circuit   : Factor w/ 13 levels "10th","11th",..: 7 3 9 11 13 11 12 2 4 8 ...
##  $ Issue     : Factor w/ 11 levels "Attorneys","CivilRights",..: 5 5 9 5 5 5 5 3 5 5 ...
##  $ Petitioner: Factor w/ 12 levels "AMERICAN.INDIAN",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ Respondent: Factor w/ 12 levels "AMERICAN.INDIAN",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ LowerCourt: Factor w/ 2 levels "conser","liberal": 2 1 1 1 1 1 1 1 1 1 ...
##  $ Unconst   : Factor w/ 2 levels "0","1": 1 1 1 2 1 2 1 1 1 2 ...
##  $ Reverse   : Factor w/ 2 levels "no","yes": 2 2 2 1 2 2 2 2 2 2 ...
```

Now that the training and test sets are ready, we can go ahead with model building.

### Models

Logistic regression is likely to provide a good starting point for this binary classification problem. The linear nature of the algorithm might limit the accuracy of the prediction, but the more important concern is interpretability of the final model, especially interpreting the coefficents by non-technical or upper management people. It also needs more thought in interpreting which factors are important and critically, predicting the outcome without a 'tree' or a graph structure to help.

#### Baseline

Before we jump into model building, its important we understand the purpose of building the model. We try to biuld the model to predict the outcomes of Justice Stevens. It can be taken for granted that no model is be 100% accurate. But what accuracy can be termed 'good'? Its important to resolve this before we start making predictions.

A good starting point is 'naive' prediction. By naive, I mean the lack of sophestication in the process of making a prediction.

This can be done in 2 ways.

First, randomly select if the verdict of the lower court is overruled by the Supreme Court. Compare this with the training data. Since there is randomness in the process, its worth averaging it over, say 100 iterations.


```r
b = rep(NA,100)
for (i in 1:100){
        b[i] = mean((sample(c("yes","no"),nrow(Train),replace=T) == Train$Reverse))
}
rand_bl = mean(b)
print(paste("the random baseline is:",as.character(round(rand_bl,3))))
```

```
## [1] "the random baseline is: 0.5"
```

Unsurprisingly, this gives a accuracy of close to 0.5. This is what one would expect in a random process.

Another naive way of doing this is to assume all the verdicts of the cases are same and is to reverse the ruling of the lower court.


```r
baseline = mean((Train$Reverse == 'yes'))

print(paste("the baseline is:",as.character(round(baseline,3))))
```

```
## [1] "the baseline is: 0.547"
```
This is a better baseline than random choice. We would use this to evaluate the goodness of the predicting algorithm.

#### Model Building

We begin with one of the more interpretable models, CART. This provides a easily interpretable solution at the possible expense of accuracy. We biuld a model without tuning. And then we proceed to tune the model in search of better results.

#### Classification Trees- CART

The caret package, by default, uses simple bootstrap resampling method to automatically choose the tuning parameters associated with the best value. Different algorithms such as repeated K-fold cross-validation, leave-one-out, etc. can be used as well and will be examined in the subsequent sections.


```r
# No tune model
set.seed(3000)
# train the model using training set Train and method as 'rpart' for classification trees
tree = train(Reverse ~ ., data = Train, method="rpart")
```

```
## Loading required package: rpart
```

```r
tree
```

```
## CART 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 397, 397, 397, 397, 397, 397, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa      Accuracy SD  Kappa SD  
##   0.01388889  0.6032075  0.1922981  0.03582297   0.07038009
##   0.04444444  0.6246983  0.2417473  0.04208487   0.08754367
##   0.20000000  0.5869898  0.1436649  0.05689850   0.13703793
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.04444444.
```

```r
# predict the outcomes of the test set, Test.
predicttree = predict(tree, newdata = Test)
confusionMatrix(Test$Reverse, predicttree)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  57  20
##        yes 24  68
##                                          
##                Accuracy : 0.7396         
##                  95% CI : (0.6667, 0.804)
##     No Information Rate : 0.5207         
##     P-Value [Acc > NIR] : 4.512e-09      
##                                          
##                   Kappa : 0.4774         
##  Mcnemar's Test P-Value : 0.6511         
##                                          
##             Sensitivity : 0.7037         
##             Specificity : 0.7727         
##          Pos Pred Value : 0.7403         
##          Neg Pred Value : 0.7391         
##              Prevalence : 0.4793         
##          Detection Rate : 0.3373         
##    Detection Prevalence : 0.4556         
##       Balanced Accuracy : 0.7382         
##                                          
##        'Positive' Class : no             
## 
```

```r
accuracy_tree = confusionMatrix(Test$Reverse, predicttree)$overall[1]
names(accuracy_tree) <- NULL

print(paste0("The untuned Tree model gives an accuracy of ", as.character(round(accuracy_tree,4)),", an improvement over the baseline of ",as.character(round((accuracy_tree/baseline - 1 ) * 100,2)),'%'))
```

```
## [1] "The untuned Tree model gives an accuracy of 0.7396, an improvement over the baseline of 35.32%"
```

The last chunk of the post processing script would be used for all the models and would be convenient to have it as a function.


```r
postProcess <- function(model,Test=Test) {
        pred <- predict(model, newdata = Test)
        print(confusionMatrix(Test$Reverse,pred))
        accuracy = confusionMatrix(Test$Reverse, pred)$overall[1]
        names(accuracy) <- NULL
        print(paste0("Model ",deparse(substitute(model)) , " gives an accuracy of ", as.character(round(accuracy,4)),", an improvement over the baseline of ",as.character(round((accuracy/baseline - 1 ) * 100,2)),'%'))
}
```

Back to the tree model, we have a working tree model that we would like to view and understand how the prediction works. We can get the tuning parameter values from the display above. 

We now tune the Tree model in search of further improvement.

We now use 10-fold cross validation as the resampling method and establish 15 tuning points to the model, using the tuneLength option. 


```
## CART 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 357, 357, 358, 358, 357, 357, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa       Accuracy SD  Kappa SD  
##   0.00000000  0.5691667  0.12721464  0.05026816   0.10545495
##   0.01428571  0.6371154  0.25265401  0.06032718   0.12708028
##   0.02857143  0.6296795  0.23888188  0.05819742   0.12179457
##   0.04285714  0.6271795  0.23846227  0.05776286   0.11715346
##   0.05714286  0.6221795  0.23788741  0.05651635   0.11683854
##   0.07142857  0.6273077  0.25064128  0.05287000   0.10434412
##   0.08571429  0.6373077  0.27249695  0.07355693   0.14838198
##   0.10000000  0.6373077  0.27249695  0.07355693   0.14838198
##   0.11428571  0.6373077  0.27249695  0.07355693   0.14838198
##   0.12857143  0.6373077  0.27249695  0.07355693   0.14838198
##   0.14285714  0.6373077  0.27249695  0.07355693   0.14838198
##   0.15714286  0.6373077  0.27249695  0.07355693   0.14838198
##   0.17142857  0.6123077  0.21249695  0.05120703   0.11979206
##   0.18571429  0.5973077  0.17249695  0.04414842   0.11697522
##   0.20000000  0.5642308  0.08092138  0.02638660   0.09124601
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.1571429.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  57  20
##        yes 24  68
##                                          
##                Accuracy : 0.7396         
##                  95% CI : (0.6667, 0.804)
##     No Information Rate : 0.5207         
##     P-Value [Acc > NIR] : 4.512e-09      
##                                          
##                   Kappa : 0.4774         
##  Mcnemar's Test P-Value : 0.6511         
##                                          
##             Sensitivity : 0.7037         
##             Specificity : 0.7727         
##          Pos Pred Value : 0.7403         
##          Neg Pred Value : 0.7391         
##              Prevalence : 0.4793         
##          Detection Rate : 0.3373         
##    Detection Prevalence : 0.4556         
##       Balanced Accuracy : 0.7382         
##                                          
##        'Positive' Class : no             
##                                          
## [1] "Model tree1 gives an accuracy of 0.7396, an improvement over the baseline of 35.32%"
```

It looks like the tree we have reached the maximum performance of the tree models. To check, lets focus on the cp values that gives best performance in both the previous models. It was 0.04444444 and 0.1571429

It may be worth establishing a cp tuning grid in the interval of 0.00 and 0.4, in steps of 0.02. This can be given the the model using the tuneGrid function.


```
## CART 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 357, 357, 358, 358, 357, 357, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp    Accuracy   Kappa       Accuracy SD  Kappa SD  
##   0.00  0.5691667  0.12721464  0.050268156  0.10545495
##   0.02  0.6371154  0.25189927  0.060327177  0.12816786
##   0.04  0.6271795  0.23846227  0.057762857  0.11715346
##   0.06  0.6221795  0.23788741  0.056516347  0.11683854
##   0.08  0.6373077  0.27249695  0.073556928  0.14838198
##   0.10  0.6373077  0.27249695  0.073556928  0.14838198
##   0.12  0.6373077  0.27249695  0.073556928  0.14838198
##   0.14  0.6373077  0.27249695  0.073556928  0.14838198
##   0.16  0.6373077  0.27249695  0.073556928  0.14838198
##   0.18  0.6123077  0.21249695  0.051207032  0.11979206
##   0.20  0.5642308  0.08092138  0.026386596  0.09124601
##   0.22  0.5465385  0.01000000  0.005573606  0.03162278
##   0.24  0.5465385  0.00000000  0.005573606  0.00000000
##   0.26  0.5465385  0.00000000  0.005573606  0.00000000
##   0.28  0.5465385  0.00000000  0.005573606  0.00000000
##   0.30  0.5465385  0.00000000  0.005573606  0.00000000
##   0.32  0.5465385  0.00000000  0.005573606  0.00000000
##   0.34  0.5465385  0.00000000  0.005573606  0.00000000
##   0.36  0.5465385  0.00000000  0.005573606  0.00000000
##   0.38  0.5465385  0.00000000  0.005573606  0.00000000
##   0.40  0.5465385  0.00000000  0.005573606  0.00000000
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.16.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  57  20
##        yes 24  68
##                                          
##                Accuracy : 0.7396         
##                  95% CI : (0.6667, 0.804)
##     No Information Rate : 0.5207         
##     P-Value [Acc > NIR] : 4.512e-09      
##                                          
##                   Kappa : 0.4774         
##  Mcnemar's Test P-Value : 0.6511         
##                                          
##             Sensitivity : 0.7037         
##             Specificity : 0.7727         
##          Pos Pred Value : 0.7403         
##          Neg Pred Value : 0.7391         
##              Prevalence : 0.4793         
##          Detection Rate : 0.3373         
##    Detection Prevalence : 0.4556         
##       Balanced Accuracy : 0.7382         
##                                          
##        'Positive' Class : no             
##                                          
## [1] "Model tree2 gives an accuracy of 0.7396, an improvement over the baseline of 35.32%"
```

It is fair to conclude that we have reached the max performance of the CART model for this data set.

#### Random Forest

Lets look at an extension of the tree based models, the Random Forest. Random forest builds numerous trees, with a set (mtree) number of variables randomly chosen to model each split.

Lets first begin with the basic random forest model, without any tuning.


```
## Random Forest 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 397, 397, 397, 397, 397, 397, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD  
##    2    0.6126192  0.2058805  0.03030442   0.06450431
##   24    0.6163681  0.2245756  0.03224272   0.06564341
##   46    0.6191684  0.2306626  0.03245494   0.06427002
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 46.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  50  27
##        yes 32  60
##                                           
##                Accuracy : 0.6509          
##                  95% CI : (0.5739, 0.7225)
##     No Information Rate : 0.5148          
##     P-Value [Acc > NIR] : 0.0002406       
##                                           
##                   Kappa : 0.2999          
##  Mcnemar's Test P-Value : 0.6025370       
##                                           
##             Sensitivity : 0.6098          
##             Specificity : 0.6897          
##          Pos Pred Value : 0.6494          
##          Neg Pred Value : 0.6522          
##              Prevalence : 0.4852          
##          Detection Rate : 0.2959          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.6497          
##                                           
##        'Positive' Class : no              
##                                           
## [1] "Model rf gives an accuracy of 0.6509, an improvement over the baseline of 19.08%"
```

This is lower than the accuracy of the Cart models. To understand if the default resampling technique of bootstrapping has any effect on the accuracy, we specify the resampling method to be 10-fold CV.


```
## Random Forest 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 357, 357, 358, 358, 357, 357, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD  
##    2    0.6451923  0.2676448  0.04803203   0.09822692
##   24    0.5894231  0.1700268  0.04444265   0.09291305
##   46    0.6046795  0.2014890  0.04939832   0.09992202
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  34  43
##        yes 12  80
##                                           
##                Accuracy : 0.6746          
##                  95% CI : (0.5983, 0.7445)
##     No Information Rate : 0.7278          
##     P-Value [Acc > NIR] : 0.9476          
##                                           
##                   Kappa : 0.3217          
##  Mcnemar's Test P-Value : 5.228e-05       
##                                           
##             Sensitivity : 0.7391          
##             Specificity : 0.6504          
##          Pos Pred Value : 0.4416          
##          Neg Pred Value : 0.8696          
##              Prevalence : 0.2722          
##          Detection Rate : 0.2012          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.6948          
##                                           
##        'Positive' Class : no              
##                                           
## [1] "Model rf1 gives an accuracy of 0.6746, an improvement over the baseline of 23.41%"
```

10-fold CV does improve the performance of the model, albeit marginally. Lets try to tune the model and see if that offers any improvement in prediction accuracy.


```
## Random Forest 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## Pre-processing: centered, scaled 
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 357, 357, 358, 358, 357, 357, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD  
##    2    0.6198718  0.2178083  0.03351910   0.06715485
##    4    0.6123077  0.2125186  0.03694518   0.07159985
##    6    0.6122436  0.2142390  0.04295823   0.08605653
##    8    0.6021154  0.1933835  0.05451277   0.11266768
##   10    0.5869872  0.1662983  0.03838198   0.07358003
##   12    0.5968590  0.1858607  0.04688645   0.09978181
##   14    0.5817949  0.1565437  0.05272299   0.10955031
##   16    0.6070513  0.2080825  0.03389700   0.06711983
##   18    0.5919872  0.1743095  0.03642301   0.07467198
##   20    0.5893590  0.1703284  0.03996720   0.08720574
##   22    0.5868590  0.1639512  0.05393024   0.11507878
##   24    0.6044231  0.1987067  0.04677335   0.09709528
##   26    0.5996154  0.1909620  0.05162651   0.10730695
##   28    0.6072436  0.2036868  0.04590194   0.09872998
##   30    0.5869872  0.1659128  0.04218779   0.08366608
##   32    0.5970513  0.1841262  0.04226364   0.09310156
##   34    0.6019872  0.1938782  0.03318707   0.06438269
##   36    0.6046795  0.2008579  0.03963321   0.08282031
##   38    0.5920513  0.1758645  0.04605105   0.09334916
##   40    0.6171795  0.2271516  0.04358220   0.08895433
##   42    0.6123077  0.2146097  0.04742127   0.10019109
##   44    0.6021154  0.1957311  0.04123107   0.08100232
##   46    0.6022436  0.1959085  0.04685898   0.09815807
##   48    0.6072436  0.2061844  0.03538380   0.07340012
##   50    0.6122436  0.2176806  0.05089549   0.10251835
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  38  39
##        yes 14  78
##                                           
##                Accuracy : 0.6864          
##                  95% CI : (0.6107, 0.7555)
##     No Information Rate : 0.6923          
##     P-Value [Acc > NIR] : 0.6025330       
##                                           
##                   Kappa : 0.3506          
##  Mcnemar's Test P-Value : 0.0009784       
##                                           
##             Sensitivity : 0.7308          
##             Specificity : 0.6667          
##          Pos Pred Value : 0.4935          
##          Neg Pred Value : 0.8478          
##              Prevalence : 0.3077          
##          Detection Rate : 0.2249          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.6987          
##                                           
##        'Positive' Class : no              
##                                           
## [1] "Model rf2 gives an accuracy of 0.6864, an improvement over the baseline of 25.57%"
```

It seems that we have hit the max performance of random forest. Surprisingly random forest performs worse than CART. Though RFs were introduced to deal with issues of overfitting in normal decision trees but this doesn't happen on every data set. It looks like RF overfits the data, which is evident from the fact that rf has higher training set accuracy, but lower test set accuracy. 

Lets examine a similar forest technique, called the cforest.


```
## Conditional Inference Random Forest 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 397, 397, 397, 397, 397, 397, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa       Accuracy SD  Kappa SD  
##    2    0.5387451  0.01738281  0.03331457   0.03795542
##   24    0.6181894  0.21917427  0.03945912   0.07871952
##   46    0.6159803  0.21606729  0.03670316   0.07277996
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 24.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  50  27
##        yes 19  73
##                                           
##                Accuracy : 0.7278          
##                  95% CI : (0.6541, 0.7933)
##     No Information Rate : 0.5917          
##     P-Value [Acc > NIR] : 0.0001584       
##                                           
##                   Kappa : 0.4466          
##  Mcnemar's Test P-Value : 0.3020282       
##                                           
##             Sensitivity : 0.7246          
##             Specificity : 0.7300          
##          Pos Pred Value : 0.6494          
##          Neg Pred Value : 0.7935          
##              Prevalence : 0.4083          
##          Detection Rate : 0.2959          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.7273          
##                                           
##        'Positive' Class : no              
##                                           
## [1] "Model cf gives an accuracy of 0.7278, an improvement over the baseline of 33.15%"
```

This gives very good prediction already. But lets explore if we can make it even better by resapling.


```
## Conditional Inference Random Forest 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 357, 357, 358, 358, 357, 357, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  ROC        Sens       Spec       ROC SD      Sens SD    Spec SD  
##    2    0.6673882  0.0000000  1.0000000  0.07494328  0.0000000  0.0000000
##   24    0.6741703  0.5388889  0.7147186  0.06701710  0.1528199  0.1097099
##   46    0.6694565  0.5333333  0.7192641  0.06478537  0.1509004  0.1070112
## 
## ROC was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 24.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  51  26
##        yes 19  73
##                                           
##                Accuracy : 0.7337          
##                  95% CI : (0.6604, 0.7987)
##     No Information Rate : 0.5858          
##     P-Value [Acc > NIR] : 4.434e-05       
##                                           
##                   Kappa : 0.4592          
##  Mcnemar's Test P-Value : 0.3711          
##                                           
##             Sensitivity : 0.7286          
##             Specificity : 0.7374          
##          Pos Pred Value : 0.6623          
##          Neg Pred Value : 0.7935          
##              Prevalence : 0.4142          
##          Detection Rate : 0.3018          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.7330          
##                                           
##        'Positive' Class : no              
##                                           
## [1] "Model cf1 gives an accuracy of 0.7337, an improvement over the baseline of 34.23%"
```
This is the most accurate prediction yet. 

Lets proceed to the next tree based technique.

#### Boosting

We start investigating the other tree based model, Boosting. As with the other models, we begin with a no-tune model. 


```
## Stochastic Gradient Boosting 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 397, 397, 397, 397, 397, 397, ... 
## 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa      Accuracy SD
##   1                   50      0.6026858  0.1913256  0.02610980 
##   1                  100      0.5996173  0.1895312  0.02752334 
##   1                  150      0.5941583  0.1775383  0.02745476 
##   2                   50      0.6083424  0.2040874  0.03028791 
##   2                  100      0.6028348  0.1960303  0.02846634 
##   2                  150      0.6015460  0.1953766  0.03255769 
##   3                   50      0.6009813  0.1914303  0.03400594 
##   3                  100      0.5966016  0.1854888  0.03524757 
##   3                  150      0.6014542  0.1961742  0.03432607 
##   Kappa SD  
##   0.05623266
##   0.05707916
##   0.05822610
##   0.06118375
##   0.06165662
##   0.06852236
##   0.07020601
##   0.07234428
##   0.07195097
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 50, interaction.depth
##  = 2, shrinkage = 0.1 and n.minobsinnode = 10.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  49  28
##        yes 24  68
##                                           
##                Accuracy : 0.6923          
##                  95% CI : (0.6168, 0.7609)
##     No Information Rate : 0.568           
##     P-Value [Acc > NIR] : 0.0006137       
##                                           
##                   Kappa : 0.3771          
##  Mcnemar's Test P-Value : 0.6773916       
##                                           
##             Sensitivity : 0.6712          
##             Specificity : 0.7083          
##          Pos Pred Value : 0.6364          
##          Neg Pred Value : 0.7391          
##              Prevalence : 0.4320          
##          Detection Rate : 0.2899          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.6898          
##                                           
##        'Positive' Class : no              
##                                           
## [1] "Model gbm gives an accuracy of 0.6923, an improvement over the baseline of 26.66%"
```

This gives an accuracy of close to 0.7. This is already superior to the prediction of tuned RF. Lets proceed with tuning the boosting model. Lets begin with using 10-fold CV as the resampling method.


```
## Stochastic Gradient Boosting 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 357, 357, 358, 358, 357, 357, ... 
## 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa      Accuracy SD  Kappa SD 
##   1                   50      0.6049359  0.1948240  0.07616018   0.1530189
##   1                  100      0.6225000  0.2339398  0.07648227   0.1530789
##   1                  150      0.6276923  0.2424409  0.07849870   0.1564493
##   2                   50      0.6025641  0.1889512  0.07534650   0.1498038
##   2                  100      0.6250641  0.2410551  0.06286443   0.1200140
##   2                  150      0.6275641  0.2435260  0.05639811   0.1128019
##   3                   50      0.6402564  0.2674662  0.06540263   0.1295732
##   3                  100      0.6201282  0.2276094  0.06346420   0.1251619
##   3                  150      0.6301923  0.2506180  0.08443630   0.1669390
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 50, interaction.depth
##  = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  53  24
##        yes 22  70
##                                           
##                Accuracy : 0.7278          
##                  95% CI : (0.6541, 0.7933)
##     No Information Rate : 0.5562          
##     P-Value [Acc > NIR] : 3.183e-06       
##                                           
##                   Kappa : 0.4501          
##  Mcnemar's Test P-Value : 0.8828          
##                                           
##             Sensitivity : 0.7067          
##             Specificity : 0.7447          
##          Pos Pred Value : 0.6883          
##          Neg Pred Value : 0.7609          
##              Prevalence : 0.4438          
##          Detection Rate : 0.3136          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.7257          
##                                           
##        'Positive' Class : no              
##                                           
## [1] "Model gbm1 gives an accuracy of 0.7278, an improvement over the baseline of 33.15%"
```

Changing the resampling method to 10-fold CV improves the accuracy to close to 0.72. Further tuning might improve the accuracy. The tuning parameters for gbm models are 
- interaction.depth
- n.trees
- shrinkage
- n.minobsinnode


```
## Stochastic Gradient Boosting 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 357, 357, 358, 358, 357, 357, ... 
## 
## Resampling results across tuning parameters:
## 
##   shrinkage  interaction.depth  n.trees  Accuracy   Kappa      Accuracy SD
##   0.05       1                   10      0.5973077  0.1518015  0.06773457 
##   0.05       1                   20      0.6173077  0.2235062  0.05856494 
##   0.05       1                   30      0.6122436  0.2158218  0.06377799 
##   0.05       1                   40      0.6021795  0.1925966  0.05501396 
##   0.05       1                   50      0.6046795  0.1963058  0.06566963 
##   0.05       1                   60      0.6047436  0.1964582  0.07619983 
##   0.05       1                   70      0.6072436  0.2029396  0.06342379 
##   0.05       1                   80      0.5973718  0.1831738  0.07517873 
##   0.05       1                   90      0.6149359  0.2157694  0.07185339 
##   0.05       1                  100      0.6098077  0.2079300  0.06853817 
##   0.05       3                   10      0.6220513  0.2135610  0.06086946 
##   0.05       3                   20      0.6399359  0.2600604  0.08140944 
##   0.05       3                   30      0.6300000  0.2407770  0.08747217 
##   0.05       3                   40      0.6327564  0.2474576  0.07983190 
##   0.05       3                   50      0.6176282  0.2176764  0.07496514 
##   0.05       3                   60      0.6175000  0.2188497  0.07524830 
##   0.05       3                   70      0.6175641  0.2174373  0.08388770 
##   0.05       3                   80      0.6326923  0.2481518  0.07449026 
##   0.05       3                   90      0.6352564  0.2548195  0.07063445 
##   0.05       3                  100      0.6352564  0.2531925  0.07630569 
##   0.05       5                   10      0.6070513  0.1815242  0.05930709 
##   0.05       5                   20      0.6147436  0.2071525  0.07422069 
##   0.05       5                   30      0.6073718  0.1939087  0.08177273 
##   0.05       5                   40      0.6075000  0.1974392  0.07078944 
##   0.05       5                   50      0.6124359  0.2083508  0.06160386 
##   0.05       5                   60      0.6124359  0.2082860  0.06177927 
##   0.05       5                   70      0.6225641  0.2308915  0.06760435 
##   0.05       5                   80      0.6124359  0.2104082  0.06381860 
##   0.05       5                   90      0.6099359  0.2055853  0.05806091 
##   0.05       5                  100      0.6174359  0.2217414  0.05417229 
##   0.10       1                   10      0.6121154  0.2148089  0.07632690 
##   0.10       1                   20      0.6148077  0.2203293  0.06886872 
##   0.10       1                   30      0.6174359  0.2237677  0.06804527 
##   0.10       1                   40      0.6122436  0.2134320  0.07386827 
##   0.10       1                   50      0.6123077  0.2113775  0.06350347 
##   0.10       1                   60      0.6223718  0.2319038  0.06346942 
##   0.10       1                   70      0.6175000  0.2234836  0.07735223 
##   0.10       1                   80      0.6275641  0.2429483  0.08245649 
##   0.10       1                   90      0.6175000  0.2239230  0.06548164 
##   0.10       1                  100      0.6149359  0.2170603  0.06920640 
##   0.10       3                   10      0.6271795  0.2350472  0.05126086 
##   0.10       3                   20      0.6249359  0.2311376  0.06344639 
##   0.10       3                   30      0.6274359  0.2369637  0.05967915 
##   0.10       3                   40      0.6224359  0.2311677  0.06289605 
##   0.10       3                   50      0.6276282  0.2450167  0.06940824 
##   0.10       3                   60      0.6226923  0.2341787  0.07004976 
##   0.10       3                   70      0.6200641  0.2283461  0.06583558 
##   0.10       3                   80      0.6224359  0.2347157  0.05581547 
##   0.10       3                   90      0.6174359  0.2242430  0.04870183 
##   0.10       3                  100      0.6150000  0.2189853  0.06462974 
##   0.10       5                   10      0.6121795  0.2004701  0.08355439 
##   0.10       5                   20      0.6072436  0.1963056  0.06585153 
##   0.10       5                   30      0.6400641  0.2657305  0.06195715 
##   0.10       5                   40      0.6175641  0.2209108  0.06838154 
##   0.10       5                   50      0.6225641  0.2339035  0.06225676 
##   0.10       5                   60      0.6201282  0.2285402  0.07354849 
##   0.10       5                   70      0.6276282  0.2442296  0.07050405 
##   0.10       5                   80      0.6225641  0.2351621  0.07066986 
##   0.10       5                   90      0.6074359  0.2038331  0.06994344 
##   0.10       5                  100      0.6099359  0.2118329  0.06376138 
##   0.15       1                   10      0.6124359  0.2098011  0.06040640 
##   0.15       1                   20      0.6225000  0.2342892  0.07841528 
##   0.15       1                   30      0.6100641  0.2048330  0.07899457 
##   0.15       1                   40      0.6150000  0.2187764  0.06017847 
##   0.15       1                   50      0.6148077  0.2141507  0.06026433 
##   0.15       1                   60      0.6150000  0.2175862  0.06457391 
##   0.15       1                   70      0.6047436  0.1966870  0.06325234 
##   0.15       1                   80      0.6175000  0.2240526  0.06115148 
##   0.15       1                   90      0.6100000  0.2074214  0.06874115 
##   0.15       1                  100      0.6123718  0.2144296  0.06305631 
##   0.15       3                   10      0.6147436  0.2097563  0.07007049 
##   0.15       3                   20      0.6075641  0.1993369  0.06408960 
##   0.15       3                   30      0.6251923  0.2366931  0.07485165 
##   0.15       3                   40      0.6175641  0.2230998  0.05598062 
##   0.15       3                   50      0.6226282  0.2328859  0.07781067 
##   0.15       3                   60      0.6225641  0.2324567  0.07920280 
##   0.15       3                   70      0.6098077  0.2055076  0.08031633 
##   0.15       3                   80      0.6198077  0.2269422  0.04158764 
##   0.15       3                   90      0.6199359  0.2273101  0.05631017 
##   0.15       3                  100      0.6173077  0.2229215  0.06595265 
##   0.15       5                   10      0.6098718  0.2032867  0.04782622 
##   0.15       5                   20      0.6227564  0.2306615  0.07743435 
##   0.15       5                   30      0.6076282  0.2017731  0.07934371 
##   0.15       5                   40      0.6278205  0.2428786  0.07825485 
##   0.15       5                   50      0.6151282  0.2178484  0.07626451 
##   0.15       5                   60      0.5974359  0.1805623  0.06509488 
##   0.15       5                   70      0.5947436  0.1765520  0.06120652 
##   0.15       5                   80      0.5946154  0.1774600  0.05699419 
##   0.15       5                   90      0.5821154  0.1559756  0.04664433 
##   0.15       5                  100      0.6047436  0.2022871  0.06792735 
##   Kappa SD  
##   0.16267587
##   0.11832805
##   0.12864135
##   0.10865738
##   0.13285257
##   0.15370697
##   0.12590323
##   0.14795195
##   0.14376112
##   0.13548941
##   0.12720545
##   0.17044294
##   0.18020991
##   0.16125726
##   0.15213186
##   0.14906728
##   0.16865425
##   0.14765597
##   0.14033488
##   0.15287891
##   0.12913718
##   0.15740548
##   0.17381135
##   0.14967193
##   0.13060727
##   0.13170134
##   0.13524964
##   0.12755222
##   0.11639421
##   0.11090354
##   0.15191914
##   0.13412674
##   0.13246153
##   0.14621800
##   0.12386219
##   0.12504468
##   0.14954574
##   0.16112592
##   0.12921071
##   0.13585698
##   0.10658053
##   0.12609708
##   0.12102341
##   0.12310268
##   0.13452229
##   0.13678991
##   0.12770340
##   0.10833553
##   0.09650533
##   0.12850872
##   0.17640808
##   0.13897932
##   0.12752701
##   0.14143533
##   0.12914496
##   0.14753676
##   0.14135877
##   0.13956935
##   0.13666273
##   0.12201225
##   0.12398555
##   0.15296280
##   0.15629337
##   0.11541578
##   0.12065542
##   0.12410695
##   0.12586649
##   0.12115436
##   0.13786444
##   0.12794765
##   0.13659229
##   0.12989822
##   0.14810261
##   0.11130124
##   0.15410895
##   0.16110648
##   0.16180588
##   0.08304413
##   0.11356144
##   0.13498043
##   0.09871571
##   0.15510393
##   0.15895377
##   0.16057502
##   0.15206187
##   0.13521405
##   0.12771356
##   0.11918713
##   0.09084498
##   0.13861916
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 30, interaction.depth
##  = 5, shrinkage = 0.1 and n.minobsinnode = 10.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  50  27
##        yes 22  70
##                                           
##                Accuracy : 0.7101          
##                  95% CI : (0.6354, 0.7772)
##     No Information Rate : 0.574           
##     P-Value [Acc > NIR] : 0.0001808       
##                                           
##                   Kappa : 0.4124          
##  Mcnemar's Test P-Value : 0.5677092       
##                                           
##             Sensitivity : 0.6944          
##             Specificity : 0.7216          
##          Pos Pred Value : 0.6494          
##          Neg Pred Value : 0.7609          
##              Prevalence : 0.4260          
##          Detection Rate : 0.2959          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.7080          
##                                           
##        'Positive' Class : no              
##                                           
## [1] "Model gbm2 gives an accuracy of 0.7101, an improvement over the baseline of 29.9%"
```

The accuracy has come down slightly to 0.71. Of the boosting models in hand, the gbm1 model performs the best. But its still worth tuning to seek further improvement.


```
## Stochastic Gradient Boosting 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 357, 357, 358, 358, 357, 357, ... 
## 
## Resampling results across tuning parameters:
## 
##   shrinkage  interaction.depth  n.trees  ROC        Sens       Spec     
##   0.05       1                   10      0.6125842  0.3055556  0.8259740
##   0.05       1                   20      0.6319144  0.4888889  0.7012987
##   0.05       1                   30      0.6315416  0.5055556  0.6690476
##   0.05       1                   40      0.6511724  0.4944444  0.6874459
##   0.05       1                   50      0.6589827  0.5388889  0.6554113
##   0.05       1                   60      0.6581770  0.5333333  0.6735931
##   0.05       1                   70      0.6596561  0.5222222  0.6642857
##   0.05       1                   80      0.6532468  0.5055556  0.6690476
##   0.05       1                   90      0.6541126  0.5000000  0.6917749
##   0.05       1                  100      0.6576178  0.5000000  0.6963203
##   0.05       3                   10      0.6550926  0.3777778  0.8114719
##   0.05       3                   20      0.6585798  0.4555556  0.7419913
##   0.05       3                   30      0.6596861  0.4777778  0.7103896
##   0.05       3                   40      0.6624880  0.4833333  0.7106061
##   0.05       3                   50      0.6582852  0.4944444  0.7058442
##   0.05       3                   60      0.6641414  0.5000000  0.7103896
##   0.05       3                   70      0.6656566  0.5055556  0.7012987
##   0.05       3                   80      0.6627826  0.5166667  0.7012987
##   0.05       3                   90      0.6643098  0.5277778  0.6876623
##   0.05       3                  100      0.6633237  0.5333333  0.6831169
##   0.05       5                   10      0.6394781  0.3833333  0.7837662
##   0.05       5                   20      0.6530964  0.4833333  0.7471861
##   0.05       5                   30      0.6527297  0.4944444  0.7056277
##   0.05       5                   40      0.6506253  0.4777778  0.6963203
##   0.05       5                   50      0.6529822  0.4833333  0.7051948
##   0.05       5                   60      0.6624218  0.5166667  0.6826840
##   0.05       5                   70      0.6598966  0.5166667  0.6779221
##   0.05       5                   80      0.6609067  0.5166667  0.6872294
##   0.05       5                   90      0.6593555  0.5388889  0.6690476
##   0.05       5                  100      0.6593555  0.5277778  0.6735931
##   0.10       1                   10      0.6158971  0.5055556  0.6867965
##   0.10       1                   20      0.6222884  0.5500000  0.6642857
##   0.10       1                   30      0.6483586  0.5166667  0.6735931
##   0.10       1                   40      0.6585077  0.5333333  0.6688312
##   0.10       1                   50      0.6601251  0.5333333  0.6922078
##   0.10       1                   60      0.6597403  0.5277778  0.6783550
##   0.10       1                   70      0.6674182  0.5000000  0.7196970
##   0.10       1                   80      0.6650734  0.5222222  0.7201299
##   0.10       1                   90      0.6684764  0.5166667  0.7153680
##   0.10       1                  100      0.6619108  0.5277778  0.6971861
##   0.10       3                   10      0.6522427  0.4555556  0.7335498
##   0.10       3                   20      0.6501744  0.4722222  0.7151515
##   0.10       3                   30      0.6429413  0.4722222  0.7151515
##   0.10       3                   40      0.6587061  0.4944444  0.6969697
##   0.10       3                   50      0.6517196  0.5277778  0.6922078
##   0.10       3                   60      0.6538721  0.5333333  0.6919913
##   0.10       3                   70      0.6605099  0.5222222  0.6874459
##   0.10       3                   80      0.6618446  0.5333333  0.6647186
##   0.10       3                   90      0.6620010  0.5222222  0.6874459
##   0.10       3                  100      0.6536195  0.5444444  0.6787879
##   0.10       5                   10      0.6596441  0.4666667  0.7290043
##   0.10       5                   20      0.6544913  0.4777778  0.7106061
##   0.10       5                   30      0.6640272  0.5000000  0.7103896
##   0.10       5                   40      0.6621934  0.5055556  0.6924242
##   0.10       5                   50      0.6496032  0.5222222  0.6831169
##   0.10       5                   60      0.6505772  0.5444444  0.6829004
##   0.10       5                   70      0.6591270  0.5444444  0.6826840
##   0.10       5                   80      0.6597042  0.5444444  0.6878788
##   0.10       5                   90      0.6601972  0.5333333  0.7062771
##   0.10       5                  100      0.6561207  0.5611111  0.6969697
##   0.15       1                   10      0.6385462  0.5222222  0.7095238
##   0.15       1                   20      0.6587843  0.5222222  0.7151515
##   0.15       1                   30      0.6523509  0.5166667  0.7151515
##   0.15       1                   40      0.6606902  0.5055556  0.6919913
##   0.15       1                   50      0.6635402  0.4944444  0.6922078
##   0.15       1                   60      0.6498016  0.5055556  0.6781385
##   0.15       1                   70      0.6683502  0.5166667  0.6924242
##   0.15       1                   80      0.6605459  0.5166667  0.6974026
##   0.15       1                   90      0.6661616  0.5166667  0.6878788
##   0.15       1                  100      0.6584476  0.5333333  0.6690476
##   0.15       3                   10      0.6518879  0.5111111  0.7333333
##   0.15       3                   20      0.6754028  0.4888889  0.7331169
##   0.15       3                   30      0.6772727  0.4833333  0.7238095
##   0.15       3                   40      0.6764310  0.5111111  0.7056277
##   0.15       3                   50      0.6656686  0.5055556  0.6880952
##   0.15       3                   60      0.6710077  0.5388889  0.6785714
##   0.15       3                   70      0.6653319  0.5166667  0.6831169
##   0.15       3                   80      0.6698413  0.5222222  0.7062771
##   0.15       3                   90      0.6596320  0.5277778  0.6790043
##   0.15       3                  100      0.6614959  0.5444444  0.6558442
##   0.15       5                   10      0.6567400  0.4722222  0.6867965
##   0.15       5                   20      0.6605940  0.5055556  0.7106061
##   0.15       5                   30      0.6519481  0.4888889  0.6831169
##   0.15       5                   40      0.6607263  0.5166667  0.6924242
##   0.15       5                   50      0.6563492  0.5500000  0.6740260
##   0.15       5                   60      0.6564334  0.5388889  0.6831169
##   0.15       5                   70      0.6542208  0.5611111  0.6688312
##   0.15       5                   80      0.6578403  0.5833333  0.6417749
##   0.15       5                   90      0.6626383  0.5611111  0.6556277
##   0.15       5                  100      0.6314574  0.5666667  0.6597403
##   ROC SD      Sens SD     Spec SD   
##   0.09754151  0.21316416  0.15685698
##   0.10944676  0.13557782  0.15377955
##   0.10949022  0.09240722  0.11842484
##   0.07636597  0.11249143  0.12000880
##   0.06931316  0.09816562  0.10110694
##   0.06896421  0.09868824  0.10450957
##   0.06800246  0.09147473  0.09657578
##   0.07332887  0.08050765  0.10289142
##   0.06831994  0.08685955  0.09196347
##   0.07487291  0.08281733  0.09737347
##   0.06253883  0.12776436  0.10848190
##   0.06470912  0.14768446  0.12134861
##   0.06667032  0.13907395  0.11350195
##   0.06449989  0.13870358  0.09732534
##   0.06804955  0.11843168  0.09498332
##   0.06306696  0.13353894  0.08455299
##   0.05911045  0.10940041  0.09965072
##   0.05194033  0.11126533  0.09965072
##   0.05315155  0.13159880  0.10489365
##   0.05435212  0.13146844  0.11099955
##   0.05622026  0.12129276  0.09110614
##   0.05890567  0.12015651  0.10754379
##   0.05867217  0.11549975  0.08751896
##   0.06200986  0.11475506  0.09076611
##   0.05623330  0.12015651  0.09638882
##   0.06104045  0.12843364  0.11224010
##   0.05256501  0.13620871  0.09561341
##   0.05579386  0.09816562  0.09527771
##   0.05224300  0.07430519  0.11019586
##   0.05576784  0.09886184  0.13239804
##   0.09777180  0.15811388  0.14212301
##   0.09840415  0.10294031  0.11278259
##   0.07545480  0.09091065  0.11374477
##   0.06967845  0.10210406  0.11841407
##   0.06560046  0.10540926  0.09953468
##   0.06791160  0.09532991  0.10368646
##   0.06493651  0.11415581  0.11629486
##   0.06817270  0.12614360  0.11924663
##   0.07206747  0.10159901  0.11197101
##   0.06473804  0.09532991  0.10940411
##   0.08842469  0.16101530  0.12412098
##   0.07623931  0.11785113  0.10594618
##   0.07702374  0.11490439  0.11823458
##   0.08129195  0.11843168  0.09863736
##   0.06868897  0.12627946  0.10559478
##   0.06578496  0.11770555  0.10861426
##   0.06447620  0.11475506  0.09035886
##   0.05750561  0.12614360  0.09545894
##   0.06217677  0.10861391  0.11770063
##   0.06106250  0.10075163  0.11720423
##   0.07604280  0.16604824  0.10038354
##   0.07090443  0.14151832  0.08897043
##   0.05951285  0.15713484  0.10557999
##   0.05327644  0.16239379  0.14431980
##   0.06148010  0.14392118  0.13302720
##   0.06569085  0.11653432  0.12091650
##   0.05631974  0.14296488  0.10915521
##   0.06317693  0.12776436  0.11707358
##   0.06229844  0.14151832  0.11773977
##   0.06213611  0.13468960  0.09362845
##   0.09004970  0.14151832  0.12438220
##   0.08371770  0.15537909  0.12987895
##   0.11158530  0.10813927  0.12792434
##   0.07216172  0.09240722  0.10647967
##   0.06842927  0.09604669  0.11046645
##   0.10156512  0.11843168  0.09961963
##   0.06811441  0.07878536  0.08827967
##   0.06841464  0.08302412  0.11223361
##   0.06485432  0.08705674  0.11387284
##   0.06195722  0.08364141  0.10594741
##   0.11494635  0.11355341  0.09706156
##   0.07034902  0.11049210  0.09812436
##   0.06182000  0.12015651  0.08592344
##   0.06309208  0.12776436  0.11229018
##   0.05185213  0.12682143  0.12297069
##   0.05752826  0.12573515  0.10991585
##   0.06349083  0.10492011  0.11344414
##   0.05275190  0.11475506  0.10837916
##   0.04541506  0.11490439  0.11700353
##   0.04306806  0.11049210  0.09963504
##   0.06739692  0.13671132  0.09981800
##   0.06288738  0.11249143  0.11163484
##   0.05359752  0.10075163  0.10551785
##   0.06200265  0.11430592  0.10920409
##   0.05095122  0.12408789  0.10074796
##   0.05275303  0.07878536  0.10239313
##   0.04268601  0.09240722  0.10396122
##   0.05229931  0.09886184  0.10360132
##   0.05435175  0.14451565  0.12101204
##   0.09801921  0.10734353  0.08069720
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## ROC was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 30, interaction.depth
##  = 3, shrinkage = 0.15 and n.minobsinnode = 10.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  53  24
##        yes 22  70
##                                           
##                Accuracy : 0.7278          
##                  95% CI : (0.6541, 0.7933)
##     No Information Rate : 0.5562          
##     P-Value [Acc > NIR] : 3.183e-06       
##                                           
##                   Kappa : 0.4501          
##  Mcnemar's Test P-Value : 0.8828          
##                                           
##             Sensitivity : 0.7067          
##             Specificity : 0.7447          
##          Pos Pred Value : 0.6883          
##          Neg Pred Value : 0.7609          
##              Prevalence : 0.4438          
##          Detection Rate : 0.3136          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.7257          
##                                           
##        'Positive' Class : no              
##                                           
## [1] "Model gbm3 gives an accuracy of 0.7278, an improvement over the baseline of 33.15%"
```

This boosting model gives almost as much accuracy as the best model we have had yet. To get a sense of the performance of few other popolar prediction algorithms, like Support Vector Machines, Regularized Discriminant Analysis, K nearest neighbours, Neural Network and Naive Bayes. We wont spend much time into tuning the models, instead the objective is to familarize its use for predictions.

#### Support Vector Machines

Lets begin with SVM.


```
## Support Vector Machines with Radial Basis Function Kernel 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 397, 397, 397, 397, 397, 397, ... 
## 
## Resampling results across tuning parameters:
## 
##   C     Accuracy   Kappa      Accuracy SD  Kappa SD  
##   0.25  0.6133543  0.2089126  0.03255269   0.06744128
##   0.50  0.6111554  0.2058083  0.02981929   0.06070347
##   1.00  0.6075006  0.1998183  0.03138355   0.06182042
## 
## Tuning parameter 'sigma' was held constant at a value of 0.01455909
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were sigma = 0.01455909 and C = 0.25.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  26  51
##        yes 16  76
##                                           
##                Accuracy : 0.6036          
##                  95% CI : (0.5256, 0.6778)
##     No Information Rate : 0.7515          
##     P-Value [Acc > NIR] : 1               
##                                           
##                   Kappa : 0.1701          
##  Mcnemar's Test P-Value : 3.271e-05       
##                                           
##             Sensitivity : 0.6190          
##             Specificity : 0.5984          
##          Pos Pred Value : 0.3377          
##          Neg Pred Value : 0.8261          
##              Prevalence : 0.2485          
##          Detection Rate : 0.1538          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.6087          
##                                           
##        'Positive' Class : no              
##                                           
## [1] "Model svm gives an accuracy of 0.6036, an improvement over the baseline of 10.42%"
```

The untuned SVM model gives a modest improvement in prediction accuracy over the baseline. Lets begin with 10-fold CV and scaling, centering to improve the accuracy.


```
## Support Vector Machines with Radial Basis Function Kernel 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## Pre-processing: centered, scaled 
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 357, 357, 358, 358, 357, 357, ... 
## 
## Resampling results across tuning parameters:
## 
##   C     ROC        Sens       Spec       ROC SD      Sens SD    Spec SD   
##   0.25  0.6767316  0.5944444  0.6649351  0.05560034  0.1049201  0.11043817
##   0.50  0.6759139  0.5000000  0.7383117  0.05333753  0.1385799  0.09914507
##   1.00  0.6711039  0.4888889  0.7560606  0.03826464  0.1588712  0.07756082
## 
## Tuning parameter 'sigma' was held constant at a value of 0.01372188
## ROC was used to select the optimal model using  the largest value.
## The final values used for the model were sigma = 0.01372188 and C = 0.25.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  52  25
##        yes 36  56
##                                           
##                Accuracy : 0.6391          
##                  95% CI : (0.5617, 0.7114)
##     No Information Rate : 0.5207          
##     P-Value [Acc > NIR] : 0.001249        
##                                           
##                   Kappa : 0.2808          
##  Mcnemar's Test P-Value : 0.200415        
##                                           
##             Sensitivity : 0.5909          
##             Specificity : 0.6914          
##          Pos Pred Value : 0.6753          
##          Neg Pred Value : 0.6087          
##              Prevalence : 0.5207          
##          Detection Rate : 0.3077          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.6411          
##                                           
##        'Positive' Class : no              
##                                           
## [1] "Model svm1 gives an accuracy of 0.6391, an improvement over the baseline of 16.91%"
```

This gives an increase in the accuracy, but its still lagging behind the accuracies of well tuned tree or boosting models. Lets use the tuneLength parameter to get an estimate of the 'C' parameter value.


```
## Support Vector Machines with Radial Basis Function Kernel 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## Pre-processing: centered, scaled 
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 357, 357, 358, 358, 357, 357, ... 
## 
## Resampling results across tuning parameters:
## 
##   C        ROC        Sens       Spec       ROC SD      Sens SD  
##      0.25  0.6767316  0.5833333  0.6601732  0.05560034  0.1118801
##      0.50  0.6759139  0.5000000  0.7385281  0.05333753  0.1283001
##      1.00  0.6711039  0.4888889  0.7471861  0.03826464  0.1544937
##      2.00  0.6584776  0.4722222  0.7655844  0.04638002  0.1640745
##      4.00  0.6380351  0.4611111  0.7471861  0.06255246  0.1698260
##      8.00  0.5931337  0.4666667  0.7471861  0.10271874  0.1462846
##     16.00  0.6018398  0.4111111  0.7606061  0.08676926  0.1086139
##     32.00  0.6016234  0.4000000  0.7235931  0.07105305  0.1165343
##     64.00  0.5980519  0.4388889  0.6824675  0.07238782  0.1559299
##    128.00  0.5873377  0.4222222  0.6915584  0.08342341  0.1760721
##    256.00  0.5878187  0.4333333  0.7051948  0.08171143  0.1631312
##    512.00  0.5875902  0.3722222  0.7417749  0.08369282  0.1201565
##   1024.00  0.5873256  0.4055556  0.7095238  0.08333488  0.1505591
##   2048.00  0.5875902  0.4277778  0.6779221  0.08369282  0.1482638
##   4096.00  0.5878427  0.4000000  0.7147186  0.08327920  0.1566977
##   8192.00  0.5880952  0.4222222  0.6963203  0.08331462  0.1509004
##   Spec SD   
##   0.10909225
##   0.11017862
##   0.09697571
##   0.11079700
##   0.10816641
##   0.09089649
##   0.08167439
##   0.08361972
##   0.10809058
##   0.12895400
##   0.10995633
##   0.08978583
##   0.12081981
##   0.11661470
##   0.12069480
##   0.11801357
## 
## Tuning parameter 'sigma' was held constant at a value of 0.01372188
## ROC was used to select the optimal model using  the largest value.
## The final values used for the model were sigma = 0.01372188 and C = 0.25.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  52  25
##        yes 35  57
##                                           
##                Accuracy : 0.645           
##                  95% CI : (0.5678, 0.7169)
##     No Information Rate : 0.5148          
##     P-Value [Acc > NIR] : 0.0004286       
##                                           
##                   Kappa : 0.2918          
##  Mcnemar's Test P-Value : 0.2452781       
##                                           
##             Sensitivity : 0.5977          
##             Specificity : 0.6951          
##          Pos Pred Value : 0.6753          
##          Neg Pred Value : 0.6196          
##              Prevalence : 0.5148          
##          Detection Rate : 0.3077          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.6464          
##                                           
##        'Positive' Class : no              
##                                           
## [1] "Model svm2 gives an accuracy of 0.645, an improvement over the baseline of 18%"
```

This again, has improved the accuracy by about 1%, but its still well behind the best models. We now check by manual tuning of the parameters to explore further improvements.


```
## Support Vector Machines with Radial Basis Function Kernel 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## Pre-processing: centered, scaled 
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 357, 357, 358, 358, 357, 357, ... 
## 
## Resampling results across tuning parameters:
## 
##   sigma  C     ROC        Sens       Spec       ROC SD      Sens SD   
##   0.001  0.05  0.6935786  0.5944444  0.6651515  0.05572458  0.09091065
##   0.001  0.10  0.6935907  0.5944444  0.6744589  0.05637977  0.09091065
##   0.001  0.15  0.6940957  0.5944444  0.6835498  0.05596321  0.08302412
##   0.001  0.20  0.6946128  0.5944444  0.6696970  0.05586476  0.08705674
##   0.001  0.25  0.6943603  0.6000000  0.6790043  0.05620007  0.08606630
##   0.001  0.30  0.6946128  0.6055556  0.6603896  0.05586476  0.09240722
##   0.001  0.35  0.6943603  0.5944444  0.6699134  0.05620007  0.09091065
##   0.001  0.40  0.6946128  0.5777778  0.6785714  0.05586476  0.08364141
##   0.001  0.45  0.6946128  0.5944444  0.6696970  0.05586476  0.10159901
##   0.001  0.50  0.6946128  0.5944444  0.6649351  0.05586476  0.09091065
##   0.010  0.05  0.6808201  0.5833333  0.6701299  0.05633375  0.10227186
##   0.010  0.10  0.6808201  0.5833333  0.6792208  0.05633375  0.10227186
##   0.010  0.15  0.6808201  0.5833333  0.6744589  0.05633375  0.10227186
##   0.010  0.20  0.6808201  0.5944444  0.6608225  0.05633375  0.10492011
##   0.010  0.25  0.6808201  0.5833333  0.6746753  0.05633375  0.10227186
##   0.010  0.30  0.6813492  0.5611111  0.7110390  0.05697262  0.11549975
##   0.010  0.35  0.6790043  0.5333333  0.7246753  0.05446319  0.12614360
##   0.010  0.40  0.6787157  0.5333333  0.7339827  0.05533154  0.12883353
##   0.010  0.45  0.6774772  0.5111111  0.7430736  0.05415181  0.12776436
##   0.010  0.50  0.6772006  0.5166667  0.7430736  0.05418430  0.13620871
##   0.100  0.05  0.5946489  0.2277778  0.8569264  0.06187653  0.13721210
##   0.100  0.10  0.5941318  0.2166667  0.8616883  0.06185088  0.09604669
##   0.100  0.15  0.5941318  0.2111111  0.8571429  0.06211435  0.11049210
##   0.100  0.20  0.5941438  0.2277778  0.8391775  0.06248510  0.15146744
##   0.100  0.25  0.5936388  0.1888889  0.8722944  0.06270263  0.09868824
##   0.100  0.30  0.5931337  0.2388889  0.8582251  0.06313014  0.14356331
##   0.100  0.35  0.5920875  0.2166667  0.8344156  0.06244458  0.08050765
##   0.100  0.40  0.5923280  0.2277778  0.8482684  0.06152028  0.08861842
##   0.100  0.45  0.5933622  0.2222222  0.8616883  0.06194309  0.11712139
##   0.100  0.50  0.5938793  0.1777778  0.8991342  0.06169048  0.12227833
##   Spec SD   
##   0.13236363
##   0.12731248
##   0.12771093
##   0.12042450
##   0.12661073
##   0.12536041
##   0.12783091
##   0.11401652
##   0.12601377
##   0.12149438
##   0.13770725
##   0.13486777
##   0.11540945
##   0.14305760
##   0.13721817
##   0.10727942
##   0.11376307
##   0.12519420
##   0.11467573
##   0.11467573
##   0.07964303
##   0.05717136
##   0.07671694
##   0.10787847
##   0.10193352
##   0.09899689
##   0.07189167
##   0.06043791
##   0.06881506
##   0.07922853
## 
## ROC was used to select the optimal model using  the largest value.
## The final values used for the model were sigma = 0.001 and C = 0.2.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  53  24
##        yes 29  63
##                                           
##                Accuracy : 0.6864          
##                  95% CI : (0.6107, 0.7555)
##     No Information Rate : 0.5148          
##     P-Value [Acc > NIR] : 4.444e-06       
##                                           
##                   Kappa : 0.3711          
##  Mcnemar's Test P-Value : 0.5827          
##                                           
##             Sensitivity : 0.6463          
##             Specificity : 0.7241          
##          Pos Pred Value : 0.6883          
##          Neg Pred Value : 0.6848          
##              Prevalence : 0.4852          
##          Detection Rate : 0.3136          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.6852          
##                                           
##        'Positive' Class : no              
##                                           
## [1] "Model svm3 gives an accuracy of 0.6864, an improvement over the baseline of 25.57%"
```

This gives a respectable prediction accuracy of 0.68. We now move on to other methods.

# Regularized Discriminant Analysis

Lets look at RDA.


```
## Regularized Discriminant Analysis 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 357, 357, 358, 358, 357, 357, ... 
## 
## Resampling results across tuning parameters:
## 
##   gamma  lambda  ROC        Sens       Spec       ROC SD      Sens SD   
##   0.0    0.0     0.6648749  0.5777778  0.7339827  0.05303859  0.11475506
##   0.0    0.5     0.6563853  0.5611111  0.7248918  0.05857020  0.11843168
##   0.0    1.0     0.6671116  0.5777778  0.7248918  0.07141216  0.12058386
##   0.5    0.0     0.6731241  0.5500000  0.6831169  0.05038671  0.08466022
##   0.5    0.5     0.6663179  0.5555556  0.6967532  0.04679623  0.08685955
##   0.5    1.0     0.6727032  0.5611111  0.6740260  0.05574737  0.09240722
##   1.0    0.0     0.6731361  0.6222222  0.6415584  0.06609888  0.11944086
##   1.0    0.5     0.6728716  0.6222222  0.6415584  0.06590914  0.11944086
##   1.0    1.0     0.6728716  0.6222222  0.6415584  0.06590914  0.11944086
##   Spec SD   
##   0.09421291
##   0.09918706
##   0.10147517
##   0.11506161
##   0.10231303
##   0.12473493
##   0.10216409
##   0.10216409
##   0.10216409
## 
## ROC was used to select the optimal model using  the largest value.
## The final values used for the model were gamma = 1 and lambda = 0.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  57  20
##        yes 24  68
##                                          
##                Accuracy : 0.7396         
##                  95% CI : (0.6667, 0.804)
##     No Information Rate : 0.5207         
##     P-Value [Acc > NIR] : 4.512e-09      
##                                          
##                   Kappa : 0.4774         
##  Mcnemar's Test P-Value : 0.6511         
##                                          
##             Sensitivity : 0.7037         
##             Specificity : 0.7727         
##          Pos Pred Value : 0.7403         
##          Neg Pred Value : 0.7391         
##              Prevalence : 0.4793         
##          Detection Rate : 0.3373         
##    Detection Prevalence : 0.4556         
##       Balanced Accuracy : 0.7382         
##                                          
##        'Positive' Class : no             
##                                          
## [1] "Model rda gives an accuracy of 0.7396, an improvement over the baseline of 35.32%"
```

The rda model is providing accuracy of close to 0.6. Improvements to this might be possible with tuning of the gamma and lambda parameters. Displaying the tuning grid takes up too much space. SO we move to the confision matrix.


```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  50  27
##        yes 27  65
##                                         
##                Accuracy : 0.6805        
##                  95% CI : (0.6045, 0.75)
##     No Information Rate : 0.5444        
##     P-Value [Acc > NIR] : 0.0002144     
##                                         
##                   Kappa : 0.3559        
##  Mcnemar's Test P-Value : 1.0000000     
##                                         
##             Sensitivity : 0.6494        
##             Specificity : 0.7065        
##          Pos Pred Value : 0.6494        
##          Neg Pred Value : 0.7065        
##              Prevalence : 0.4556        
##          Detection Rate : 0.2959        
##    Detection Prevalence : 0.4556        
##       Balanced Accuracy : 0.6779        
##                                         
##        'Positive' Class : no            
##                                         
## [1] "Model rda1 gives an accuracy of 0.6805, an improvement over the baseline of 24.49%"
```

This leads to a drop in accuracy. It can be noted that the best gamma and lambda values gives improved training set predictions, but gives poor(er) test set predictions. This suggests possible overfitting.

#### K Nearest Neighbours

Lets proceed to KNN algorithm. Lets start with a 10 fold CV model with 'range' as an option for pre-processing

![plot of chunk knn](figure/knn-1.png) 

```
## k-Nearest Neighbors 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## Pre-processing: re-scaling to [0, 1] 
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 357, 357, 358, 358, 357, 357, ... 
## 
## Resampling results across tuning parameters:
## 
##   k  Accuracy   Kappa      Accuracy SD  Kappa SD  
##   5  0.6276923  0.2374072  0.08472043   0.17270343
##   7  0.6198077  0.2175716  0.05458509   0.11281346
##   9  0.6097436  0.1983742  0.04120864   0.08094304
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was k = 5.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  41  36
##        yes 21  71
##                                           
##                Accuracy : 0.6627          
##                  95% CI : (0.5861, 0.7335)
##     No Information Rate : 0.6331          
##     P-Value [Acc > NIR] : 0.23742         
##                                           
##                   Kappa : 0.3091          
##  Mcnemar's Test P-Value : 0.06369         
##                                           
##             Sensitivity : 0.6613          
##             Specificity : 0.6636          
##          Pos Pred Value : 0.5325          
##          Neg Pred Value : 0.7717          
##              Prevalence : 0.3669          
##          Detection Rate : 0.2426          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.6624          
##                                           
##        'Positive' Class : no              
##                                           
## [1] "Model knn gives an accuracy of 0.6627, an improvement over the baseline of 21.24%"
```

This gives a reasonable accuracy of 0.66. It could be possible to achieve more out of this model by tuning the value of k

![plot of chunk knn1](figure/knn1-1.png) 

```
## k-Nearest Neighbors 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## Pre-processing: re-scaling to [0, 1] 
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 357, 357, 358, 358, 357, 357, ... 
## 
## Resampling results across tuning parameters:
## 
##   k   Accuracy   Kappa      Accuracy SD  Kappa SD  
##    5  0.6124359  0.2042230  0.08424778   0.17485229
##    7  0.6299359  0.2396976  0.05400114   0.11069407
##    9  0.6173077  0.2094559  0.03872771   0.07836929
##   11  0.6173718  0.2104139  0.04646092   0.09236163
##   13  0.6224359  0.2217465  0.05886665   0.11725322
##   15  0.6149359  0.2057918  0.06172592   0.12016341
##   17  0.6301282  0.2335797  0.07016490   0.14065212
##   19  0.6250000  0.2215782  0.06702185   0.13758900
##   21  0.6377564  0.2470054  0.07030236   0.14530983
##   23  0.6176923  0.2021838  0.06780423   0.14126876
##   25  0.6301923  0.2287259  0.06852911   0.14253896
##   27  0.6301282  0.2296694  0.06475029   0.13207155
##   29  0.6351923  0.2404409  0.06281066   0.13113850
##   31  0.6326282  0.2353749  0.06716846   0.13636648
##   33  0.6376282  0.2455823  0.06419726   0.13105653
##   35  0.6325641  0.2337310  0.05731501   0.11560423
##   37  0.6325641  0.2344452  0.06197230   0.12584760
##   39  0.6401282  0.2488195  0.05847381   0.11949322
##   41  0.6375000  0.2449574  0.06468344   0.13094495
##   43  0.6401282  0.2509412  0.05721088   0.11754323
##   45  0.6426282  0.2561128  0.06152603   0.12511628
##   47  0.6376282  0.2437418  0.05829227   0.12170583
##   49  0.6350641  0.2388651  0.06213729   0.12937236
##   51  0.6301923  0.2292966  0.06647151   0.13825265
##   53  0.6300000  0.2295329  0.05953361   0.12143721
##   55  0.6274359  0.2231636  0.06507229   0.13284181
##   57  0.6349359  0.2375156  0.05170079   0.10561015
##   59  0.6223077  0.2115549  0.06039930   0.12505812
##   61  0.6248718  0.2181318  0.05821840   0.11989479
##   63  0.6223718  0.2124266  0.06661696   0.13801155
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was k = 45.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  36  41
##        yes 10  82
##                                          
##                Accuracy : 0.6982         
##                  95% CI : (0.623, 0.7663)
##     No Information Rate : 0.7278         
##     P-Value [Acc > NIR] : 0.8294         
##                                          
##                   Kappa : 0.371          
##  Mcnemar's Test P-Value : 2.659e-05      
##                                          
##             Sensitivity : 0.7826         
##             Specificity : 0.6667         
##          Pos Pred Value : 0.4675         
##          Neg Pred Value : 0.8913         
##              Prevalence : 0.2722         
##          Detection Rate : 0.2130         
##    Detection Prevalence : 0.4556         
##       Balanced Accuracy : 0.7246         
##                                          
##        'Positive' Class : no             
##                                          
## [1] "Model knn1 gives an accuracy of 0.6982, an improvement over the baseline of 27.74%"
```

Approaching the 'correct' value of k, we improve the prediction to close to 0.68. 

#### Conditional Inference Trees

We move on to the next method, the Conditional Inference Trees, called ctree. It can be thought of as the tree models with significance test procedure in order to select variables instead of selecting the variable that maximizes an information measure (e.g. Gini coefficient).


```
## Conditional Inference Tree 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## Pre-processing: centered, scaled 
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 357, 357, 358, 358, 357, 357, ... 
## 
## Resampling results across tuning parameters:
## 
##   mincriterion  ROC        Sens       Spec       ROC SD     Sens SD  
##   0.01          0.5613817  0.5166667  0.6251082  0.1105877  0.1229775
##   0.50          0.5866703  0.4944444  0.7515152  0.1186979  0.1445157
##   0.99          0.6039683  0.6333333  0.6411255  0.1206029  0.1233950
##   Spec SD   
##   0.15000554
##   0.07166358
##   0.09642122
## 
## ROC was used to select the optimal model using  the largest value.
## The final value used for the model was mincriterion = 0.99.
```

![plot of chunk ctree](figure/ctree-1.png) 

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  57  20
##        yes 24  68
##                                          
##                Accuracy : 0.7396         
##                  95% CI : (0.6667, 0.804)
##     No Information Rate : 0.5207         
##     P-Value [Acc > NIR] : 4.512e-09      
##                                          
##                   Kappa : 0.4774         
##  Mcnemar's Test P-Value : 0.6511         
##                                          
##             Sensitivity : 0.7037         
##             Specificity : 0.7727         
##          Pos Pred Value : 0.7403         
##          Neg Pred Value : 0.7391         
##              Prevalence : 0.4793         
##          Detection Rate : 0.3373         
##    Detection Prevalence : 0.4556         
##       Balanced Accuracy : 0.7382         
##                                          
##        'Positive' Class : no             
##                                          
## [1] "Model ctree gives an accuracy of 0.7396, an improvement over the baseline of 35.32%"
```
We see that this tree is strikingly similar to the one from the tree model, as one might expect!

This gives close to 0.73 accuracy. It may be worth investigating further tuning of the model using the mincriterion parameter. We wont cover it here. 

#### C 4.5

We move on to what was the #1 in the [Top 10 Algorithms in Data Mining pre-eminent paper published by Springer LNCS in 2008]( #1 in the Top 10 Algorithms in Data Mining pre-eminent paper published by Springer LNCS in 200), C4.5


```
## C4.5-like Trees 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 357, 357, 358, 358, 357, 357, ... 
## 
## Resampling results across tuning parameters:
## 
##   C     ROC        Sens       Spec       ROC SD      Sens SD   
##   0.10  0.5306337  0.4277778  0.6461039  0.09229011  0.12573515
##   0.12  0.5660594  0.4333333  0.6316017  0.07084497  0.12505143
##   0.14  0.5607684  0.4333333  0.6316017  0.06181051  0.12505143
##   0.16  0.5573473  0.4222222  0.6409091  0.06554052  0.10861391
##   0.18  0.5364177  0.4500000  0.6129870  0.07196226  0.09955319
##   0.20  0.5598124  0.4500000  0.6270563  0.09392228  0.08861842
##   0.22  0.5555796  0.4500000  0.6270563  0.07950613  0.08861842
##   0.24  0.5665224  0.4555556  0.6409091  0.08579838  0.09728834
##   0.26  0.5630111  0.4777778  0.6272727  0.09157032  0.10540926
##   0.28  0.5671717  0.5055556  0.6134199  0.09070136  0.09240722
##   0.30  0.5624519  0.4944444  0.6043290  0.08365880  0.09240722
##   0.32  0.5879389  0.5000000  0.6090909  0.06109511  0.09072184
##   0.34  0.5667689  0.5055556  0.5906926  0.06274017  0.08861842
##   0.36  0.5713143  0.5000000  0.5952381  0.04931437  0.08685955
##   0.38  0.5694925  0.4944444  0.5997835  0.05356216  0.09955319
##   0.40  0.5699976  0.4944444  0.6043290  0.05305391  0.09955319
##   Spec SD   
##   0.12560434
##   0.09738817
##   0.09738817
##   0.08357863
##   0.06773505
##   0.07726495
##   0.07393218
##   0.07488642
##   0.06964482
##   0.06890389
##   0.07148904
##   0.08288910
##   0.08317279
##   0.07879237
##   0.08264538
##   0.08868767
## 
## ROC was used to select the optimal model using  the largest value.
## The final value used for the model was C = 0.32.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  39  38
##        yes 21  71
##                                           
##                Accuracy : 0.6509          
##                  95% CI : (0.5739, 0.7225)
##     No Information Rate : 0.645           
##     P-Value [Acc > NIR] : 0.47107         
##                                           
##                   Kappa : 0.2833          
##  Mcnemar's Test P-Value : 0.03725         
##                                           
##             Sensitivity : 0.6500          
##             Specificity : 0.6514          
##          Pos Pred Value : 0.5065          
##          Neg Pred Value : 0.7717          
##              Prevalence : 0.3550          
##          Detection Rate : 0.2308          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.6507          
##                                           
##        'Positive' Class : no              
##                                           
## [1] "Model c45 gives an accuracy of 0.6509, an improvement over the baseline of 19.08%"
```

This doesnt give as much accuracy as expected. The model may be further tuned by the 'C' parameter. 

#### Neural Network

Lets move on to Neural Networks. We begin with a no-tune model and work our way from there depending on the accuracy of this untuned model.


```
## Neural Network 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 357, 357, 358, 358, 357, 357, ... 
## 
## Resampling results across tuning parameters:
## 
##   size  decay  Accuracy   Kappa      Accuracy SD  Kappa SD  
##   1     0e+00  0.6370513  0.2662408  0.07363448   0.12985620
##   1     1e-04  0.6175641  0.2351257  0.07038333   0.13415323
##   1     1e-01  0.6250641  0.2348198  0.05485043   0.11495261
##   3     0e+00  0.6270513  0.2306810  0.06884773   0.14580106
##   3     1e-04  0.6172436  0.2210186  0.03746751   0.08151291
##   3     1e-01  0.6070513  0.2044527  0.05435588   0.11731785
##   5     0e+00  0.5896795  0.1776232  0.06111518   0.12183399
##   5     1e-04  0.5794231  0.1471939  0.04918379   0.10330557
##   5     1e-01  0.5891026  0.1676907  0.06139585   0.11927074
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were size = 1 and decay = 0.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  43  34
##        yes 22  70
##                                          
##                Accuracy : 0.6686         
##                  95% CI : (0.5922, 0.739)
##     No Information Rate : 0.6154         
##     P-Value [Acc > NIR] : 0.08857        
##                                          
##                   Kappa : 0.3234         
##  Mcnemar's Test P-Value : 0.14158        
##                                          
##             Sensitivity : 0.6615         
##             Specificity : 0.6731         
##          Pos Pred Value : 0.5584         
##          Neg Pred Value : 0.7609         
##              Prevalence : 0.3846         
##          Detection Rate : 0.2544         
##    Detection Prevalence : 0.4556         
##       Balanced Accuracy : 0.6673         
##                                          
##        'Positive' Class : no             
##                                          
## [1] "Model nn gives an accuracy of 0.6686, an improvement over the baseline of 22.33%"
```

Its worth using resampling to reduce overfitting and improve accuracy on the test set.

Using 10-fold CV as resampling...


```
## Neural Network 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 357, 357, 358, 358, 357, 357, ... 
## 
## Resampling results across tuning parameters:
## 
##   size  decay  ROC        Sens       Spec       ROC SD      Sens SD  
##   1     0e+00  0.5380772  0.5166667  0.6456710  0.11512734  0.1636559
##   1     1e-04  0.5842352  0.5722222  0.6071429  0.11251940  0.2160406
##   1     1e-01  0.6766955  0.5222222  0.7244589  0.04400099  0.1462846
##   3     0e+00  0.6235810  0.6722222  0.5402597  0.11354593  0.1581139
##   3     1e-04  0.6389971  0.5444444  0.6683983  0.08976527  0.1829495
##   3     1e-01  0.6359548  0.5500000  0.6038961  0.04368420  0.1184317
##   5     0e+00  0.5887205  0.5055556  0.6584416  0.10515187  0.1294973
##   5     1e-04  0.6632215  0.5611111  0.6859307  0.09213122  0.1537149
##   5     1e-01  0.6402958  0.5222222  0.6173160  0.06124763  0.1855552
##   Spec SD  
##   0.1944469
##   0.2454253
##   0.1142619
##   0.1197961
##   0.1338410
##   0.1326934
##   0.1478313
##   0.1412943
##   0.1081664
## 
## ROC was used to select the optimal model using  the largest value.
## The final values used for the model were size = 1 and decay = 0.1.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  46  31
##        yes 22  70
##                                           
##                Accuracy : 0.6864          
##                  95% CI : (0.6107, 0.7555)
##     No Information Rate : 0.5976          
##     P-Value [Acc > NIR] : 0.01068         
##                                           
##                   Kappa : 0.3617          
##  Mcnemar's Test P-Value : 0.27182         
##                                           
##             Sensitivity : 0.6765          
##             Specificity : 0.6931          
##          Pos Pred Value : 0.5974          
##          Neg Pred Value : 0.7609          
##              Prevalence : 0.4024          
##          Detection Rate : 0.2722          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.6848          
##                                           
##        'Positive' Class : no              
##                                           
## [1] "Model nn1 gives an accuracy of 0.6864, an improvement over the baseline of 25.57%"
```

This gives improvement, but is worth checking if the model can be pushed further.

The output of the next model is too large to be helpful in this context. So we look at the confusion matrix directly.


```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  46  31
##        yes 22  70
##                                           
##                Accuracy : 0.6864          
##                  95% CI : (0.6107, 0.7555)
##     No Information Rate : 0.5976          
##     P-Value [Acc > NIR] : 0.01068         
##                                           
##                   Kappa : 0.3617          
##  Mcnemar's Test P-Value : 0.27182         
##                                           
##             Sensitivity : 0.6765          
##             Specificity : 0.6931          
##          Pos Pred Value : 0.5974          
##          Neg Pred Value : 0.7609          
##              Prevalence : 0.4024          
##          Detection Rate : 0.2722          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.6848          
##                                           
##        'Positive' Class : no              
##                                           
## [1] "Model nn2 gives an accuracy of 0.6864, an improvement over the baseline of 25.57%"
```

This gives a prediction accuracy of close to 0.68. 

#### Naive Bayes

Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.


```
## Naive Bayes 
## 
## 397 samples
##   6 predictor
##   2 classes: 'no', 'yes' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 357, 357, 358, 358, 357, 357, ... 
## 
## Resampling results across tuning parameters:
## 
##   usekernel  ROC        Sens        Spec       ROC SD      Sens SD   
##   FALSE      0.6723614  0.55555556  0.7495362  0.05102162  0.11111111
##    TRUE      0.6837903  0.03333333  0.9816017  0.03974696  0.05972043
##   Spec SD   
##   0.10552708
##   0.03199512
## 
## Tuning parameter 'fL' was held constant at a value of 0
## ROC was used to select the optimal model using  the largest value.
## The final values used for the model were fL = 0 and usekernel = TRUE.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no   0  77
##        yes  0  92
##                                          
##                Accuracy : 0.5444         
##                  95% CI : (0.4661, 0.621)
##     No Information Rate : 1              
##     P-Value [Acc > NIR] : 1              
##                                          
##                   Kappa : 0              
##  Mcnemar's Test P-Value : <2e-16         
##                                          
##             Sensitivity :     NA         
##             Specificity : 0.5444         
##          Pos Pred Value :     NA         
##          Neg Pred Value :     NA         
##              Prevalence : 0.0000         
##          Detection Rate : 0.0000         
##    Detection Prevalence : 0.4556         
##       Balanced Accuracy :     NA         
##                                          
##        'Positive' Class : no             
##                                          
## [1] "Model nb gives an accuracy of 0.5444, an improvement over the baseline of -0.41%"
```


Naive Bayes performs poorly. We wont try tuning the NB model.

At this stage, we have explored a number of models. The best prediction performance seems to be in the order of 0.7, with the best being 0.74 from the CART model. Its surprising that a CART model outperforms Random Forest and Boosting models. 

With a quick wrap-up of what we have seen in this part, lets end it here.

### End of Part 1

This brings us to the end of part 1. In this section, we have 

- established the motivation for the work

- a brief description of the US courts setup

- 'cleaned' summary of the cases heard by Justice Stevens during the years of 1994-2001 

- strategy for selecting the training and test set

- establishing baseline prediction values that would serve as the guide to good predictions

- simple to complex tree based models ranging from trees to C4.5, their tuning and predictions on test set

- Support Vector Machines, KNN, Regularized Discriminant Analysis, Neural Network and their tuning and predictions.

- setting up parallel processing

The next parts would focus on diving a bit deeper into the predictions of some of the models, select the best model in terms of overall accuracy, and accuracies if individual Issues, Circuits, Petitioner and Respondent, i.e. explore if we can conclude if a particular model does well for a particular situation. We would also focus more on interpreting the results of a prediction



#### Footnote 
If the document is too large to be knit as html via the 'Knit HTML' option in R Studio, as it was the case for me, we can either

- add this snippet to the introduction section of the rmd file.
Output:
  html_document:
    pandoc_args: [
      "+RTS", "-K64m",
      "-RTS"
    ]

- or use the following code snippet on the r console:
library(knitr)
knit2html('courts.Rmd')



### Part 2: Interpretting the results and understanding the decisions 

In part 1 of this series, we discussed:

- motivation for the work

- brief description of the US courts setup.

- 'cleaned' summary of the cases heard by Justice Stevens during the years of 1994-2001 

- strategy for selecting the training and test set

- establishing baseline prediction values that would serve as the guide to good predictions

- simple to complex tree based models ranging from trees to C4.5, their tuning and predictions on test set

- Support Vector Machines, KNN, Regularized Discriminant Analysis, Neural Network and their tuning and predictions.

- setting up parallel processing

In this section, we start looking at the results of the predictions, interpretting the results, visualizing the decisions, implications and trade-offs of probability threshold.

#### CART Prediction Results

We started off with CART for its ease of interpretation. The interpretation is made easy by appropriate visualization. As such, the CART trained model, 'tree' is not an 'rpart' object. To view the tree better, we use the 'rpart' library to create a tree object of rpart class and plot it ising the prp function of the rpart.plot package.

![plot of chunk tree_viz](figure/tree_viz-1.png) 

A more colourful way to represent the tree is offered by the 'rattle' package, as shown below.

![plot of chunk tree_color](figure/tree_color-1.png) 

We see two ways of visualizing the decisions. Both convey the same meaning, one being more colourful than the other.

While we attempt to do this visualization, its important to keep in mind that this visualization does not describe the ideologies and decision making process of the judge in this causal way. Instead it looks for patterns and explains the outcomes of the training set based on these patterns.

Lets look at interpreting the visualization. 

The first split or deicsion is on the ideological direction of the lower court, represented by variable 'LowerCourt' and takes either 'liberal' or 'conservative'. If the lower court's ideological direction is not liberal (conservative, which happens in 52% of the cases), then the model predicts that Justics Stevens will reverse the decision of the lower court. 

The remaining 48% of the cases whose lower court's ideological direction is liberal, reaches another decision point. If the Circuit of origin of the case. If the circuit is 2nd, 5th, 6th, 9th or the FED, the lower court ruling is reversed. This happens in approximately 27% of the training set cases. The cases from Circuits 1,3,4,7,8,10,11 and DC, which comprise 22% of the Train data, the lower court ruling is affirmed by Justics Stevens. 


The fractions in each of the box denotes the probability of a 'yes' or 'no' (reverse or affirm lower court decision). The probability of yes is given in the right and the probability of a no is given in the left side of the box.

#### Goodness of the model

An important parameter in evaluating the goodness of the model is the ROC curve. Receiver operating characteristic, or ROC curve, is a graphical plot that illustrates the performance of a binary classifier system as its discrimination threshold is varied. The curve is created by plotting the true positive rate against the false positive rate at various threshold settings. The true-positive rate is also known as sensitivity. False-positive rate can be calculated as (1 - specificity). 

The area under a ROC curve quantifies the overall ability of the test to discriminate between those cases whose lower court decision is to be reversed and those that are to be affirmed. A truly random prediction (one no better at identifying true positives than flipping a coin) has an area of 0.5. A perfect prediction (one that has zero false positives and zero false negatives) has an area of 1.00. Our AUC will be inbetween these numbers, but we would aspire to have the ROC curve 'hug' the north-west corner of the ROC plot to get higher AUC.

We use the ROCR package to get the ROC curve for this tree model.

![plot of chunk tree_roc](figure/tree_roc-1.png) 

```
## [1] "The Area under the ROC Curve- AUC of the untuned Tree mode 'tree' = 0.7397"
```

This nicely illustrates the concepts of sensitivity and specificity. A generalization of the model performance unser various combinations of Sensitivity and Specificity is the ROC. Lets try to look at a specific example to understand the Sensitivity and Specificity, in context of prediction accuracy.

#### Manual tweeks to Probability Thresholds: The Sensitivity-Specificity Trade off

Lets go back to the random forest models, rf, rf1 and rf2 from Part 1 of this series. One may recall that these models give lower accuracy than the CART models. In this section, we explore if the Sensitivity and Specificity can be manipulated to achieve better prediction accuracy. One tool to manipulate them with is the class probability threshold.

The random forest model select the prediction class based on the probability of the prediction for decision reversal. The default threshold for selecting a class is 0.5. So if the probability for a class is >= 0.5, we choose the corresponding class. So if the probability for reversal is >= 0.5, the model selects 'yes' for reversal.

We can try changing this default of 0.5 and see if this affects the prediction quality. It has to be kept in mind that this can affect, in some cases, severely, the sensitivity and specificity of the outcomes. Such manipulation of the probabilities can be helpful if one of the outcomes is more important (or carries financial penalties) than the other. But in out case, both the outcomes mean the same and we do not have a preference. But lets examine the performance if we relax the class selection probability by +/- 0.05.

Lets begin with rf. First we see the confusion matrix for rf1 with default settings, then the probability threshold set to 0.55, and then to 0.45


```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  50  27
##        yes 32  60
##                                           
##                Accuracy : 0.6509          
##                  95% CI : (0.5739, 0.7225)
##     No Information Rate : 0.5148          
##     P-Value [Acc > NIR] : 0.0002406       
##                                           
##                   Kappa : 0.2999          
##  Mcnemar's Test P-Value : 0.6025370       
##                                           
##             Sensitivity : 0.6098          
##             Specificity : 0.6897          
##          Pos Pred Value : 0.6494          
##          Neg Pred Value : 0.6522          
##              Prevalence : 0.4852          
##          Detection Rate : 0.2959          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.6497          
##                                           
##        'Positive' Class : no              
## 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  51  26
##        yes 37  55
##                                           
##                Accuracy : 0.6272          
##                  95% CI : (0.5496, 0.7003)
##     No Information Rate : 0.5207          
##     P-Value [Acc > NIR] : 0.003365        
##                                           
##                   Kappa : 0.2572          
##  Mcnemar's Test P-Value : 0.207712        
##                                           
##             Sensitivity : 0.5795          
##             Specificity : 0.6790          
##          Pos Pred Value : 0.6623          
##          Neg Pred Value : 0.5978          
##              Prevalence : 0.5207          
##          Detection Rate : 0.3018          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.6293          
##                                           
##        'Positive' Class : no              
## 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  46  31
##        yes 24  68
##                                           
##                Accuracy : 0.6746          
##                  95% CI : (0.5983, 0.7445)
##     No Information Rate : 0.5858          
##     P-Value [Acc > NIR] : 0.01106         
##                                           
##                   Kappa : 0.339           
##  Mcnemar's Test P-Value : 0.41849         
##                                           
##             Sensitivity : 0.6571          
##             Specificity : 0.6869          
##          Pos Pred Value : 0.5974          
##          Neg Pred Value : 0.7391          
##              Prevalence : 0.4142          
##          Detection Rate : 0.2722          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.6720          
##                                           
##        'Positive' Class : no              
## 
```

Now, the same for rf1...


```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  34  43
##        yes 12  80
##                                           
##                Accuracy : 0.6746          
##                  95% CI : (0.5983, 0.7445)
##     No Information Rate : 0.7278          
##     P-Value [Acc > NIR] : 0.9476          
##                                           
##                   Kappa : 0.3217          
##  Mcnemar's Test P-Value : 5.228e-05       
##                                           
##             Sensitivity : 0.7391          
##             Specificity : 0.6504          
##          Pos Pred Value : 0.4416          
##          Neg Pred Value : 0.8696          
##              Prevalence : 0.2722          
##          Detection Rate : 0.2012          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.6948          
##                                           
##        'Positive' Class : no              
## 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  49  28
##        yes 20  72
##                                           
##                Accuracy : 0.716           
##                  95% CI : (0.6416, 0.7826)
##     No Information Rate : 0.5917          
##     P-Value [Acc > NIR] : 0.0005352       
##                                           
##                   Kappa : 0.4226          
##  Mcnemar's Test P-Value : 0.3123214       
##                                           
##             Sensitivity : 0.7101          
##             Specificity : 0.7200          
##          Pos Pred Value : 0.6364          
##          Neg Pred Value : 0.7826          
##              Prevalence : 0.4083          
##          Detection Rate : 0.2899          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.7151          
##                                           
##        'Positive' Class : no              
## 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  28  49
##        yes 10  82
##                                           
##                Accuracy : 0.6509          
##                  95% CI : (0.5739, 0.7225)
##     No Information Rate : 0.7751          
##     P-Value [Acc > NIR] : 0.9999          
##                                           
##                   Kappa : 0.2659          
##  Mcnemar's Test P-Value : 7.53e-07        
##                                           
##             Sensitivity : 0.7368          
##             Specificity : 0.6260          
##          Pos Pred Value : 0.3636          
##          Neg Pred Value : 0.8913          
##              Prevalence : 0.2249          
##          Detection Rate : 0.1657          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.6814          
##                                           
##        'Positive' Class : no              
## 
```

and for rf2...


```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  38  39
##        yes 14  78
##                                           
##                Accuracy : 0.6864          
##                  95% CI : (0.6107, 0.7555)
##     No Information Rate : 0.6923          
##     P-Value [Acc > NIR] : 0.6025330       
##                                           
##                   Kappa : 0.3506          
##  Mcnemar's Test P-Value : 0.0009784       
##                                           
##             Sensitivity : 0.7308          
##             Specificity : 0.6667          
##          Pos Pred Value : 0.4935          
##          Neg Pred Value : 0.8478          
##              Prevalence : 0.3077          
##          Detection Rate : 0.2249          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.6987          
##                                           
##        'Positive' Class : no              
## 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  45  32
##        yes 20  72
##                                           
##                Accuracy : 0.6923          
##                  95% CI : (0.6168, 0.7609)
##     No Information Rate : 0.6154          
##     P-Value [Acc > NIR] : 0.02287         
##                                           
##                   Kappa : 0.3717          
##  Mcnemar's Test P-Value : 0.12715         
##                                           
##             Sensitivity : 0.6923          
##             Specificity : 0.6923          
##          Pos Pred Value : 0.5844          
##          Neg Pred Value : 0.7826          
##              Prevalence : 0.3846          
##          Detection Rate : 0.2663          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.6923          
##                                           
##        'Positive' Class : no              
## 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction no yes
##        no  24  53
##        yes  7  85
##                                           
##                Accuracy : 0.645           
##                  95% CI : (0.5678, 0.7169)
##     No Information Rate : 0.8166          
##     P-Value [Acc > NIR] : 1               
##                                           
##                   Kappa : 0.2477          
##  Mcnemar's Test P-Value : 6.267e-09       
##                                           
##             Sensitivity : 0.7742          
##             Specificity : 0.6159          
##          Pos Pred Value : 0.3117          
##          Neg Pred Value : 0.9239          
##              Prevalence : 0.1834          
##          Detection Rate : 0.1420          
##    Detection Prevalence : 0.4556          
##       Balanced Accuracy : 0.6951          
##                                           
##        'Positive' Class : no              
## 
```

We see the prediction accuracy increases to around 0.7 as we change the class probability threshold from 0.5 to 0.5 +/- 0.05. But as one might expect with this change, the specificity increases at the cost of sensitivity. 

Lets take a step aside and define sensitivity and specificity in context of this problem.

Specificity- the proportion of predicted reversals to actual reversals.

Sensitivity-  the proportion of predicted non-reversals to actual non-reversals.

It has to be kept in mind that this increase in threshold from 0.5 to 0.55 means the model can miss to identify the reversals, but improves on the correct prediction of non-reversals. A graphical of representing this is the ROC curve. So we see that resampling, tuning and manipulating the sensitivity-specificity, random forests gives predictions with an accuracy of close to 0.72

This brings us to the end of part 2 of this 3 part series. In this section we have covered:

- discussion of results of the predictions from part 1
- interpretting the results
- visualizing the decisions
- implications and trade-offs of class selection probability threshold

In the third and final part, we will look into model selection and concluse the study.


### Part 3: Model Selection and Conclusion

In part 2 of the series, we discussed the results of the predictions from part 1, interpretting the results, vizualizing the decisions and the sensitivity-specificity trade-off.

In this section we will look into model selection and conclude our study.

First, lets look at the summary of all the models with us right now.



| Model | Resampling |  Metric  | Training Set Accuracy | Test Set Accuracy |  Summary Function | PreProcessing | Manual Tune |
|:-----:|:----------:|:--------:|:---------------------:|:-----------------:|:-----------------:|:-------------:|:-----------:|
|  tree |   default  | accuracy |          0.62         |        0.74       |         no        |       no      |      no     |
| tree1 |   default  | accuracy |          0.64         |        0.74       |         no        |       no      |      no     |
| tree2 | 10-fold CV | accuracy |          0.64         |        0.74       |         no        |       no      |      no     |
|   rf  |   default  | accuracy |          0.62         |        0.66       |         no        |       no      |      no     |
|  rf1  | 10-fold CV | accuracy |          0.65         |        0.68       |         no        |       no      |      no     |
|  rf2  | 10-fold CV | accuracy |          0.62         |        0.69       |         no        |  scale+center |      no     |
|   cf  |   default  | accuracy |          0.62         |        0.73       |         no        |       no      |      no     |
|  cf1  | 10-fold CV |    ROC   |          0.67         |        0.73       | Two Class Summary |       no      |      no     |
|  gbm  |   default  | accuracy |          0.61         |        0.69       |         no        |       no      |      no     |
|  gbm1 | 10-fold CV | accuracy |          0.64         |        0.73       |         no        |       no      |      no     |
|  gbm2 | 10-fold CV | accuracy |          0.64         |        0.71       |         no        |       no      |     yes     |
|  gbm3 | 10-fold CV |    ROC   |          0.68         |        0.73       | Two Class Summary |       no      |      no     |
|  svm  |   default  | accuracy |          0.61         |        0.6        |         no        |       no      |      no     |
|  svm1 | 10-fold CV |    ROC   |          0.68         |        0.64       | Two Class Summary |  scale+center |      no     |
|  svm2 | 10-fold CV |    ROC   |          0.68         |        0.65       | Two Class Summary |  scale+center |     yes     |
|  svm3 | 10-fold CV |    ROC   |          0.69         |        0.69       | Two Class Summary |  scale+center |     yes     |
|  rda  | 10-fold CV |    ROC   |          0.67         |        0.74       | Two Class Summary |       no      |      no     |
|  rda1 | 10-fold CV |    ROC   |          0.69         |        0.68       | Two Class Summary |       no      |     yes     |
|  knn  | 10-fold CV | accuracy |          0.63         |        0.66       |         no        |     range     |      no     |
|  knn1 | 10-fold CV | accuracy |          0.64         |        0.69       |         no        |     range     |     yes     |
| ctree | 10-fold CV |    ROC   |          0.59         |        0.65       | Two Class Summary |       no      |     yes     |
|  c45  | 10-fold CV |    ROC   |          0.59         |        0.65       | Two Class Summary |       no      |     yes     |
|   nn  | 10-fold CV | accuracy |          0.64         |        0.67       |         no        |       no      |      no     |
|  nn1  | 10-fold CV |    ROC   |          0.68         |        0.69       | Two Class Summary |       no      |      no     |
|  nn2  | 10-fold CV |    ROC   |          0.63         |        0.69       | Two Class Summary |       no      |     yes     |
|   nb  | 10-fold CV |    ROC   |          0.68         |        0.54       | Two Class Summary |       no      |      no     |



Lets look into it in a bit more detail. Lets get some graphical representation of the same data. 


```
## 'data.frame':	26 obs. of  8 variables:
##  $ Model                : Factor w/ 26 levels "   cf  ","   nb  ",..: 23 25 26 4 17 18 1 6 7 8 ...
##  $ Resampling           : Factor w/ 2 levels "   default  ",..: 1 1 2 1 2 2 1 2 1 2 ...
##  $ Metric               : Factor w/ 2 levels "    ROC   "," accuracy ": 2 2 2 2 2 2 2 1 2 2 ...
##  $ Training.Set.Accuracy: num  0.62 0.64 0.64 0.62 0.65 0.62 0.62 0.67 0.61 0.64 ...
##  $ Test.Set.Accuracy    : num  0.74 0.74 0.74 0.66 0.68 0.69 0.73 0.73 0.69 0.73 ...
##  $ Summary.Function     : Factor w/ 2 levels "         no        ",..: 1 1 1 1 1 1 1 2 1 1 ...
##  $ PreProcessing        : Factor w/ 3 levels "       no      ",..: 1 1 1 1 1 3 1 1 1 1 ...
##  $ Manual.Tune          : Factor w/ 2 levels "      no     ",..: 1 1 1 1 1 1 1 1 1 1 ...
```

Lets plot the training set accuracies first.

![plot of chunk trg_plot](figure/trg_plot-1.png) 

Lets now plot the test set accuracies.

![plot of chunk test_plot](figure/test_plot-1.png) 

Lets now look at models which gave better test set accuracy than the training set

![plot of chunk ratio](figure/ratio-1.png) 


This gives an idea of which models do well and which ones dont do as well. The easy way out is to take the model(s) that performs well or is expected to perform well based on prior experience with the model.

#### Final Models

Lets consider the following models for our final model selection:
- tree
- cf
- gbm1
- svm3
- nn2
- knn1

Lets see how each of these selected models do in the overall prediction accuracy, but also the categorical predictions, i.e. how good is each model in predicting cases that are from different circuits, different Respondents ect. This is important for this study as we expect cases from all types of respondents, from different places and if we can identify models that work for each sub-category well, we can improve our overall accuracy. This method is used in place of the more structured approach given [here](http://topepo.github.io/caret/training.html)

Lets explore by looking into the predictions of the each model and their breakdown for individual sub-categories.


```
## 'data.frame':	169 obs. of  8 variables:
##  $ Circuit   : Factor w/ 13 levels "10th","11th",..: 4 11 7 8 11 11 9 4 12 11 ...
##  $ Issue     : Factor w/ 11 levels "Attorneys","CivilRights",..: 5 5 9 9 9 7 5 8 9 9 ...
##  $ Petitioner: Factor w/ 12 levels "AMERICAN.INDIAN",..: 2 2 8 8 7 9 9 9 9 9 ...
##  $ Respondent: Factor w/ 12 levels "AMERICAN.INDIAN",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ LowerCourt: Factor w/ 2 levels "conser","liberal": 2 2 2 1 1 2 1 2 2 1 ...
##  $ Unconst   : Factor w/ 2 levels "0","1": 1 1 2 1 1 1 1 2 1 1 ...
##  $ Reverse   : Factor w/ 2 levels "no","yes": 2 2 1 2 1 2 2 1 2 2 ...
##  $ prediction: num  0 0 1 1 0 0 1 1 0 1 ...
```

```
## 'data.frame':	169 obs. of  8 variables:
##  $ Circuit   : Factor w/ 13 levels "10th","11th",..: 4 11 7 8 11 11 9 4 12 11 ...
##  $ Issue     : Factor w/ 11 levels "Attorneys","CivilRights",..: 5 5 9 9 9 7 5 8 9 9 ...
##  $ Petitioner: Factor w/ 12 levels "AMERICAN.INDIAN",..: 2 2 8 8 7 9 9 9 9 9 ...
##  $ Respondent: Factor w/ 12 levels "AMERICAN.INDIAN",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ LowerCourt: Factor w/ 2 levels "conser","liberal": 2 2 2 1 1 2 1 2 2 1 ...
##  $ Unconst   : Factor w/ 2 levels "0","1": 1 1 2 1 1 1 1 2 1 1 ...
##  $ Reverse   : Factor w/ 2 levels "no","yes": 2 2 1 2 1 2 2 1 2 2 ...
##  $ prediction: num  0 1 1 1 0 1 1 1 0 1 ...
```

```
## 'data.frame':	169 obs. of  8 variables:
##  $ Circuit   : Factor w/ 13 levels "10th","11th",..: 4 11 7 8 11 11 9 4 12 11 ...
##  $ Issue     : Factor w/ 11 levels "Attorneys","CivilRights",..: 5 5 9 9 9 7 5 8 9 9 ...
##  $ Petitioner: Factor w/ 12 levels "AMERICAN.INDIAN",..: 2 2 8 8 7 9 9 9 9 9 ...
##  $ Respondent: Factor w/ 12 levels "AMERICAN.INDIAN",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ LowerCourt: Factor w/ 2 levels "conser","liberal": 2 2 2 1 1 2 1 2 2 1 ...
##  $ Unconst   : Factor w/ 2 levels "0","1": 1 1 2 1 1 1 1 2 1 1 ...
##  $ Reverse   : Factor w/ 2 levels "no","yes": 2 2 1 2 1 2 2 1 2 2 ...
##  $ prediction: num  1 1 1 1 0 1 1 1 0 1 ...
```

```
## 'data.frame':	169 obs. of  8 variables:
##  $ Circuit   : Factor w/ 13 levels "10th","11th",..: 4 11 7 8 11 11 9 4 12 11 ...
##  $ Issue     : Factor w/ 11 levels "Attorneys","CivilRights",..: 5 5 9 9 9 7 5 8 9 9 ...
##  $ Petitioner: Factor w/ 12 levels "AMERICAN.INDIAN",..: 2 2 8 8 7 9 9 9 9 9 ...
##  $ Respondent: Factor w/ 12 levels "AMERICAN.INDIAN",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ LowerCourt: Factor w/ 2 levels "conser","liberal": 2 2 2 1 1 2 1 2 2 1 ...
##  $ Unconst   : Factor w/ 2 levels "0","1": 1 1 2 1 1 1 1 2 1 1 ...
##  $ Reverse   : Factor w/ 2 levels "no","yes": 2 2 1 2 1 2 2 1 2 2 ...
##  $ prediction: num  1 1 1 1 0 0 1 1 0 1 ...
```

```
## 'data.frame':	169 obs. of  8 variables:
##  $ Circuit   : Factor w/ 13 levels "10th","11th",..: 4 11 7 8 11 11 9 4 12 11 ...
##  $ Issue     : Factor w/ 11 levels "Attorneys","CivilRights",..: 5 5 9 9 9 7 5 8 9 9 ...
##  $ Petitioner: Factor w/ 12 levels "AMERICAN.INDIAN",..: 2 2 8 8 7 9 9 9 9 9 ...
##  $ Respondent: Factor w/ 12 levels "AMERICAN.INDIAN",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ LowerCourt: Factor w/ 2 levels "conser","liberal": 2 2 2 1 1 2 1 2 2 1 ...
##  $ Unconst   : Factor w/ 2 levels "0","1": 1 1 2 1 1 1 1 2 1 1 ...
##  $ Reverse   : Factor w/ 2 levels "no","yes": 2 2 1 2 1 2 2 1 2 2 ...
##  $ prediction: num  1 1 1 1 0 0 1 1 0 1 ...
```

```
## 'data.frame':	169 obs. of  8 variables:
##  $ Circuit   : Factor w/ 13 levels "10th","11th",..: 4 11 7 8 11 11 9 4 12 11 ...
##  $ Issue     : Factor w/ 11 levels "Attorneys","CivilRights",..: 5 5 9 9 9 7 5 8 9 9 ...
##  $ Petitioner: Factor w/ 12 levels "AMERICAN.INDIAN",..: 2 2 8 8 7 9 9 9 9 9 ...
##  $ Respondent: Factor w/ 12 levels "AMERICAN.INDIAN",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ LowerCourt: Factor w/ 2 levels "conser","liberal": 2 2 2 1 1 2 1 2 2 1 ...
##  $ Unconst   : Factor w/ 2 levels "0","1": 1 1 2 1 1 1 1 2 1 1 ...
##  $ Reverse   : Factor w/ 2 levels "no","yes": 2 2 1 2 1 2 2 1 2 2 ...
##  $ prediction: num  1 1 1 1 0 1 1 0 1 1 ...
```


Lets tabulate and visualize the categorical results. It has to be kept in mind that the following recommendations are valid for the respective Circuits, Issues, etc. IN ISOLATION. No interaction effect is considered. Lets begin by looking at the Circuits.

##### Circuits

| Circuit | 1st   | 2nd   | 3rd   | 4th   | 5th   | 6th   | 7th   | 8th   | 9th   | 10th  | 11th  | DC    | FED   | Overall Accuracy |
|---------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|------------------|
| Tree    | 0.571 | 0.556 | 0.714 | 1.000 | 0.769 | 0.786 | 0.556 | 0.813 | 0.676 | 0.833 | 0.818 | 0.714 | 0.571 | 0.740            |
| CF      | 0.571 | 0.556 | 0.714 | 1.000 | 0.769 | 0.786 | 0.556 | 0.813 | 0.618 | 0.833 | 0.818 | 0.714 | 0.571 | 0.728            |
| GBM1    | 0.571 | 0.667 | 0.714 | 1.000 | 0.769 | 0.786 | 0.556 | 0.813 | 0.647 | 0.750 | 0.818 | 0.571 | 0.571 | 0.728            |
| SVM3    | 0.429 | 0.778 | 0.643 | 0.938 | 0.769 | 0.714 | 0.444 | 0.688 | 0.618 | 0.750 | 0.818 | 0.571 | 0.571 | 0.686            |
| NN2     | 0.571 | 0.667 | 0.643 | 1.000 | 0.692 | 0.643 | 0.556 | 0.813 | 0.588 | 0.750 | 0.818 | 0.429 | 0.571 | 0.686            |
| KNN1    | 0.429 | 0.556 | 0.429 | 1.000 | 0.769 | 0.786 | 0.556 | 0.813 | 0.588 | 0.583 | 0.818 | 0.857 | 0.571 | 0.686            |

![plot of chunk circuits](figure/circuits-1.png) 


The plot shows that most of the selected models give accuracies in similar order of magnitude. Noteable deviations from this trend are observed in Circuits 3, 10 and DC. When decisions from these circuits are important, its recomended to avoid using the models giving the low accuracies for the respective circuits.

##### Issues:

| Issue | Attorneys | CivilRights | CriminalProcedure | DueProcess | EconomicActivity | FederalismAndInterstateRelations | FederalTaxation | FirstAmendment | JudicialPower | Privacy | Unions | Overall Accuracy |
|-------|-----------|-------------|-------------------|------------|------------------|----------------------------------|-----------------|----------------|---------------|---------|--------|------------------|
| Tree  | NA        | 0.778       | 0.750             | 0.909      | 0.643            | 0.750                            | 0.667           | 0.900          | 0.727         | 0.333   | 0.750  | 0.740            |
| CF    | NA        | 0.778       | 0.682             | 0.909      | 0.607            | 0.750                            | 0.833           | 0.900          | 0.727         | 0.667   | 0.750  | 0.728            |
| GBM1  | NA        | 0.778       | 0.727             | 0.909      | 0.679            | 0.750                            | 0.833           | 0.700          | 0.667         | 0.667   | 0.750  | 0.728            |
| SVM3  | NA        | 0.778       | 0.659             | 1.000      | 0.607            | 0.625                            | 0.333           | 0.800          | 0.697         | 0.667   | 0.625  | 0.686            |
| NN2   | NA        | 0.778       | 0.659             | 0.909      | 0.643            | 0.750                            | 0.333           | 0.600          | 0.697         | 0.667   | 0.750  | 0.686            |
| KNN1  | NA        | 0.667       | 0.659             | 0.818      | 0.607            | 1.000                            | 0.833           | 0.500          | 0.667         | 1.000   | 0.625  | 0.686            |


![plot of chunk issue](figure/issue-1.png) 

A similar plot for Issues reveal a similar trend, with a clear difference in the the Attourneys section. There are no values to plot. Meaning that there are no Attourney Issues in the Test set. In the other Issues, we see that DueProcess, FederalismAndInterstateRelations, FederalTaxation, FirstAmendment and Privacy show varying levsls of accuracy differences. Depending on which Issue is important, we choose a suitable model to give best prediction accuracy.

 
##### Respondent

| Respondent | AMERICAN.INDIAN | BUSINESS | CITY  | CRIMINAL.DEFENDENT | EMPLOYEE | EMPLOYER | GOVERNMENT.OFFICIAL | INJURED.PERSON | OTHER | POLITICIAN | STATE | US    | Overall Accuracy |
|------------|-----------------|----------|-------|--------------------|----------|----------|---------------------|----------------|-------|------------|-------|-------|------------------|
| Tree       | 0.333           | 0.619    | 1.000 | 0.813              | 0.833    | 1.000    | 0.750               | 1.000          | 0.650 | 1.000      | 0.947 | 0.684 | 0.740            |
| CF         | 0.333           | 0.714    | 1.000 | 0.813              | 0.667    | 1.000    | 0.625               | 1.000          | 0.633 | 1.000      | 0.947 | 0.632 | 0.728            |
| GBM1       | 0.333           | 0.762    | 1.000 | 0.813              | 0.667    | 1.000    | 0.750               | 1.000          | 0.600 | 0.800      | 0.947 | 0.684 | 0.728            |
| SVM3       | 0.333           | 0.667    | 0.500 | 0.875              | 0.500    | 0.714    | 0.750               | 0.333          | 0.617 | 1.000      | 0.947 | 0.579 | 0.686            |
| NN2        | 0.667           | 0.714    | 1.000 | 0.688              | 0.667    | 0.857    | 0.500               | 1.000          | 0.600 | 0.800      | 0.947 | 0.579 | 0.686            |
| KNN1       | 0.667           | 0.667    | 1.000 | 0.813              | 0.667    | 0.857    | 0.750               | 1.000          | 0.533 | 0.800      | 0.895 | 0.632 | 0.686            |


![plot of chunk respo](figure/respo-1.png) 

A glance at the plot reveal that if the respondent is Americal.Indian, KNN1 and NN2 gives best predictions. If the Respondent is City, all models except SVM3 gives similarly good predictions. Predictions for Injured.Person is consistently good, barring the SVM3 model. We have seen that SVM3 model has featured in 2 of the low prediction cases in Respondents. So SVM3 model can be removed as a choice of predictor if Respondent is an important requirement.

##### Petitioner

| Petitioner | AMERICAN.INDIAN | BUSINESS | CITY  | CRIMINAL.DEFENDENT | EMPLOYEE | EMPLOYER | GOVERNMENT.OFFICIAL | INJURED.PERSON | OTHER | POLITICIAN | STATE | US    | Overall Accuracy |
|------------|-----------------|----------|-------|--------------------|----------|----------|---------------------|----------------|-------|------------|-------|-------|------------------|
| Tree       | 1.000           | 0.650    | 0.500 | 0.793              | 0.875    | 0.667    | 0.615               | 1.000          | 0.724 | 1.000      | 0.750 | 0.733 | 0.740            |
| CF         | 1.000           | 0.550    | 0.500 | 0.793              | 0.875    | 0.667    | 0.462               | 1.000          | 0.741 | 1.000      | 0.750 | 0.800 | 0.728            |
| GBM1       | 1.000           | 0.650    | 0.500 | 0.793              | 0.875    | 0.667    | 0.538               | 1.000          | 0.690 | 1.000      | 0.750 | 0.800 | 0.728            |
| SVM3       | 1.000           | 0.600    | 0.500 | 0.724              | 0.750    | 0.667    | 0.462               | 0.667          | 0.707 | 1.000      | 0.750 | 0.667 | 0.686            |
| NN2        | 1.000           | 0.650    | 0.500 | 0.724              | 0.750    | 0.667    | 0.385               | 1.000          | 0.707 | 1.000      | 0.750 | 0.600 | 0.686            |
| KNN1       | 1.000           | 0.550    | 0.000 | 0.793              | 0.750    | 0.667    | 0.385               | 1.000          | 0.690 | 0.800      | 0.750 | 0.733 | 0.686            |


![plot of chunk petit](figure/petit-1.png) 

Its immediately obvious that all the models have low prediction accuracy if the Petitioner is Business, Government.Official and City. KNN1 performs especially bad for City. So KNN1 can be eliminated as a helpful model to predict cases for which the Petitioner is important. On similar grounds, SVM3 and NN2 can be eliminated too.

##### Lower COurt Direction

| Lower Court Direction | conser | liberal | Overall Accuracy |
|-----------------------|--------|---------|------------------|
| Tree                  | 0.773  | 0.704   | 0.740            |
| CF                    | 0.773  | 0.679   | 0.728            |
| GBM1                  | 0.750  | 0.704   | 0.728            |
| SVM3                  | 0.682  | 0.691   | 0.686            |
| NN2                   | 0.705  | 0.667   | 0.686            |
| KNN1                  | 0.761  | 0.593   | 0.686            |

Since this has only 2 sub-categories, it can be easily inferred from without a visual representation of it.

Tree model performs best overall, followed by CF and GBM1.


##### Unconstititional

| Unconst | no / 0 | yes / 1 | Overall Accuracy |
|---------|--------|---------|------------------|
| Tree    | 0.693  | 0.881   | 0.740            |
| CF      | 0.693  | 0.833   | 0.728            |
| GBM1    | 0.701  | 0.810   | 0.728            |
| SVM3    | 0.677  | 0.714   | 0.686            |
| NN2     | 0.654  | 0.786   | 0.686            |
| KNN1    | 0.646  | 0.786   | 0.686            |

Similarly, this table can be interpreted too.

#### Final Model

Overall, the CART model performs the best, both in terms of overall predictions and sub-categorical prediction. This is very closely followed by Random Forest and Boosting models. Usually, one would RF and GBM models to be more accurate that the CART model.

The KNN, SVM and NNET models give good accuracy too, but for some sub-categories, it gives lower accuracy that the Tree based models.


### Conclusion

This study was intended to statistically assess and predict the US Supreme Court decision making. We found that the models predict cose to 75% of the cases correctly.

The statistical models deems that the factors and only these factors are necessary to understand and model Supreme Court decision making. Although the model incorporated a large number of past results in its analysis, it took no account of the explanations the Court itself gave for those decisions. Nor did it take into account specific precedent or relevant statutory or constitutional text. The model is essentially nonlegal in that the factors used to predict decisions-the circuit of origin, the type of the petitioner and respondent, and so forth-are indifferent to law.

In this context, the motivation for this study was to determine whether there the observable "legal" factors are important and can be used for the prediction of outcomes. Our results show that it is possible to predict outcomes of cases if the "legal" factors are coded and available.

In the manner suggested above, we think that the results of this comparative study provide interesting additive insights into the manner in
which one might conceptualize the Supreme Court's decisionmaking. 

This concludes the 3 part series on the prediction of US Supreme Court decisions.
