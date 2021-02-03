library(caret)
library(dplyr)
library(DMwR)
library(rpart) 
library(rpart.plot) 
library(kableExtra)
##### Read data #### 
imputed= read.csv("data/Imputed_data.csv",header=TRUE,stringsAsFactors = TRUE)

summary(imputed)

imputed = imputed[,-1]
imputed$y<-as.factor(imputed$y)

#Sampling the data into train and test
set.seed(110)
samp <- createDataPartition(imputed$y, p=.80, list=FALSE)
train_dt = imputed[samp, ] 
test_dt = imputed[-samp, ]

#SMOTE
set.seed(110)
train_sm_dt <- SMOTE(y~.,data = train_dt)

################  Grid Search ########################

grids <- expand.grid(cp=seq(from=0,to=.25,by=.01))

ctrl_grid <- trainControl(method="repeatedcv",
                          number = 10,
                          repeats = 5,
                          search="grid")

set.seed(101)

DT_grid <- train(form=y ~ ., 
                 data = train_sm_dt, 
                 method = "rpart",
                 trControl = ctrl_grid, 
                 tuneGrid=grids)

DT_grid
plot(DT_grid)
varImp(DT_grid)
confusionMatrix(DT_grid, positive="1")


################## Random Search ######################

ctrl_random <- trainControl(method="repeatedcv",
                            number = 10,
                            repeats = 5,
                            search ="random")

set.seed(101)

DT_random <- train(form = y ~ ., 
                   data = train_sm_dt, 
                   method = "rpart",
                   trControl = ctrl_random, 
                   tuneLength=10)

DT_random
plot(DT_random)
dt_var<-varImp(DT_random)
kable(head(dt_var), digits = 2, format = "html",caption = "Variable Importance Decision Tree Model", row.names = TRUE) %>%
  kable_styling(bootstrap_options = c("striped", "hover"),
                full_width = T,
                font_size = 12,
                position = "left")

confusionMatrix(DT_random, positive="1")

############  1. based on averaged confusion matrix across our resampled cross-validation models,
############  both the model's performance is almost similar    ############

inpreds_grid <- predict(object=DT_grid, newdata=train_sm_dt)
confusionMatrix(data=inpreds_grid, reference=train_sm_dt$y, positive="1")

inpreds_random <- predict(object=DT_random, newdata=train_sm_dt)
confusionMatrix(data=inpreds_random, reference=train_sm_dt$y, positive="1")


###### 2. based on the confusion matrix of predictions for train data, DT_grid's performance is better ########
######### So we select Grid search as our final model #############

train_perf <- confusionMatrix(data=inpreds_grid, 
                              reference=train_sm_dt$y, 
                              mode="prec_recall", positive="1")

############ Predict for test dataset based on Grid search model ####################

outpreds_grid <- predict(object=DT_grid, newdata=test_dt)
confusionMatrix(data=outpreds_grid, reference=test_dt$y, positive="1")

test_perf <- confusionMatrix(data=outpreds_grid, 
                             reference=test_dt$y,
                             mode="prec_recall", positive="1")

############### Compare Performance ########################


dt1<-cbind(train=train_perf$overall, test=test_perf$overall)
dt2<-cbind(train=train_perf$byClass, test=test_perf$byClass)

kable(head(dt1), digits = 2, format = "html",caption = "Performace of Decision Tree Model", row.names = TRUE) %>%
  kable_styling(bootstrap_options = c("striped", "hover"),
                full_width = T,
                font_size = 12,
                position = "left")

kable(head(dt2), digits = 2, format = "html",caption = "Performace of Decision Tree Model (2)", row.names = TRUE) %>%
  kable_styling(bootstrap_options = c("striped", "hover"),
                full_width = T,
                font_size = 12,
                position = "left")


############### Plot Model ########################

#grid search gave the best model cp=0

tree = rpart(y~.,data=test_dt,cp=0)
rpart.plot(tree, clip.right.labs = FALSE, branch = .3, under = TRUE)