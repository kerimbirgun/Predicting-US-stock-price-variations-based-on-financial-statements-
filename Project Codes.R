###################################################################################################################
######
######    
######    PROJECT - FINANCIAL INDICATORS US STOCKS


rm(list = ls())
RNGversion(vstr = '3.6.2')

#Packages:
install.packages("readr")
install.packages("dplyr")
install.packages("tidyr")
install.packages("ggplot2")
install.packages("stringr")
install.packages("gridExtra")
install.packages("caret") # Feature selection and preprocessing
install.packages("glmnet") # Feature selection - Shrinkage:Lasso
install.packages("randomForest") #RandomForest modeling
install.packages("rpart")
install.packages("ranger") # RandomForest modeling
install.packages("gbm")
install.packages("xgboost")
install.packages("naniar")
install.packages("missRanger")
install.packages("bestNormalize")
install.packages("e1071")
install.packages("psych")
install.packages("factoextra")
install.packages("FactoMineR")
install.packages("rattle")
install.packages("olsrr")
#Libraries:
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(caret)
library(glmnet) #Feature selection - Shrinkage:Lasso
library(gridExtra)
library(randomForest)
library(rpart)
library(ranger)
library(gbm)
library(xgboost)
library(naniar) # Check missing values
library(missRanger) #NA imputation
library(bestNormalize) # transform skewed features with BoxCox Transformation
library(e1071)
library(psych) # Bartlett's Test of sphericity and KMO_MAS
library(factoextra) #visualize PCA results
library(FactoMineR) #PCA function
library(rattle) # for binning
library(olsrr) # visualizing linear model's residual plot
#----------------------

#Useful plotting functions

#Density plot
PlotDen <- function(data_input, i){
  data <- data.frame(x=data_input[[i]])
  p <- ggplot(data,aes(x=x))+geom_density()+xlab(paste0((colnames(data_input)[i]),'\n',
                                                        'Skewness: ', round(skewness(data_input[[i]], na.rm = T))))+
    theme_light()
  return(p)
}

#Call plot functions
doPlots <- function(data_input, fun, ii, ncol=2){
  pp <- list()
  for(i in ii){
    p <- fun(data_input=data_input, i=i)
    pp <- c(pp, list(p))
  }
  do.call('grid.arrange', c(pp,ncol=ncol))
}

#------------------------------------------------------------------------

#1. Data
df_path <- file.path("2018_Financial_Data.csv")
df_fin <- read.csv(df_path, header = T)

colnames(df_fin)[224] <- 'PRICE_VAR_12M'
glimpse(df_fin)

#------------------------------------------------------------------------

#2. Explore Data

#2.1 Number of companies - stocks:
nrow(df_fin)

#2.2 Explore the response variable: PRICE_VAR_12M
summary(df_fin$Class)

df_fin %>%
  ggplot(aes(x=Class))+
  geom_density(color = 'steelblue', size=1.2)+
  theme_light()

df_fin %>%
  ggplot(aes(x=PRICE_VAR_12M))+
  geom_density(color='steelblue')+
  xlab('YoY Price Variation')+
  theme_light()

qplot(PRICE_VAR_12M, data = df_fin, bins = 50)

#Outliers and missing values in the response variable
summary(df_fin$PRICE_VAR_12M)

df_fin %>%
  ggplot(aes(x = factor(Sector), y = PRICE_VAR_12M))+
  geom_boxplot(outlier.color = 'red')+
  xlab('Sector')+
  ylab('YoY price Var')+
  theme_light()

df_fin %>%
  group_by(Sector) %>%
  summarise(Q1_0=quantile(PRICE_VAR_12M)[1],
            Q2_25=quantile(PRICE_VAR_12M)[2],
            Q3_50=quantile(PRICE_VAR_12M)[3],
            Q4_75=quantile(PRICE_VAR_12M)[4],
            Q5_100=quantile(PRICE_VAR_12M)[5])

paste('NAs in response variable =',(sum(is.na(df_fin$PRICE_VAR_12M)/nrow(df_fin)*100)), collapse = ' ')

#2.2 Number of stocks per sector

df_fin %>%
  ggplot(aes(x = factor(Sector), fill = factor(Sector)))+
  geom_bar(position = 'dodge')+
  xlab('Sector')+
  theme(axis.text.x = element_blank(),
        legend.title = element_blank())
ylab('Number of Stocks')+
  theme_light()


# Distribution of stock class across sectors

stock_sector <- table(df_fin$Sector, df_fin$Class)
stock_sector  

val <- c('#01a6ff', '#002f87')
lab <- c('Not buy-worthy','buy-worthy')

df_fin %>%
  ggplot(aes(x = factor(Sector), fill = factor(Class)))+
  geom_bar(position = 'dodge')+
  xlab('Sector')+
  ylab('Number of Stocks')+
  scale_fill_manual('Stock Class', values = val, labels = lab)+
  theme_light()

#2.3 Predictors - Exploration

# Predictors - classes
Pred_Info <- data.frame(Pred_Name = names(df_fin),
                        Pred_class = sapply(df_fin[1:225], class),
                        Pred_NAs = sapply(df_fin[1:225], function(x){sum(is.na(x))}),
                        Percent_NA = (sapply(df_fin[1:225], function(x){sum(is.na(x))})/nrow(df_fin))*100)

# Predictors with NAs>15% | Drop predictors with NAs higher than cutoff
NA_cutoff <- 15
Pred_NAs <- Pred_Info %>%
  filter(Percent_NA>=NA_cutoff)

df_Ind <- df_fin[,!(names(df_fin) %in% Pred_NAs$Pred_Name)]



#2.4 Rows/Companies - Exploration

# Companies technical information
CompNA_cutoff <- 10
Comp_info <- data.frame(Company = df_Ind$Stock.Name,
                        Company_NAs = rowSums(is.na(df_Ind)),
                        NAs_percent = ((rowSums(is.na(df_Ind)))/ncol(df_Ind))*100)

Comp_NAs <- Comp_info %>%
  filter(NAs_percent>=CompNA_cutoff)

df_Ind <- df_Ind[!(df_Ind$`Stock.Name` %in% Comp_NAs$Company),]
df_Ind <- as.data.frame(df_Ind)

#----------------------------------------------------------------

#3. Data Procesing and Analysis

#3.1 Missing Values - Imputation | Outliers identification

# Missing Values imputation
gg_miss_upset(df_Ind,
              nsets=10,
              nintersects=10)


# Value imputation
df_Indicators <- missRanger::missRanger(df_Ind, formula = .~.,
                                        pmm.k = 0L, maxiter = 10L, seed = 617, verbose = 0,
                                        returnOOB = F, case.weights = NULL, num.trees = 100)
sum(is.na(df_Indicators))

#3.2 Outliers

sum(df_Indicators$PRICE_VAR_12M>200)
outliers <- df_Indicators[which(df_Indicators$PRICE_VAR_12M>200),1]
df_Indicators <- df_Indicators[!(df_Indicators$Stock.Name %in% outliers),]

df_Indicators <- df_Indicators %>%
  dplyr::select(Sector, everything())

#3.3 Feature Selection

Indicator_Info <- data.frame(Pred_Name = names(df_Indicators[3:181]),
                             Pred_zero = (sapply(df_Indicators[3:181],function(x){sum(x==0)})/nrow(df_Indicators))*100,
                             Pred_mean = sapply(df_Indicators[3:181], function(x){mean(x)}),
                             Pred_sd = sapply(df_Indicators[3:181], function(x){sd(x)}),
                             Pred_Skewness = sapply(df_Indicators[3:181], function(x){moments::skewness(x)}))

pred_zero <- Indicator_Info %>%
  dplyr::filter(Pred_zero>10 | Pred_sd==0) %>%
  dplyr::select(Pred_Name)

df_Indicators <- df_Indicators[,!(names(df_Indicators) %in% pred_zero$Pred_Name)]

# Correlation 
# corMatrix <- cor(df_Indicators[,3:107])
# Drop_var <- caret::findCorrelation(corMatrix, cutoff = 0.60, names = T, exact = T)
# df_Indicators <- df_Indicators[,!(names(df_Indicators) %in% Drop_var)]

#Find Variable Importance
df_IndiNum <- subset(df_Indicators, select = -c(Sector,Stock.Name,Class))
set.seed(617)
quick_RF <- randomForest::randomForest(x = df_IndiNum[1:2000,1:104], y = df_IndiNum$PRICE_VAR_12M[1:2000],
                                       ntree = 200, importance = TRUE)

features_RF <- quick_RF$importance
features_RF <- data.frame(Variables = row.names(features_RF), MSE= features_RF[,1])
features_RF %>%
  dplyr::arrange(desc(MSE)) %>%
  dplyr::top_n(20) %>%
  ggplot(aes(x = reorder(Variables,MSE), y = MSE, fill = MSE))+
  geom_bar(stat = 'identity')+
  labs(x = 'Variables', y = '% Increase MSE if variable is randomly permuted')+
  coord_flip()+
  theme(legend.position = 'none')

features_RF <- features_RF %>%
  dplyr::arrange(desc(MSE)) %>%
  dplyr::top_n(20) %>%
  dplyr::select(Variables)

Quick_Ranger <- ranger(PRICE_VAR_12M~., data = df_IndiNum, importance = 'impurity')

features_RG <- data.frame(variables = names(Quick_Ranger$variable.importance), value = unname(Quick_Ranger$variable.importance))
features_RG %>%
  dplyr::arrange(desc(value)) %>%
  dplyr::top_n(25) %>%
  ggplot(aes(x = reorder(variables, value), y = value, fill = value))+
  geom_bar(stat = 'identity')+
  labs(x = 'Variables', y = 'Importance')+
  coord_flip()+
  theme(legend.position = 'none')

features_RG <- features_RG %>%
  dplyr::arrange(desc(value)) %>%
  dplyr::top_n(25) %>%
  dplyr::select(variables)


#Lasso feature selection
# x = model.matrix(PRICE_VAR_12M~., data = df_Import_RF)
# y = df_Import_RF$PRICE_VAR_12M
# 
# LassoModel <- glmnet::glmnet(x, y, alpha = 1)
# plot(LassoModel, xvar = 'dev', label = T)
# 
# set.seed(1031)
# cv.out <- cv.glmnet(x,y, alpha=1)
# bestLam <- cv.out$lambda.min
# LassoModel2 <- glmnet(x, y, alpha = 1, lambda = bestLam)
# Lasso.coef <- predict(LassoModel2, type = 'coefficients', s = bestLam)[1:104,]
# Lasso.coef

#3.4 Fixing skewed Features
features <- df_Indicators[,(names(df_Indicators) %in% features_RG$variables)]
feat_num <- names(features)

doPlots(features,PlotDen,1:10)
doPlots(features,PlotDen,11:20)

skewed_features <- sapply(feat_num, function(x){
  e1071::skewness(features[[x]], na.rm = T)
})

summary(features)

# Transform skewed features
features <- data.frame(sapply(features, function(x){predict(bestNormalize(x))}))
skewness(features$Revenue.Growth)

# Add response variables + sector + stock name
features <- cbind(df_Indicators[,1:2], features, df_Indicators$PRICE_VAR_12M)

#Export data
export.path <- file.path("C:/Users/lecheverri/Documents/Luis M. Echeverri/Luis Miguel/Universities/Columbia/07. APA Frameworks and Methods II/11. Project/CleanNormData.csv")
write.csv(features, export.path, row.names = F)

#--------------------------------------------------------


#########################################################################################
#########################################################################################

#4. Prediction Models
# linear regression w/o Normalization  (RMSE Train:42.2, Test:42.8) ,R2 = 0.11)

data = read.csv("CleanNormData.csv")
glimpse(data)

# changing the name of Y variable
data = data %>% rename(Price.Change = df_Indicators.PRICE_VAR_12M)

# splitting data 
data = data %>% select(-c("Stock.Name"))  
dim(data)

library(caret);library(lattice)
set.seed(150)
split <- createDataPartition(data$Price.Change, 
                             p = 0.7,list = F)
data_train <- data[split,]
data_test  <- data[-split,]

# linear regression   (RMSE Train:42.2, Test:42.8)

model_lm = lm(Price.Change~.,data=data_train)
summary(model_lm) # adj R squared  0.1141 

pred_lm_tr = predict(model_lm)
rmse_lm_tr = sqrt(mean((pred_lm_tr-data_train$Price.Change)^2)); rmse_lm_tr

pred_lm_ts = predict(model_lm,newdata=data_test)
rmse_lm_ts = sqrt(mean((pred_lm_ts-data_test$Price.Change)^2)); rmse_lm_ts

ols_plot_resid_qq(model_lm)
# Normalization Needed



##################################################################
#5. Prediction Model with Normalization (Linear Regression)
# linear regression   (RMSE Train: 42.1 , Test: 42.7 ,R2 = 0.13)



data = read.csv("CleanNormData.csv")
glimpse(data)

# changing the name of Y variable
data = data %>% rename(Price.Change = df_Indicators.PRICE_VAR_12M)

# splitting data 
data = data %>% select(-c("Stock.Name"))  
dim(data)

# Normalization of dependent variable

library(bestNormalize)
bestobject = bestNormalize(data$Price.Change, allow_lambert_s = TRUE)
data$Price.Change = predict(bestobject)

set.seed(150)
split <- createDataPartition(data$Price.Change, 
                             p = 0.7,list = F)
data_train <- data[split,]
data_test  <- data[-split,]

# linear regression   (RMSE Train: 42.1 , Test: 42.7 ,R2 = 0.13)

model_lm = lm(Price.Change~.,data=data_train)
summary(model_lm)
# adj R squared  0.1317
ols_plot_resid_qq(model_lm)


# Train RMSE calculation with reverse normalization
pred_lm_tr = predict(model_lm)
pred_lm_tr = predict(bestobject, newdata = pred_lm_tr, inverse = TRUE)
data_train$Price.Change = predict(bestobject, newdata = data_train$Price.Change, inverse = TRUE)
rmse_lm_tr = sqrt(mean((pred_lm_tr-data_train$Price.Change)^2)); rmse_lm_tr

# Test RMSE calculation with reverse normalization
pred_lm_ts = predict(model_lm,newdata=data_test)
pred_lm_ts = predict(bestobject, newdata = pred_lm_ts, inverse = TRUE)
data_test$Price.Change = predict(bestobject, newdata = data_test$Price.Change, inverse = TRUE)
rmse_lm_ts = sqrt(mean((pred_lm_ts-data_test$Price.Change)^2)); rmse_lm_ts



##################################################################
#6. Prediction Model with Normalization (Random Forest)
# random forest (RMSE Train: 41.9 , Test: 41.5 , R2 0.15)


data = read.csv("CleanNormData.csv")
glimpse(data)

# changing the name of Y variable
data = data %>% rename(Price.Change = df_Indicators.PRICE_VAR_12M)
# splitting data 
data = data %>% select(-c("Stock.Name"))  
dim(data)
# Normalization of dependent variable
library(bestNormalize)
bestobject = bestNormalize(data$Price.Change, allow_lambert_s = TRUE)
data$Price.Change = predict(bestobject)
set.seed(150)
split <- createDataPartition(data$Price.Change, 
                             p = 0.7,list = F)
data_train <- data[split,]
data_test  <- data[-split,]

# random forest (RMSE Train: 41.9 , Test: 41.5 , R2 0.15)
library(randomForest)
set.seed(100)
model_rf = randomForest(Price.Change~.,data=data_train,ntree=1000)
model_rf

# Train RMSE calculation with reverse normalization
pred_rf_tr = predict(model_rf)
pred_rf_tr = predict(bestobject, newdata = pred_rf_tr, inverse = TRUE)
data_train$Price.Change = predict(bestobject, newdata = data_train$Price.Change, inverse = TRUE)
rmse_rf_tr = sqrt(mean((pred_rf_tr-data_train$Price.Change)^2)); rmse_rf_tr

# Test RMSE calculation with reverse normalization
pred_rf_ts = predict(model_rf,newdata=data_test)
pred_rf_ts = predict(bestobject, newdata = pred_rf_ts, inverse = TRUE)
data_test$Price.Change = predict(bestobject, newdata = data_test$Price.Change, inverse = TRUE)
rmse_rf_ts = sqrt(mean((pred_rf_ts-data_test$Price.Change)^2)); rmse_rf_ts



##################################################################
#7. Prediction Model with Normalization (Lasso Regression)
# lasso regression (RMSE Train: 42.3 , Test: 42.2 , R2 0.138)


data = read.csv("CleanNormData.csv")
glimpse(data)

# changing the name of Y variable
data = data %>% rename(Price.Change = df_Indicators.PRICE_VAR_12M)
# splitting data 
data = data %>% select(-c("Stock.Name"))  
dim(data)
# Normalization of dependent variable
library(bestNormalize)
bestobject = bestNormalize(data$Price.Change, allow_lambert_s = TRUE)
data$Price.Change = predict(bestobject)
set.seed(150)
split <- createDataPartition(data$Price.Change, 
                             p = 0.7,list = F)
data_train <- data[split,]
data_test  <- data[-split,]

#regularization
library(caret)
dummies <- dummyVars(Price.Change ~ . , data = data_train)
train_dummies = predict(dummies, newdata = data_train)
pred_dummies = predict(dummies, newdata = data_test)

library(glmnet)
x = as.matrix(train_dummies)
y = as.matrix(pred_dummies)
y_train = data_train$Price.Change

# lasso regression (RMSE Train: 42.3 , Test: 42.2 , R2 0.138)
lambdas <- 10^seq(2, -3, by = -.1)
lasso_reg <- cv.glmnet(x, y_train, alpha = 1, lambda = lambdas,
                       standardize = TRUE, nfolds = 5)
lambda_best <- lasso_reg$lambda.min 
lambda_best
lasso_model <- glmnet(x, y_train, alpha = 1, lambda = lambda_best,
                      standardize = TRUE)
lasso_model
pred_lasso <- predict(lasso_model,
                      s = lambda_best, newx = y)

# Train RMSE calculation with reverse normalization
pred_lm_tr <- predict(lasso_model,s = lambda_best, newx = x) 
pred_lm_tr = predict(bestobject, newdata = pred_lm_tr, inverse = TRUE)
data_train$Price.Change = predict(bestobject, newdata = data_train$Price.Change, inverse = TRUE)
rmse_lm_tr = sqrt(mean((pred_lm_tr-data_train$Price.Change)^2)); rmse_lm_tr

# Test RMSE calculation with reverse normalization
pred_lm_ts <- predict(lasso_model,s = lambda_best, newx = y)
pred_lm_ts = predict(bestobject, newdata = pred_lm_ts, inverse = TRUE)
data_test$Price.Change = predict(bestobject, newdata = data_test$Price.Change, inverse = TRUE)
rmse_lm_ts = sqrt(mean((pred_lm_ts-data_test$Price.Change)^2)); rmse_lm_ts



##################################################################
#8. Prediction Model with Normalization (GBM)
# Gradient Boosting (RMSE Train: 40.1 , Test: 41.5 , R2 (not adjusted) 0.18 ???)


data = read.csv("CleanNormData.csv")
glimpse(data)

# changing the name of Y variable
data = data %>% rename(Price.Change = df_Indicators.PRICE_VAR_12M)
# splitting data 
data = data %>% select(-c("Stock.Name"))  
dim(data)
# Normalization of dependent variable
library(bestNormalize)
bestobject = bestNormalize(data$Price.Change, allow_lambert_s = TRUE)
data$Price.Change = predict(bestobject)
set.seed(150)
split <- createDataPartition(data$Price.Change, 
                             p = 0.7,list = F)
data_train <- data[split,]
data_test  <- data[-split,]

# Gradient Boosting (RMSE Train: 40.1 , Test: 41.5 , R2 (not adjusted) 0.18 ???)
library(gbm)
set.seed(617)
boosted_model = gbm(Price.Change~.,data=data_train,verbose = TRUE,shrinkage = 0.01,  
                    interaction.depth = 3, n.trees = 10000, cv.folds = 5)
summary(boosted_model)
# Train RMSE calculation with reverse normalization
pred_tr_gbm = predict(boosted_model)
pred_tr_gbm = predict(bestobject, newdata = pred_tr_gbm, inverse = TRUE)
data_train$Price.Change = predict(bestobject, newdata = data_train$Price.Change, 
                                  inverse = TRUE)
rmse_tr_gbm = sqrt(mean((pred_tr_gbm-data_train$Price.Change)^2)); rmse_tr_gbm

# Test RMSE calculation with reverse normalization
pred_ts_gbm = predict(boosted_model,newdata=data_test)
pred_ts_gbm = predict(bestobject, newdata = pred_ts_gbm, inverse = TRUE)
data_test$Price.Change = predict(bestobject, newdata = data_test$Price.Change, inverse = TRUE)
rmse_ts_gbm = sqrt(mean((pred_ts_gbm-data_test$Price.Change)^2)); rmse_ts_gbm

#R2 Calculation 
residuals = data_train$Price.Change - pred_tr_gbm 
rss =  sum(residuals^2)
y_train_mean = mean(data_train$Price.Change)
tss =  sum((data_train$Price.Change - y_train_mean)^2)
rsq_train  =  1 - (rss/tss)
rsq_train

residuals = data_test$Price.Change - pred_ts_gbm 
rss =  sum(residuals^2)
y_test_mean = mean(data_test$Price.Change)
tss =  sum((data_test$Price.Change - y_test_mean)^2)
rsq_test  =  1 - (rss/tss)

R2  = (nrow(data_train)*rsq_train + nrow(data_test)*rsq_test) / (nrow(data_train) + nrow(data_test))
R2




#### DEEP LEARNING TRIAL FOR REGRESSION ####
###############################################

data = read.csv("CleanNormData.csv")
data = data %>% rename(Price.Change = df_Indicators.PRICE_VAR_12M)
data = data %>% select(-c("Stock.Name"))  


set.seed(1031)
split = sample(x = c('train','validation','test'),size = nrow(data),replace = T,prob = c(0.5,0.25,0.25))
train = data[split=='train',]
validation = data[split=='validation',]
test = data[split=='test',]

dim(test);dim(train);dim(validation)

library(h2o)
train_h2o = as.h2o(train)
valid_h2o = as.h2o(validation)
test_h2o = as.h2o(test)


#  train rmse 42.1  test rmse 41.1 (slightly changes everytime you run)

model_nn = h2o.deeplearning(y = "Price.Change",training_frame = train_h2o,
                          hidden = c(3,10,3,10,3),seed=1031)

h2o.r2(model_nn)

pred_nn_tr = predict(model_nn,newdata=train_h2o)
rmse_nn_tr = sqrt(mean((pred_nn_tr-train_h2o$Price.Change)^2)); rmse_nn_tr

pred_nn_ts = predict(model_nn,newdata=test_h2o)
rmse_nn_ts = sqrt(mean((pred_nn_ts-test_h2o$Price.Change)^2)); rmse_nn_ts


# EXTRA
####################################################
# RF + Lasso + GBM  (RMSE Train: 41.2 , Test: 41.5)

pred_combined_tr = (pred_rf_tr*0.50) + (pred_tr_gbm*0.3) + (pred_lm_tr*0.2)
rmse_tr_comb = sqrt(mean((pred_combined_tr-data_train$Price.Change)^2)); rmse_tr_comb

pred_combined_ts = (pred_rf_ts*0.50) + (pred_ts_gbm*0.3) + (pred_lm_ts*0.2)
rmse_ts_comb = sqrt(mean((pred_combined_ts-data_test$Price.Change)^2)); rmse_ts_comb

































