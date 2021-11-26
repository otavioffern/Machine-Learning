##APS##
#install.packages('Rcpp')
#install.packages('caret')
library(Rcpp)
library(pROC)
library(ISLR)
library(skimr)
library(tree)
library(randomForest)
library(fastAdaboost)
library(MASS)
library(gbm)
library(caret)

setwd("C:/Users/ferna/Desktop/Modelagem Preditiva/APS")

## Classificação ##

# Início---------------------------------------------------------------------
#O objetivo é prever a variável Exited, que determina se o
#cliente cancelará o serviço (churned) ou não.
churn <- read.csv("churn.csv")
#fazendo de Yes/No para 1/0
require(dplyr)
churn <- churn %>%
  mutate(Exited = factor(ifelse(Exited == "No",0,1)))

set.seed(1234)

idx <- sample(1:nrow(churn), size = round(0.5*nrow(churn)), replace = FALSE)

training <- churn[idx, ]
test <- churn[-idx, ]

#RL-------------------------------------------------------------------
#REG LOGÍSTICA

model_RegLog <- glm(Exited ~., data = training, family = binomial)

summary(model_RegLog)

prob <- predict(model_RegLog, newdata = test, type = "response")

y_hat <- ifelse(prob > 0.5, 1, 0)

table(Predicted = y_hat, Observed = test$Exited)

#CART---------------------------------------------------------------------
#Árvore de Classificação

ctree <- tree(Exited ~ ., data = training)

y_hat_tree <- predict(ctree, newdata = test, type = "class")			 


table(Predicted = y_hat_tree, Observed = test$Exited)

pr_tree <- predict(ctree, newdata = test, type = "vector")[, 2]
#RF---------------------------------------------------------------------
#Random Forest
rf <- randomForest(Exited ~ .,data = training)

y_hat_rf <- predict(rf, newdata = test)

table(Predicted = y_hat_rf, Observed = test$Exited)

pr_rf <- predict(rf, newdata = test, type = "prob")[, 2]
#B---------------------------------------------------------------------
#Boosting
boost <- adaboost(Exited ~ ., data = training, nIter = 50)

pred_boost <- predict(boost, newdata = test)

y_hat_boost <- pred_boost$class

table(Predicted = y_hat_boost, Observed = test$Exited)

pr_boost <- pred_boost$prob[, 2]
#ROC---------------------------------------------------------------------
#Curva ROC

roc_rlog <- roc(test$Exited, prob)
roc_tree <- roc(test$Exited, pr_tree)
roc_rf <- roc(test$Exited, pr_rf)
roc_boost <- roc(test$Exited, pr_boost)

#fazendo a apresentacao do gráfico como só um de novo
par(mfrow = c(1,1))

plot(roc_rlog, col = "blue", grid = TRUE,
     xlab = "FPR (1- Specificity)",
     ylab = "TPR (Sensitivity)",
     main = "ROC", legacy.axes = TRUE, asp = FALSE, las = 1)
plot(roc_rf, col = "red", add = TRUE)
plot(roc_boost, col = "black", add = TRUE)
plot(roc_tree, col = "green", add = TRUE)
legend("bottomright",
       legend = c("Árvore de Classificação", "Regressão Logística", "Random Forest", "Boosting"),
       col = c("green", "blue", "red", "black"), lwd = 3, cex = 0.8)

# AUC

AUC <- data.frame(Modelo = c("Regressão Logística", "Árvores de Classificação"
                                ,"Random Forest","Boosting"),
                  AUC = c(auc(roc_rlog),
                          auc(roc_tree),
                          auc(roc_rf),
                          auc(roc_boost)))
AUC

#ACURÁCIA

rl_acur <- mean(y_hat == test$Exited)
tree_acur <- mean(y_hat_tree == test$Exited)
rf_acur <- mean(y_hat_rf == test$Exited)
boost_acur <- mean(y_hat_boost == test$Exited)

acuracia <- data.frame(Modelo = c("Regressão Logística", "Árvores de Classificação",
                       "Random Forest","Boosting"), Acurácia = c(rl_acur, tree_acur, rf_acur, boost_acur))
acuracia


## REGRESSÃO ##



# Início used cars----------

#O objetivo é prever a variável price
used <- read.csv("used_cars.csv")

price = used$price
used = model.matrix(price ~ ., data = used)
used = data.frame(used)
used$price = price
str(used)

set.seed(1234)

idx_used <- sample(1:nrow(used), size = round(0.7 * nrow(used)), replace = FALSE)

training_used <- used[idx_used, ]
test_used <- used[-idx_used, ]

# Linear regression---------------

ols <- lm(price ~ ., data = training_used)

y_hat_ols <- predict(ols, newdata = test_used)

(RMSE_ols <- sqrt(mean((y_hat_ols - test_used$price)^2)))

# Single tree----------------

regtree <- tree(price ~ ., data = training_used)

summary(regtree)

plot(regtree, type = "uniform")
text(regtree, cex = 0.75)

y_hat_regtree <- predict(regtree, newdata = test_used)

(RMSE_regtree <- sqrt(mean((y_hat_regtree - test_used$price)^2)))

# Random Forest----------

rf_used <- randomForest(price ~ ., data = training_used)

y_hat_rf_used <- predict(rf_used, newdata = test_used)
(RMSE_rf_used <- sqrt(mean((y_hat_rf_used - test_used$price)^2)))

# Boosting------------
training_used$isOneOwner = as.numeric(training_used$isOneOwner)
test_used$isOneOwner = as.numeric(test_used$isOneOwner)

boost_used <- gbm(price ~ ., data = training_used,
             distribution = "gaussian",
             n.trees = 1000,
             interaction.depth = 5, # maximum depth of each tree
             shrinkage = 0.002)     # default lambda

y_hat_boost_used <- predict(boost_used, newdata = test_used, n.trees = 5000)

(RMSE_boost_used <- sqrt(mean((y_hat_boost_used - test_used$price)^2)))

#RMSE-----------------------
RMSE <- data.frame(Modelo = c("OLS", "Árvore de Regressão"
                             ,"Random Forest","Boosting"),
                  RMSE = c(RMSE_ols,
                          RMSE_regtree,
                          RMSE_rf_used,
                          RMSE_boost_used))
RMSE
