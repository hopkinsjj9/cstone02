## https://github.com/hopkinsjj9/cstone02

## Setup

library(mlbench)
library(caret)
library(randomForest)
library(knitr)
library(dplyr)
library(tidyr)
library(missCompare)
library(mice)
library(ROSE)
library(xgboost)
library(Matrix)
library(ggplot2)
library(gridExtra)
library(scales)
library(e1071) 
library(MLmetrics)

df.orig <- read.csv("../data/Gonzalez-Suarez_et_al_Dataset.csv", header=T, stringsAsFactors=F)

## rearrange / reclassify char columns
df.orig <- df.orig %>%
  select(everything()) %>%
  filter(Year_introduction >= 1800)

df.orig <- df.orig %>% select(-Establishment_success,Establishment_success)

df.orig$Year_introduction <-  as.character(df.orig$Year_introduction)

for( col in names(df.orig[, sapply(df.orig, class) == 'character'])) {
  colnbr <- which(colnames(df.orig) == col )
  df.orig[,colnbr] <- (factor(df.orig[,colnbr] ))
  print(paste( col))
  df.orig <- df.orig %>% select(-col,col)
}

df.orig.dvar <- which( colnames(df.orig)=="Establishment_success" )

df <- df.orig

remove.named.cols <- function(dt, nme.tok) {
  for (i in sort(1:ncol(dt), decreasing = T)) {
    #print(paste(nme.tok, i, names(df)[i]))
    if (grepl(nme.tok, names(dt)[i])) {
      dt[,i] <- NULL
    }
  }
  return(dt)
}

df  <- remove.named.cols(df, 'CV_')
dvar <- which( colnames(df)=="Establishment_success" )

for (i in (dvar + 1): (length(df))) {
  df[,length(df)] <- NULL
}


post.ol.stats <- function(nme, ol.nbr, ol.prop,  ol.mean, ol.mean.before, ol.mean.after ) {
tmp <- data.frame( Name    = nme,
Count = ol.nbr,
Prop  = ol.prop, 
Mean  = ol.mean,
MeanPrior = ol.mean.before,
MeanAfter = ol.mean.after )
rownames(tmp) <- ''
ol.stats <-rbind(ol.stats, tmp)
return (ol.stats)
}

outlierKD <- function(dt, nme) {
var_name <- dt[[nme]]
na1 <- sum(is.na(var_name))
m1 <- mean(var_name, na.rm = T)
outlier <- boxplot(var_name, plot=FALSE)$out
mo <- mean(outlier)
var_name <- ifelse(var_name %in% outlier, NA, var_name)
na2 <- sum(is.na(var_name))
m2 <- mean(var_name, na.rm = T)
dt[[nme]] <- invisible(var_name)
assign(as.character(as.list(match.call())$dt), dt, envir = .GlobalEnv)
ol.nbr  <- (na2 - na1)
ol.prop <- round((na2 - na1) / sum(!is.na(var_name))*100, 1)
ol.mean <- round(mo, 2)
ol.mean.before <- round(m1, 2)
ol.mean.after <- round(m2, 2)
ol.tmp <- post.ol.stats(nme, ol.nbr , ol.prop,  ol.mean, ol.mean.before, ol.mean.after ) 
return(ol.tmp)
}
ol.stats <- data.frame()

for (i in 1:26) {
ol.stats <- outlierKD(df, names(df)[i]) 
}

kable(ol.stats)

df.clean <- missCompare::clean(df,
var_removal_threshold = 0.5, 
ind_removal_threshold = 0.8,
missingness_coding = -9)

kable(sapply(df.clean, function(x) sum(is.na(x))))

dvar <- which( colnames(df)=="Establishment_success" )

par(mfrow=c(1, 1))

## impute missing values

df.mice <- mice(df.clean, m=5, maxit = 20, method = 'cart', seed = 500, print=F)
df <- mice::complete(df.mice ,1)

# center / scale data
df.pp <- preProcess(df[, -dvar], 
method = c("center", "scale", "YeoJohnson", "nzv"))
df.pp
df.pp <- predict(df.pp, newdata = df[, -dvar])

df.pp$Establishment_success = factor(df$Establishment_success)
df <- df.pp

## check for near zero predictors

nzv <- nearZeroVar(df)
nzv

## Balance Dataset

## data is unbalanced and requires balancing

dvar <- which( colnames(df)=="Establishment_success" )
cbind(freq=table(df[,dvar]), percentage=prop.table(table(df[,dvar]))*100)

set.seed(777)
train.no.cnt <- summary(df$Establishment_success)['0']
train.yes.cnt <- summary(df$Establishment_success)['1']
train.both.cnt <- train.no.cnt + train.yes.cnt

## create training samples to deal with unbalanced class

train.full <- list(
train = df,
over  = ovun.sample(Establishment_success ~ ., data = df, 
method = "over", N = train.yes.cnt * 2)$data,
under = ovun.sample(Establishment_success ~ ., data = df, 
method = "under", N = train.no.cnt * 2)$data,
both  = ovun.sample(Establishment_success ~ ., data = df, p = 0.5,
seed = 777, method = "both", N = train.both.cnt)$data,
rose  = ROSE(Establishment_success ~ ., data = df, N = train.no.cnt * 2, 
seed=111)$data
)

# determine the best balancing method
cm <- data.frame()
for( i in 1:length(train.full)) {
  dt <- train.full[[i]]
  learn_rf <- randomForest(dt$Establishment_success ~ .,
    data=dt, ntree=500,
    proximity=T, importance=T, na.action=na.roughfix)
  pre_rf <- predict(learn_rf, df[,-dvar])
  cfm <- confusionMatrix(pre_rf, df$Establishment_success)
  stats <- data.frame( name = names(train.full)[i],
  Accuracy = cfm$overall['Accuracy'],
  AccuracyNull = cfm$overall['AccuracyNull'],
  Sensitivity = cfm$byClass['Sensitivity'],
  Specificity = cfm$byClass['Specificity'],
  Balanced_Accuracy = cfm$byClass['Balanced Accuracy'],
  stringsAsFactors = FALSE)
  rownames(stats) <- c()
  cm <- rbind(cm, stats)
}

kable(cm)

## over method maximizes sensitivity/specificity and balanced accuracy
df <- train.full[['over']]

cm.stats <- data.frame()

post.cm.stats <- function(meth,cmtx,auc) {
  tmp <- data.frame( Method = meth,
    Accuracy = cmtx$overall[1], 
    Sensitivity = cmtx$byClass[1],
    Specificity = cmtx$byClass[2],
    BalAccuracy = cmtx$byClass[6],
    AUC = auc
  )
  rownames(tmp) <- ''
  cm.stats <-rbind(cm.stats, tmp)
return (cm.stats)
}


## Random Forest

set.seed(222)
ind <- sample(2, nrow(df), replace = T, prob = c(0.6, 0.4))
train <- df[ind==1,]
test <- df[ind==2,]

rf01 <- randomForest(train$Establishment_success ~ ., data=train, 
ntree=500, proximity=T, importance=T, na.action=na.roughfix)
rf01.pred <- predict(rf01, test[,-dvar])
rf01.roc <- roc.curve( test[,dvar], rf01.pred)
rect(0, 1.1, 1, 1.7, xpd=TRUE, col="white", border="white")
title("Random Forest ROC Curve")

print(rf01.roc$auc) 
# Prediction & Confusion Matrix - Test
rf01.cm <-confusionMatrix(rf01.pred, test[, dvar])

cm.stats <- post.cm.stats('randomForest',rf01.cm, rf01.roc$auc)

importance    <- importance(rf01)
varImportance <- data.frame(Variables = row.names(importance), 
Importance = round(importance[ ,'MeanDecreaseGini'],2))

rankImportance <- varImportance %>%
mutate(Rank = paste0('#',dense_rank(desc(Importance))))

ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() 


## GLM

glm.model <- glm(Establishment_success ~ ., data=train, family='binomial')
glm.pred <- predict(glm.model, test, type="response")
glm.tbl <- table(test$Establishment_success, glm.pred > 0.5)
colnames(glm.tbl) <- c('TRUE', 'FALSE')
rownames(glm.tbl) <- c('TRUE', 'FALSE')
glm.cm <- confusionMatrix(glm.tbl)

glm.roc <-roc.curve( test[,dvar], glm.pred)
rect(0, 1.1, 1, 1.7, xpd=TRUE, col="white", border="white")
title("GLM ROC Curve")
print(glm.roc$auc) 

cm.stats <- post.cm.stats('glm', glm.cm, glm.roc$auc)

## XGBoost

train$Establishment_success <- as.numeric(train$Establishment_success)-1
test$Establishment_success <- as.numeric(test$Establishment_success)-1

trainm <- sparse.model.matrix(Establishment_success~., data = train)
train.label <- train[,"Establishment_success"]
train.matrix <- xgb.DMatrix(data = as.matrix(trainm), label = train.label)

testm <- sparse.model.matrix(Establishment_success~., data = test)
test.label <- test[,"Establishment_success"]
test.matrix <- xgb.DMatrix(data = as.matrix(testm), label = test.label)

nc <- length(unique(train.label))
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = nc)

watchlist <- list(train = train.matrix, test = test.matrix)

best.xgboost.model <- function(nbr_rounds, params) {
  best <- xgb.train(params = params,
                    data = train.matrix,
                    nrounds = nbr_rounds,
                    watchlist = watchlist,
                    eta = 0.01,
                    max.depth = 3,
                    gamma = 0,
                    subsample = 1,
                    colsample_bytree = 1,
                    missing = NA,
                    seed = 333,
                    verbose = 0)
  # Training & test error plot
  e <- data.frame(best$evaluation_log)
  plot(e$iter, e$train_mlogloss, col = 'blue')
  lines(e$iter, e$test_mlogloss, col = 'red')
  
  print(e[e$test_mlogloss == min(e$test_mlogloss),])
  return(list(best = best, iter = e[e$test_mlogloss == min(e$test_mlogloss),]))
}

best.iter <- 5000
res <- best.xgboost.model(best.iter, xgb_params)
rect(0, 1.1, 1, 1.7, xpd=TRUE, col="white", border="white")
title("XGBoost First Pass")
if (best.iter != res$iter[[1]]) {
  res <- best.xgboost.model(res$iter[[1]], xgb_params)
}
rect(0, 1.1, 1, 1.7, xpd=TRUE, col="white", border="white")
title("XGBoost Best Model")
best.model <- res$best

imp <- xgb.importance(colnames(train.matrix), model = best.model)
xgb.plot.importance(imp)
rect(0, 1.1, 1, 1.7, xpd=TRUE, col="white", border="white")
title("XGBoost Important Features")

# Prediction & confusion matrix - test data
xg01.pred <- predict(best.model, newdata = test.matrix)
pred <- matrix(xg01.pred, nrow = nc, ncol = length(xg01.pred)/nc) %>%
  t() %>%
  data.frame() %>%
  mutate(label = test.label, max_prob = max.col(., "last")-1)

xg01.roc <-roc.curve( pred$max_prob, pred$label)
rect(0, 1.1, 1, 1.7, xpd=TRUE, col="white", border="white")
title("XGBoost ROC Curve")
print(xg01.roc$auc) 

xgboost.cm <- confusionMatrix(factor(pred$max_prob), factor(pred$label))

cm.stats <- post.cm.stats('xgboost', xgboost.cm, xg01.roc$auc)

## Compare Models
cm.stats

## Identify Important Features

observation_level_variable_importance = function(train_data, live_data, 
                                                 outcome_name, 
                                                 eta = 0.2, 
                                                 max_depth=4, max_rounds=3000,
                                                 number_of_factors=2) {
  set.seed(1234)
  split <- sample(nrow(train_data), floor(0.9 * nrow(train_data)))
  train_data_tmp <- train_data[split,]
  val_data_tmp <- train_data[-split,]
  
  feature_names <- setdiff(names(train_data_tmp), outcome_name)
  dtrain <- xgb.DMatrix(data.matrix(train_data_tmp[,feature_names]), 
                        label=train_data_tmp[,outcome_name], missing=NaN)
  dval <- xgb.DMatrix(data.matrix(val_data_tmp[,feature_names]), 
                      label=val_data_tmp[,outcome_name], missing=NaN)
  watchlist <- list(eval = dval, train = dtrain)
  param <- list(  objective = "binary:logistic",
                  eta = eta,
                  max_depth = max_depth,
                  subsample= 0.9,
                  colsample_bytree= 0.9
  )
  
  xgb_model <- xgb.train ( params = param,
                           data = dtrain,
                           eval_metric = "auc",
                           nrounds = max_rounds,
                           missing=NaN,
                           verbose = 0,
                           print_every_n = 10,
                           early_stop_round = 20,
                           watchlist = watchlist,
                           maximize = TRUE)
  
  original_predictions <- predict(xgb_model, 
                                  data.matrix(live_data[,feature_names]), 
                                  outputmargin=FALSE, missing=NaN)
  
  # strongest factors
  new_preds <- c()
  for (feature in feature_names) {
    live_data_trsf <- live_data
    # neutralize feature to population mean
    if (sum(is.na(train_data[,feature])) > (nrow(train_data) / 2)) {
      live_data_trsf[,feature] <- NA
    } else {
      live_data_trsf[,feature] <- mean(train_data[,feature], na.rm = TRUE)
    }
    predictions <- predict(object=xgb_model, data.matrix(live_data_trsf[,feature_names]),
                           outputmargin=FALSE, missing=NaN)
    new_preds <- cbind(new_preds, original_predictions - predictions)
  }
  
  positive_features <- c()
  negative_features <- c()
  
  feature_effect_df <- data.frame(new_preds)
  names(feature_effect_df) <- c(feature_names)
  
  for (pred_id in seq(nrow(feature_effect_df))) {
    vector_vals <- feature_effect_df[pred_id,]
    vector_vals <- vector_vals[,!is.na(vector_vals)]
    positive_features <- rbind(positive_features, 
                               c(colnames(vector_vals)[order(vector_vals, 
                                                             decreasing=TRUE)][1:number_of_factors]))
    negative_features <- rbind(negative_features, 
                               c(colnames(vector_vals)[order(vector_vals,                                                         decreasing=FALSE)][1:number_of_factors]))
  }
  
  positive_features <- data.frame(positive_features)
  names(positive_features) <- paste0('Pos_', names(positive_features))
  negative_features <- data.frame(negative_features)
  names(negative_features) <- paste0('Neg_', names(negative_features))
  
  return(data.frame(original_predictions, positive_features, negative_features))
}

xg.train <- train	
xg.test <- test

xg.train$Establishment_success <-ifelse(xg.train$Establishment_success=='1', 1,0)
xg.test$Establishment_success <-ifelse(xg.test$Establishment_success=='1', 1,0)

outcome_name <- 'Establishment_success'

preds <- observation_level_variable_importance(train_data = xg.train, 
                                               live_data = xg.test, 
                                               outcome_name = outcome_name,
                                               number_of_factors=2)
preds <- preds[order(-preds$original_predictions),]

## Survival Predictive Strength

kable(preds[1:5,2:3])
kable(preds[1:5,4:5])

## NonSurvival Predictive Strength

nbrr <- nrow(preds)
kable(preds[(nbrr-5):nbrr,2:3])
kable(preds[(nbrr-5):nbrr,4:5])

i <- sapply(preds, is.factor)
preds[i] <- lapply(preds[i], as.character)

## Reports


## restore categorical data
df.cmpl <- mice::complete(df.mice ,1)
df.cmpl[, dvar] <- factor(df.orig[, df.orig.dvar])

for(i in (df.orig.dvar +1) : length(df.orig)) {
  df.cmpl[, (length(df.cmpl) + 1)] <- factor(df.orig[, i])
  names(df.cmpl)[length(df.cmpl)] <- names(df.orig)[i]
}

# density plot
ggplot(df.cmpl, aes(x=Propagule_pressure,  color=Establishment_success)) + 
  geom_density(size=1.0) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        axis.title.x=element_blank(),
        plot.title = element_text(hjust = 0.5)) +
  labs(color = 'Survived') +
  scale_color_manual(labels = c("No", "Yes"), values = c("#164E80", "#E69F00")) +
  ggtitle('Propagule_pressure Survivors vs Nonsurvivors')

pos_x1_cnt <- preds %>% 
  group_by(gr=cut(original_predictions, breaks= seq(0, 1, by = 0.25)) ) %>% 
  dplyr::count(Pos_X1) %>%
  arrange(desc(gr),-n)

pos_x2_cnt <- preds %>% 
  group_by(gr=cut(original_predictions, breaks= seq(0, 1, by = 0.25)) ) %>% 
  dplyr::count(Pos_X2) %>%
  arrange(desc(gr),-n)

par(mfrow=c(1, 1))

ggplot(pos_x1_cnt,aes(Pos_X1,n,fill=gr)) +
  geom_bar(stat="identity",position='dodge') +
  theme(axis.text.x = element_text(angle = 90),
        axis.title.x=element_blank(),
        plot.title = element_text(hjust = 0.5))  +
  ggtitle('Pos_X1')

ggplot(pos_x2_cnt,aes(Pos_X2,n,fill=gr)) +
  geom_bar(stat="identity",position='dodge') +
  theme(axis.text.x = element_text(angle = 90),
        axis.title.x=element_blank(),
        plot.title = element_text(hjust = 0.5)) +
  ggtitle('Pos_X2') 

TSur01 <- ggplot(arrange(top_n(df.cmpl, 20, Propagule_pressure), -Propagule_pressure),
                 aes( Species_name, Propagule_pressure, fill=Establishment_success)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        axis.title.x=element_blank(),
        plot.title = element_text(hjust = 0.5)) +
  scale_fill_manual(values = c('#A5D8DD', '#1C4E80'), name = "Survived") +
  scale_color_manual(labels = c("No", "Yes"), values = c('#A5D8DD', '#1C4E80')) +
  geom_bar(stat="identity") + 
  geom_text(aes(y =ave( Propagule_pressure, Species_name, FUN = mean), label='')) + 
  coord_flip() +
  ggtitle('Top Survivors by Propagule Pressure')

TSur02 <- ggplot(arrange(top_n(df.cmpl, 80, Mean_neonate_body_neonate_body_mass_g), -Mean_neonate_body_neonate_body_mass_g),
                 aes( Species_name, Mean_neonate_body_neonate_body_mass_g, fill=Establishment_success)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        axis.title.x=element_blank(),
        plot.title = element_text(hjust = 0.5)) +
  scale_fill_manual(values = c('#A5D8DD', '#1C4E80'), name = "Survived") +
  scale_color_manual(labels = c("No", "Yes"), values = c('#A5D8DD', '#1C4E80')) +
  geom_bar(stat="identity") + 
  geom_text(aes(y =ave( Mean_neonate_body_neonate_body_mass_g, Species_name, FUN = mean), label='')) + 
  coord_flip() +
  ggtitle('Top Survivors by Mean Neonate Body Mass')

TSur03 <- ggplot(arrange(top_n(df.cmpl, 5, Mean_interbirth_interval), -Mean_interbirth_interval),
                 aes( Species_name, Mean_interbirth_interval, fill=Establishment_success)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        axis.title.x=element_blank(),
        plot.title = element_text(hjust = 0.5)) +
  scale_fill_manual(values = c('#A5D8DD', '#1C4E80'), name = "Survived") +
  scale_color_manual(labels = c("No", "Yes"), values = c('#A5D8DD', '#1C4E80')) +
  geom_bar(stat="identity") + 
  geom_text(aes(y =ave( Mean_interbirth_interval, Species_name, FUN = mean), label='')) + 
  coord_flip() +
  ggtitle('Top Survivors by Mean Interbirth Interval')

TSur04 <- ggplot(arrange(top_n(df.cmpl, 100, Mean_home_range_size_km2), -Mean_home_range_size_km2),
                 aes( Species_name, Mean_home_range_size_km2, fill=Establishment_success)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        axis.title.x=element_blank(),
        plot.title = element_text(hjust = 0.5)) +
  scale_fill_manual(values = c('#A5D8DD', '#1C4E80'), name = "Survived") +
  scale_color_manual(labels = c("No", "Yes"), values = c('#A5D8DD', '#1C4E80')) +
  geom_bar(stat="identity") + 
  geom_text(aes(y =ave( Mean_home_range_size_km2, Species_name, FUN = mean), label='')) + 
  coord_flip() +
  ggtitle('Top Survivors by Mean Home Range Size')

grid.arrange(TSur01, TSur02, TSur03, TSur04,ncol=2, nrow=2)

## spit into successful/unsuccessful
df0 <- subset(df.cmpl, Establishment_success == 0)
df1 <- subset(df.cmpl, Establishment_success == 1)

year.tbl <- df.cmpl[,c(1:dvar, 25)] %>%
  group_by(Year_introduction) %>%
  summarise(counts = n())

intro.date <- as.Date(lubridate::ymd(year.tbl$Year_introduction, truncated = 2L))

ggplot(year.tbl, aes(x=intro.date, y=counts)) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        axis.title.x=element_blank(),
        plot.title = element_text(hjust = 0.5)) +
  geom_bar(stat = "identity", fill="firebrick") + 
  scale_x_date(breaks =intro.date[seq(1, length(intro.date), by = 10)],
               labels = date_format("%Y")) +
  ggtitle('Introductions by Year')

ppress.tbl <- df.cmpl[,c(1:dvar, 25)] %>%
  group_by(Year_introduction) %>%
  summarise(counts = mean(Propagule_pressure))

ppress.date <- as.Date(lubridate::ymd(ppress.tbl$Year_introduction, truncated = 2L))

ggplot(year.tbl, aes(x=ppress.date, y=counts)) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        axis.title.x = element_blank(),
        plot.title = element_text(hjust = 0.5)) +
  geom_bar(stat="identity", fill="firebrick") + 
  scale_x_date(breaks = ppress.date[seq(1, length(ppress.date), by = 10)],
               labels = date_format("%Y")) +
  ggtitle('Propagule Pressure by Year')

## species comparison table
spec.no <- df.cmpl %>%
  filter(Establishment_success == 0) %>%
  group_by(Species_name  ) %>%
  summarise(ncnt = n(), nppress= mean(Propagule_pressure))

spec.yes <- df.cmpl %>%
  filter(Establishment_success == 1) %>%
  group_by(Species_name  ) %>%
  summarise(ycnt = n(), yppress= mean(Propagule_pressure))

spec.yn <- full_join(spec.yes,spec.no)
spec.yn[is.na(spec.yn)] <- 0
spec.yn <- spec.yn %>%
  mutate( pctSur = ycnt / (ycnt + ncnt))

spec.ynTop <- spec.yn %>%
  filter(ycnt >= 10) %>%
  mutate( pctSur = ycnt / (ycnt + ncnt))

spec.ynTop<- spec.ynTop %>%
  arrange(desc(pctSur)) %>%
  mutate(lab.ypos = cumsum(ycnt) - 0.5*ycnt)

mycols <- c("#0073C2FF", "#EFC000FF", "#868686FF", "#CD534CFF",
            "#A073C2FF", "#BFC000FF", "#168686FF", "#2D534CFF",
            "#3073C2FF", "#4FC000FF", "#168686FF", "#CD534CFF", "#758686FF")

ggplot(spec.ynTop, aes(x = "", y = ycnt, fill = Species_name)) +
  geom_bar(width = 1, stat = "identity", color = "white") +
  coord_polar("y", start = 0)+
  geom_text(aes(y = lab.ypos, label = Species_name), color = "white")+
  scale_fill_manual(values = mycols) +
  theme(plot.title = element_text(hjust = 0.5)) +
  ggtitle('Top Survivors')

spec.ynBot <- spec.yn %>%
  filter(ycnt == 0) %>%
  mutate( pctSur = ycnt / (ycnt + ncnt))

spec.ynBot<- spec.ynBot %>%
  arrange(desc(Species_name)) %>%
  mutate(lab.ypos = cumsum(ncnt) - 0.5*ncnt)

ggplot(spec.ynBot, aes(x = "", y = ncnt, fill = Species_name)) +
  geom_bar(width = 1, stat = "identity", color = "white") +
  coord_polar("y", start = 0)+
  geom_text(aes(y = lab.ypos, label = Species_name), color = "white")+
  scale_fill_manual(values = mycols) +
  theme(plot.title = element_text(hjust = 0.5)) +
  ggtitle('100% Non Survivors')

ggplot(data=spec.ynTop, aes(x=ycnt, y=yppress, fill=Species_name)) +
  geom_bar(stat="identity")+
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Introductions", y = "Propagule Pressure") +
  ggtitle('Survivors - Propagule Pressure By Introductions')

ggplot(data=spec.ynBot, aes(x=ncnt, y=nppress, fill=Species_name)) +
  geom_bar(stat="identity")+
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(x = "Introductions", y = "Propagule Pressure") +
  ggtitle('NonSurvivors - Propagule Pressure By Introductions')

