setwd("D:/OneDrive/前桌面")
data1<-read.csv("RETRO.csv",header = T)
data2<-read.csv("OBSE.csv",header = T)

library(Boruta)      
library(caret)       
library(randomForest)
library(e1071)       
library(xgboost)     
library(class)       
library(nnet)        
library(pROC)        
library(MLmetrics)   
library(ggplot2)
library(dplyr)
library(tidyr)
library(RColorBrewer)
library(glmnet)

data1$luster_3 <- as.factor(data1$luster_3)
data2$luster_3<- as.factor(data2$luster_3)
set.seed(123)
train_index <- createDataPartition(
  y = data1$luster_3,  
  p = 0.7,                  
  list = FALSE
)

derivation_set <- data1[train_index, ]
validation_set <- data1[-train_index, ]


selected_features<-c("T","HDL","BMI0","LH_FSH","HOMA","TG")



derivation_features <- derivation_set[, c("luster_3", selected_features)]
validation_features <- validation_set[, c("luster_3", selected_features)]
external_features <- data2[, c("luster_3", selected_features)]
levels(derivation_features$luster_3) <- make.names(levels(derivation_features$luster_3))
levels(validation_features$luster_3) <- make.names(levels(validation_features$luster_3))
levels(external_features$luster_3) <- make.names(levels(external_features$luster_3))

cat("\n=== 模型训练与调参阶段（10折交叉验证） ===\n")

ctrl <- trainControl(
  method = "cv",           
  number = 10,            
  classProbs = TRUE,       
  summaryFunction = multiClassSummary,  
  savePredictions = "final",
  verboseIter = TRUE,      
  allowParallel = TRUE     
)

cat("\n训练多个模型...\n")
ctrl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = multiClassSummary,
  savePredictions = "final"
)

cat("\n=== 定义10折交叉验证方案 ===\n")
cv_folds <- createFolds(
  y = derivation_features$luster_3,  
  k = 10,            
  returnTrain = TRUE,
  list = TRUE        
)

cat("创建了", length(cv_folds), "个交叉验证折叠\n")
for(i in 1:length(cv_folds)) {
  train_size <- length(cv_folds[[i]])
  val_size <- nrow(derivation_features) - train_size
  cat(sprintf("折叠%d: 训练集%d个样本, 验证集%d个样本\n", 
              i, train_size, val_size))
}

cat("\n=== 定义超参数搜索空间 ===\n")
hyperparam_grid <- expand.grid(
  mtry = c(2, 4, 6, 8, 10),      
  ntree = c(100, 200, 300),      
  nodesize = c(1, 5, 10)         
)

cat("超参数组合总数:", nrow(hyperparam_grid), "\n")
print(head(hyperparam_grid))

cat("\n=== 开始交叉验证超参数优化 ===\n")
cv_results <- data.frame()
set.seed(123)
for(i in 1:nrow(hyperparam_grid)) {
  mtry_val <- hyperparam_grid$mtry[i]
  ntree_val <- hyperparam_grid$ntree[i]
  nodesize_val <- hyperparam_grid$nodesize[i]

  fold_performance <- numeric(10)  
  
  for(fold in 1:10) {
    train_idx <- cv_folds[[fold]]
    val_idx <- setdiff(1:nrow(derivation_features), train_idx)
    
    train_fold <- derivation_features[train_idx, ]
    val_fold <- derivation_features[val_idx, ]
 
    rf_temp <- randomForest(
      luster_3 ~ .,                    
      data = train_fold,
      mtry = mtry_val,
      ntree = ntree_val,
      nodesize = nodesize_val,
      importance = FALSE,             
      do.trace = FALSE
    )
 
    pred <- predict(rf_temp, newdata = val_fold)

    accuracy <- mean(pred == val_fold$luster_3, na.rm = TRUE)
    fold_performance[fold] <- accuracy
  }

  avg_performance <- mean(fold_performance)
  sd_performance <- sd(fold_performance)

  cv_results <- rbind(cv_results, data.frame(
    mtry = mtry_val,
    ntree = ntree_val,
    nodesize = nodesize_val,
    avg_accuracy = avg_performance,
    sd_accuracy = sd_performance,
    min_accuracy = min(fold_performance),
    max_accuracy = max(fold_performance)
  ))
}

cat("\n=== 交叉验证结果 ===\n")
cat("前10个最佳超参数组合:\n")
cv_results_sorted <- cv_results[order(-cv_results$avg_accuracy), ]
print(head(cv_results_sorted, 10))

best_params <- cv_results_sorted[1, ]
cat("\n=== 最佳超参数组合 ===\n")
print(best_params)

cat("\n=== 训练最终模型（使用最佳参数） ===\n")
final_RF_model <- randomForest(
  luster_3 ~ .,                    # 使用luster_3作为目标变量
  data = derivation_features,
  mtry = best_params$mtry,
  ntree = best_params$ntree,
  nodesize = best_params$nodesize,
  importance = TRUE,               # 计算特征重要性
  proximity = FALSE,
  do.trace = FALSE
)
cat("模型参数:\n")
cat("mtry:", best_params$mtry, "\n")
cat("ntree:", best_params$ntree, "\n")
cat("nodesize:", best_params$nodesize, "\n")

cat("\n=== 内部验证集评估 ===\n")
val_predictions <- predict(final_RF_model, newdata = validation_features)
val_accuracy <- mean(val_predictions == validation_features$luster_3, na.rm = TRUE)
val_accuracy

cat("\n=== 内部验证集评估 ===\n")
val_predictions <- predict(final_RF_model, newdata = external_features )
val_accuracy <- mean(val_predictions == external_features $luster_3, na.rm = TRUE)
val_accuracy
mtry = best_params$mtry
ntree = best_params$ntree
nodesize = best_params$nodesize



X_train <- as.matrix(derivation_features[, -which(colnames(derivation_features) == "luster_3")])
y_train <- derivation_features$luster_3

X_val <- as.matrix(validation_features[, -which(colnames(validation_features) == "luster_3")])
y_val <- validation_features$luster_3

X_ext <- as.matrix(external_features[, -which(colnames(external_features) == "luster_3")])
y_ext <- external_features$luster_3

tune_grid <- expand.grid(
  alpha = seq(0, 1, by = 0.2),  # 0, 0.2, 0.4, 0.6, 0.8, 1
  lambda = 10^seq(-4, 0, length = 20)  # 从10^-4到1
)

cat("超参数组合总数:", nrow(tune_grid), "\n")

cat("\n=== 开始逻辑回归超参数优化（10折交叉验证） ===\n")

library(doParallel)


set.seed(123)
cat("\n开始训练glmnet模型...\n")
set.seed(123)  # 确保可重复性
glmnet_model <- train(
  x = X_train,
  y = y_train,
  method = "glmnet",
  family = "multinomial",  # 多分类
  trControl = ctrl,
  tuneGrid = tune_grid,
  metric = "Accuracy",      # 主要优化指标
  maximize = TRUE,          # 最大化准确率
  standardize = TRUE,       # 标准化特征
  maxit = 100000            # 增加最大迭代次数确保收敛
)

cat("\n=== 超参数优化结果 ===\n")
cat("最佳参数组合:\n")
print(glmnet_model$bestTune)

cat("\n=== 可视化超参数优化结果 ===\n")

results_df <- glmnet_model$results
cat("\n=== 使用最佳参数训练最终模型 ===\n")

best_alpha <- glmnet_model$bestTune$alpha
best_lambda <- glmnet_model$bestTune$lambda

cat("最佳参数: alpha =", best_alpha, ", lambda =", 
    format(best_lambda, scientific = TRUE), "\n")

final_model <- glmnet_model

cat("最终模型训练完成\n")

cat("\n=== 创建模型评估函数 ===\n")

evaluate_glmnet_model_caret <- function(model, X, y_true, dataset_name = "数据集") {
 
  X_df <- as.data.frame(X)

  pred_class <- predict(model, newdata = X_df)

  pred_prob <- predict(model, newdata = X_df, type = "prob")

  y_true <- factor(y_true)

  pred_class <- factor(pred_class, levels = levels(y_true))
 
  cm <- confusionMatrix(pred_class, y_true)

  metrics <- cm$byClass

  macro_metrics <- data.frame(
    Accuracy = cm$overall["Accuracy"],
    Kappa = cm$overall["Kappa"],
    Macro_Precision = mean(metrics[, "Precision"], na.rm = TRUE),
    Macro_Recall = mean(metrics[, "Recall"], na.rm = TRUE),
    Macro_F1 = mean(metrics[, "F1"], na.rm = TRUE)
  )
  
  y_true_numeric <- as.numeric(y_true) - 1

  prob_matrix <- as.matrix(pred_prob)
  colnames(prob_matrix) <- colnames(pred_prob)
 
  if (ncol(prob_matrix) != length(levels(y_true))) {
    cat("警告: 概率矩阵的列数与类别数不匹配\n")
    macro_metrics$LogLoss <- NA
  } else {
    macro_metrics$LogLoss <- MLmetrics::LogLoss(
      y_true = y_true_numeric, 
      y_pred = prob_matrix
    )
  }
  
  cat("\n=== ", dataset_name, "评估结果 ===\n")
  cat("准确率:", round(macro_metrics$Accuracy, 4), "\n")
  cat("Kappa系数:", round(macro_metrics$Kappa, 4), "\n")
  cat("宏平均F1:", round(macro_metrics$Macro_F1, 4), "\n")
  cat("LogLoss:", round(macro_metrics$LogLoss, 4), "\n")
 
  cat("\n每个类别的详细指标:\n")
  print(metrics[, c("Precision", "Recall", "F1")])
  
  return(list(
    predictions = pred_class,
    probabilities = pred_prob,
    confusion_matrix = cm,
    metrics = macro_metrics,
    per_class_metrics = metrics
  ))
}

cat("\n=== 模型性能评估 ===\n")

cat("\n--- 训练集（derivation_features）评估 ---\n")
train_results <- evaluate_glmnet_model_caret(final_model, X_train, y_train, "训练集")

cat("\n--- 内部验证集（validation_features）评估 ---\n")
val_results <- evaluate_glmnet_model_caret(final_model, X_val, y_val, "内部验证集")

cat("\n--- 外部验证集（external_features）评估 ---\n")
ext_results <- evaluate_glmnet_model_caret(final_model, X_ext, y_ext, "外部验证集")

cat("\n=== 逻辑回归模型最佳参数 ===\n")
print(glmnet_model$bestTune)


cat("\n=== 设置SVM超参数网格 ===\n")

linear_svm_grid <- expand.grid(
  C = 10^seq(-3, 3, length = 7)  # 正则化参数：0.001, 0.01, 0.1, 1, 10, 100, 1000
)
rbf_svm_grid <- expand.grid(
  sigma = 10^seq(-3, 1, length = 5),  # 核宽度参数：0.001, 0.01, 0.1, 1, 10
  C = 10^seq(-2, 2, length = 5)       # 正则化参数：0.01, 0.1, 1, 10, 100
)

cat("线性SVM网格大小:", nrow(linear_svm_grid), "个组合\n")
cat("RBF SVM网格大小:", nrow(rbf_svm_grid), "个组合\n\n")

cat("\n=== 训练线性核SVM模型 ===\n")

cat("开始训练线性SVM...\n")

# 使用caret训练线性SVM
linear_svm_model <- train(
  x = X_train,
  y = y_train,
  method = "svmLinear",          # 线性核SVM
  trControl = ctrl,              # 使用您已有的ctrl设置（10折CV）
  tuneGrid = linear_svm_grid,    # 线性SVM网格
  metric = "Accuracy",           # 优化准确率
  maximize = TRUE,               # 最大化准确率
  preProcess = c("center", "scale"),  # 标准化数据（SVM需要）
  verbose = FALSE
)

cat("线性SVM最佳参数:\n")
print(linear_svm_model$bestTune)

cat("\n=== 训练径向基核(RBF) SVM模型 ===\n")

cat("开始训练RBF SVM...\n")

rbf_svm_model <- train(
  x = X_train,
  y = y_train,
  method = "svmRadial",          # 径向基核SVM
  trControl = ctrl,              # 使用您已有的ctrl设置
  tuneGrid = rbf_svm_grid,       # RBF SVM网格
  metric = "Accuracy",           # 优化准确率
  maximize = TRUE,               # 最大化准确率
  preProcess = c("center", "scale"),  # 标准化数据
  verbose = FALSE
)
cat("RBF SVM最佳参数:\n")
print(rbf_svm_model$bestTune)

cat("\n=== 训练多项式核SVM模型 ===\n")

# 定义多项式核SVM网格
poly_svm_grid <- expand.grid(
  degree = c(2, 3),              # 多项式次数：2次或3次
  scale = c(0.1, 1),             # 缩放参数
  C = 10^seq(-2, 2, length = 5)  # 正则化参数
)

cat("多项式SVM网格大小:", nrow(poly_svm_grid), "个组合\n")

cat("开始训练多项式SVM...\n")

poly_svm_model <- train(
  x = X_train,
  y = y_train,
  method = "svmPoly",            # 多项式核SVM
  trControl = ctrl,              # 使用您已有的ctrl设置
  tuneGrid = poly_svm_grid,      # 多项式SVM网格
  metric = "Accuracy",           # 优化准确率
  maximize = TRUE,               # 最大化准确率
  preProcess = c("center", "scale"),  # 标准化数据
  verbose = FALSE
)


cat("多项式SVM最佳参数:\n")
print(poly_svm_model$bestTune)

cat("\n=== 比较不同核函数的SVM性能 ===\n\n")

svm_models <- list(
  "线性SVM" = linear_svm_model,
  "RBF SVM" = rbf_svm_model,
  "多项式SVM" = poly_svm_model
)

svm_comparison <- data.frame()

for(model_name in names(svm_models)) {
  model <- svm_models[[model_name]]
  
  svm_comparison <- rbind(svm_comparison, data.frame(
    Model = model_name,
    Kernel = ifelse(model_name == "线性SVM", "Linear",
                    ifelse(model_name == "RBF SVM", "RBF", "Polynomial")),
    Best_Params = paste(names(model$bestTune), "=", 
                        sapply(model$bestTune, function(x) 
                          ifelse(is.numeric(x), round(x, 4), x)), 
                        collapse = ", "),
    CV_Accuracy = round(max(model$results$Accuracy), 4),
    CV_Kappa = round(model$results$Kappa[which.max(model$results$Accuracy)], 4),
    CV_LogLoss = round(model$results$logLoss[which.max(model$results$Accuracy)], 4)
  ))
}

cat("SVM模型交叉验证性能比较:\n")
print(svm_comparison, row.names = FALSE)


cat("\n=== 选择最佳SVM模型 ===\n\n")

best_svm_idx <- which.max(svm_comparison$CV_Accuracy)
best_svm_name <- svm_comparison$Model[best_svm_idx]
best_svm_model <- svm_models[[best_svm_name]]

cat("最佳SVM模型:", best_svm_name, "\n")
cat("交叉验证准确率:", svm_comparison$CV_Accuracy[best_svm_idx], "\n")
cat("最佳参数:", svm_comparison$Best_Params[best_svm_idx], "\n\n")

cat("训练最终SVM模型（使用最佳参数）...\n")

final_svm_model <- best_svm_model$finalModel
cat("最终SVM模型训练完成\n")

cat("\n=== 创建SVM模型评估函数 ===\n")

evaluate_svm_model <- function(model, X, y_true, dataset_name = "数据集") {
  
  pred_class <- predict(model, newdata = X)
 
  tryCatch({
    pred_prob <- predict(model, newdata = X, type = "prob")
  }, error = function(e) {
    pred_prob <- NULL
    cat("注意: 该SVM模型不支持概率预测\n")
  })

  cm <- confusionMatrix(pred_class, y_true)

  metrics <- cm$byClass

  macro_metrics <- data.frame(
    Accuracy = cm$overall["Accuracy"],
    Kappa = cm$overall["Kappa"],
    Macro_Precision = mean(metrics[, "Precision"], na.rm = TRUE),
    Macro_Recall = mean(metrics[, "Recall"], na.rm = TRUE),
    Macro_F1 = mean(metrics[, "F1"], na.rm = TRUE)
  )
  
  cat("\n=== ", dataset_name, "评估结果 ===\n")
  cat("准确率:", round(macro_metrics$Accuracy, 4), "\n")
  cat("Kappa系数:", round(macro_metrics$Kappa, 4), "\n")
  cat("宏平均F1:", round(macro_metrics$Macro_F1, 4), "\n")
  
  if(!is.null(pred_prob)) {
    logloss_value <- MLmetrics::LogLoss(y_true = as.numeric(y_true) - 1, 
                                        y_pred = as.matrix(pred_prob))
    cat("LogLoss:", round(logloss_value, 4), "\n")
    macro_metrics$LogLoss <- logloss_value
  }
  
  cat("\n每个类别的详细指标:\n")
  print(metrics[, c("Precision", "Recall", "F1")])
  
  return(list(
    predictions = pred_class,
    probabilities = pred_prob,
    confusion_matrix = cm,
    metrics = macro_metrics,
    per_class_metrics = metrics
  ))
}


cat("\n=== SVM模型性能评估 ===\n")
library(kernlab)

cat("\n--- 训练集评估 ---\n")
svm_train_results <- evaluate_svm_model(linear_svm_model, X_train, y_train, "训练集")

cat("\n--- 内部验证集评估 ---\n")
svm_val_results <- evaluate_svm_model(linear_svm_model, X_val, y_val, "内部验证集")

cat("\n--- 外部验证集评估 ---\n")
svm_ext_results <- evaluate_svm_model(linear_svm_model, X_ext, y_ext, "外部验证集")


set.seed(123)

nb_grid <- expand.grid(
  laplace = c(0, 0.5, 1, 1.5, 2),
  usekernel = c(TRUE, FALSE),
  adjust = c(0.5, 1, 1.5, 2)
)

nb_control <- trainControl(
  method = "repeatedcv",
  number = 10,          # 10折交叉验证
  repeats = 3,          # 重复3次
  classProbs = TRUE,
  summaryFunction = multiClassSummary,
  savePredictions = "final",
  verboseIter = TRUE,
  search = "grid"
)

nb_model <- train(
  luster_3 ~ .,
  data = derivation_features,
  method = "naive_bayes",
  trControl = nb_control,
  tuneGrid = nb_grid,
  metric = "Accuracy",  # 使用准确率作为优化指标
  preProcess = c("center", "scale")  # 标准化数据
)

print("朴素贝叶斯最佳超参数:")
print(nb_model$bestTune)

print("朴素贝叶斯交叉验证结果:")
print(nb_model$results)

nb_der_pred <- predict(nb_model, newdata = derivation_features)
nb_der_prob <- predict(nb_model, newdata = derivation_features, type = "prob")

nb_der_cm <- confusionMatrix(nb_der_pred, derivation_features$luster_3)
print("朴素贝叶斯内部验证集混淆矩阵:")
print(nb_der_cm)

nb_val_pred <- predict(nb_model, newdata = validation_features)
nb_val_prob <- predict(nb_model, newdata = validation_features, type = "prob")

nb_val_cm <- confusionMatrix(nb_val_pred, validation_features$luster_3)
print("朴素贝叶斯内部验证集混淆矩阵:")
print(nb_val_cm)

nb_ext_pred <- predict(nb_model, newdata = external_features)
nb_ext_prob <- predict(nb_model, newdata = external_features, type = "prob")

nb_ext_cm <- confusionMatrix(nb_ext_pred, external_features$luster_3)
print("朴素贝叶斯外部验证集混淆矩阵:")
print(nb_ext_cm)

set.seed(123)

knn_grid <- expand.grid(
  k = seq(3, 31, by = 2)  # 尝试奇数k值以避免平局
)

knn_control <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = multiClassSummary,
  savePredictions = "final",
  verboseIter = TRUE,
  search = "grid"
)

knn_model <- train(
  luster_3 ~ .,
  data = derivation_features,
  method = "knn",
  trControl = knn_control,
  tuneGrid = knn_grid,
  metric = "Accuracy",
  preProcess = c("center", "scale"),  # KNN特别需要标准化
  # 可选: 添加距离权重
  tuneLength = 15  # 如果不使用自定义网格，可以指定调优参数数量
)

print("KNN最佳超参数 (k值):")
print(knn_model$bestTune)

print("KNN交叉验证结果:")
print(knn_model$results)

knn_val_pred <- predict(knn_model, newdata = validation_features)
knn_val_prob <- predict(knn_model, newdata = validation_features, type = "prob")

knn_val_cm <- confusionMatrix(knn_val_pred, validation_features$luster_3)
print("KNN内部验证集混淆矩阵:")
print(knn_val_cm)

knn_ext_pred <- predict(knn_model, newdata = external_features)
knn_ext_prob <- predict(knn_model, newdata = external_features, type = "prob")

knn_ext_cm <- confusionMatrix(knn_ext_pred, external_features$luster_3)
print("KNN外部验证集混淆矩阵:")
print(knn_ext_cm)
knn_params <- knn_model$bestTune
k <- knn_params$k
knn_str <- paste("Algorithm: kd_tree, n_neighbors:", k, ", weights: uniform")

best_params_list <- list()

rf_params <- list(
  Model = "RF",
  mtry = best_params$mtry,
  ntree = best_params$ntree,
  nodesize = best_params$nodesize
)

glmnet_params <- list(
  Model = "Logistic Regression (glmnet)",
  Alpha = best_alpha,
  Lambda = best_lambda
)

svm_params <- list(
  Model = "SVM",
  Kernel = ifelse(best_svm_name == "线性SVM", "linear",
                  ifelse(best_svm_name == "RBF SVM", "rbf", "polynomial")),
  C = best_svm_model$bestTune$C,
  # 根据核函数类型添加其他参数
  Gamma = ifelse("sigma" %in% names(best_svm_model$bestTune), 
                 best_svm_model$bestTune$sigma, NA),
  Degree = ifelse("degree" %in% names(best_svm_model$bestTune), 
                  best_svm_model$bestTune$degree, NA)
)

knn_params <- list(
  Model = "kNN",
  k = knn_model$bestTune$k,
  Algorithm = "kd_tree",  # 需要在knn训练时指定
  Weights = "distance"    # 需要在knn训练时指定
)

nb_params <- list(
  Model = "Naive Bayes",
  Laplace = nb_model$bestTune$laplace,
  UseKernel = nb_model$bestTune$usekernel,
  Adjust = nb_model$bestTune$adjust
)

params_table <- data.frame(
  ML_Model = c("kNN", "Random Forest", "SVM", "Logistic Regression", "Naive Bayes"),
  Best_Parameters = c(
    sprintf("k: %d, Algorithm: kd_tree, Weights: distance", knn_model$bestTune$k),
    sprintf("mtry: %d, ntree: %d, nodesize: %d", 
            best_params$mtry, best_params$ntree, best_params$nodesize),
    sprintf("Kernel: %s, C: %.3f", 
            ifelse(best_svm_name == "线性SVM", "linear",
                   ifelse(best_svm_name == "RBF SVM", "rbf", "polynomial")),
            best_svm_model$bestTune$C),
    sprintf("Alpha: %.3f, Lambda: %.4f", best_alpha, best_lambda),
    sprintf("Laplace: %.1f, usekernel: %s, adjust: %.1f",
            nb_model$bestTune$laplace,
            nb_model$bestTune$usekernel,
            nb_model$bestTune$adjust)
  )
)

write.csv(params_table, "Table_S4_Hyperparameters.csv", row.names = FALSE)





