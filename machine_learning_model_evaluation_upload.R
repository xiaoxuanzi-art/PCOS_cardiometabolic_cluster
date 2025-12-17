evaluate_model_comprehensive <- function(model, model_name, 
                                         X_train, y_train,
                                         X_val, y_val,
                                         X_ext, y_ext,
                                         model_type = "auto") {  # 改为自动检测
  
  cat("\n=== ", model_name, "模型详细评估 ===\n")

  if (model_type == "auto") {
    model_class <- class(model)[1]
    
    if (model_class == "train") {
      model_type <- "caret"
    } else if (model_class == "glmnet") {
      model_type <- "glmnet"
    } else if (model_class == "randomForest") {
      model_type <- "randomForest"
    } else if (model_class == "svm" || model_class == "ksvm") {
      model_type <- "svm"
    } else if (model_class == "naive_bayes") {
      model_type <- "naive_bayes"
    } else {
      model_type <- "caret"
    }
    
    cat("自动检测模型类型:", model_type, "\n")
  }
  
  results <- list(model_name = model_name, model_type = model_type)
  datasets <- list(
    "Derivation" = list(X = X_train, y = y_train),
    "Validation" = list(X = X_val, y = y_val),
    "External" = list(X = X_ext, y = y_ext)
  )
  
  for(dataset_name in names(datasets)) {
    X <- datasets[[dataset_name]]$X
    y_true <- datasets[[dataset_name]]$y
    y_true <- factor(y_true)
    if(model_type == "caret") {
      X_df <- as.data.frame(X)
      pred_class <- predict(model, newdata = X_df)
      pred_prob <- tryCatch({
        predict(model, newdata = X_df, type = "prob")
      }, error = function(e) {
        cat("警告: 无法获取概率预测:", e$message, "\n")
        NULL
      })
    } else if(model_type == "glmnet") {
      if (inherits(model, "glmnet")) {
        pred_prob <- predict(model, newx = X, type = "response")[, , 1]
        pred_class_idx <- apply(pred_prob, 1, which.max)
        class_names <- colnames(pred_prob)
        if (is.null(class_names)) {
          class_names <- paste0("Class", 1:ncol(pred_prob))
        }
        pred_class <- factor(class_names[pred_class_idx], levels = levels(y_true))
      } else {
        stop("模型类型指定为glmnet，但实际不是glmnet对象")
      }
    } else if(model_type == "randomForest") {
      X_df <- as.data.frame(X)
      pred_class <- predict(model, newdata = X_df)
      pred_prob <- tryCatch({
        predict(model, newdata = X_df, type = "prob")
      }, error = function(e) {
        cat("警告: 无法获取概率预测:", e$message, "\n")
        NULL
      })
    } else if(model_type == "svm") {
      X_df <- as.data.frame(X)
      pred_class <- predict(model, newdata = X_df)
      pred_prob <- tryCatch({
        predict(model, newdata = X_df, type = "prob")
      }, error = function(e) {
        cat("警告: SVM可能不支持概率预测\n")
        NULL
      })
    } else if(model_type == "naive_bayes") {
      # naiveBayes模型
      X_df <- as.data.frame(X)
      pred_class <- predict(model, newdata = X_df)
      pred_prob <- tryCatch({
        predict(model, newdata = X_df, type = "prob")
      }, error = function(e) {
        cat("警告: 无法获取概率预测\n")
        NULL
      })
    } else {
      # 默认情况
      X_df <- as.data.frame(X)
      pred_class <- predict(model, newdata = X_df)
      pred_prob <- tryCatch({
        predict(model, newdata = X_df, type = "prob")
      }, error = function(e) NULL)
    }
    pred_class <- factor(pred_class, levels = levels(y_true))
    cm <- confusionMatrix(pred_class, y_true)
    accuracy <- cm$overall["Accuracy"]
    kappa <- cm$overall["Kappa"]
    class_metrics <- cm$byClass
    macro_precision <- mean(class_metrics[, "Precision"], na.rm = TRUE)
    macro_recall <- mean(class_metrics[, "Recall"], na.rm = TRUE)
    macro_f1 <- mean(class_metrics[, "F1"], na.rm = TRUE)
    weighted_precision <- sum(class_metrics[, "Precision"] * 
                                class_metrics[, "Prevalence"], na.rm = TRUE)
    weighted_recall <- sum(class_metrics[, "Recall"] * 
                             class_metrics[, "Prevalence"], na.rm = TRUE)
    weighted_f1 <- sum(class_metrics[, "F1"] * 
                         class_metrics[, "Prevalence"], na.rm = TRUE)
    logloss <- NA
    if(!is.null(pred_prob) && !is.null(y_true)) {
      try({
        if (is.matrix(pred_prob) && ncol(pred_prob) == length(levels(y_true))) {
          y_true_numeric <- as.numeric(y_true) - 1
          logloss <- MLmetrics::LogLoss(y_true = y_true_numeric, 
                                        y_pred = as.matrix(pred_prob))
        }
      }, silent = TRUE)
    }
    results[[dataset_name]] <- list(
      accuracy = accuracy,
      kappa = kappa,
      macro_precision = macro_precision,
      macro_recall = macro_recall,
      macro_f1 = macro_f1,
      weighted_precision = weighted_precision,
      weighted_recall = weighted_recall,
      weighted_f1 = weighted_f1,
      logloss = logloss,
      confusion_matrix = cm,
      class_metrics = class_metrics,
      predictions = pred_class,
      probabilities = pred_prob
    )
    cat(sprintf("\n--- %s集评估结果 ---\n", dataset_name))
    cat(sprintf("准确率: %.4f\n", accuracy))
    cat(sprintf("Kappa系数: %.4f\n", kappa))
    cat(sprintf("宏平均F1: %.4f\n", macro_f1))
    if(!is.na(logloss)) cat(sprintf("LogLoss: %.4f\n", logloss))
    cat("\n每个类别的详细指标:\n")
    print(class_metrics[, c("Precision", "Recall", "F1", "Prevalence")])
  }
  
  return(results)
}

X_train <- as.matrix(derivation_features[, -which(colnames(derivation_features) == "luster_3")])
y_train <- derivation_features$luster_3

X_val <- as.matrix(validation_features[, -which(colnames(validation_features) == "luster_3")])
y_val <- validation_features$luster_3

X_ext <- as.matrix(external_features[, -which(colnames(external_features) == "luster_3")])
y_ext <- external_features$luster_3
all_model_results <- list()
cat("随机森林模型评估\n")
rf_results <- evaluate_model_comprehensive(
  model = final_RF_model, 
  model_name = "随机森林",
  X_train = X_train, y_train = y_train,
  X_val = X_val, y_val = y_val,
  X_ext = X_ext, y_ext = y_ext,
  model_type = "auto"  # 自动检测
)
all_model_results[["随机森林"]] <- rf_results
cat("\n逻辑回归模型评估\n")
lr_results <- evaluate_model_comprehensive(
  model = final_model,  # 这是caret训练的glmnet模型
  model_name = "逻辑回归",
  X_train = X_train, y_train = y_train,
  X_val = X_val, y_val = y_val,
  X_ext = X_ext, y_ext = y_ext,
  model_type = "auto"  # 自动检测为caret
)
all_model_results[["逻辑回归"]] <- lr_results
cat("\nSVM模型评估\n")
svm_results <- evaluate_model_comprehensive(
  model = linear_svm_model,  # 这是caret训练的SVM模型
  model_name = "SVM",
  X_train = X_train, y_train = y_train,
  X_val = X_val, y_val = y_val,
  X_ext = X_ext, y_ext = y_ext,
  model_type = "auto"  # 自动检测
)
all_model_results[["SVM"]] <- svm_results
cat("\n朴素贝叶斯模型评估\n")
nb_results <- evaluate_model_comprehensive(
  model = nb_model,  # 这是caret训练的朴素贝叶斯
  model_name = "朴素贝叶斯",
  X_train = X_train, y_train = y_train,
  X_val = X_val, y_val = y_val,
  X_ext = X_ext, y_ext = y_ext,
  model_type = "auto"  # 自动检测
)
all_model_results[["朴素贝叶斯"]] <- nb_results
cat("\nKNN模型评估\n")
knn_results <- evaluate_model_comprehensive(
  model = knn_model,  # 这是caret训练的KNN
  model_name = "KNN",
  X_train = X_train, y_train = y_train,
  X_val = X_val, y_val = y_val,
  X_ext = X_ext, y_ext = y_ext,
  model_type = "auto"  # 自动检测
)
all_model_results[["KNN"]] <- knn_results
model_names <- names(all_model_results)
metrics_list <- c("accuracy", "kappa", "macro_f1", "logloss")
comparison_df <- data.frame(
  Model = character(),
  Dataset = character(),
  Accuracy = numeric(),
  Kappa = numeric(),
  Macro_F1 = numeric(),
  LogLoss = numeric(),
  stringsAsFactors = FALSE
)
for(model_name in model_names) {
  for(dataset in c("Derivation", "Validation", "External")) {
    if (!is.null(all_model_results[[model_name]][[dataset]])) {
      results <- all_model_results[[model_name]][[dataset]]
      
      comparison_df <- rbind(comparison_df, data.frame(
        Model = model_name,
        Dataset = dataset,
        Accuracy = round(results$accuracy, 4),
        Kappa = round(results$kappa, 4),
        Macro_F1 = round(results$macro_f1, 4),
        LogLoss = ifelse(is.na(results$logloss), NA, round(results$logloss, 4)),
        stringsAsFactors = FALSE
      ))
    }
  }
}
cat("\n=== 模型性能比较 ===\n")
print(comparison_df)
write.csv(comparison_df, "model_comparison_results.csv", row.names = FALSE)
cat("\n模型比较结果已保存到: model_comparison_results.csv\n")
saveRDS(all_model_results, "all_model_detailed_results.rds")
cat("详细模型结果已保存到: all_model_detailed_results.rds\n")

cat(sprintf("\n结果已保存到文件:\n"))
cat("1. best_model_performance_summary.csv - 最佳模型详细性能指标\n")
cat("2. all_models_comparison.csv - 所有模型比较\n")
model_performance <- data.frame(
  Model = names(all_model_results),
  Accuracy = sapply(all_model_results, function(x) x$Validation$accuracy),
  F1_Score = sapply(all_model_results, function(x) x$Validation$macro_f1),
  Kappa = sapply(all_model_results, function(x) x$Validation$kappa),
  stringsAsFactors = FALSE
)

cat("\n=== 所有模型性能比较（内部验证集）===\n")
print(model_performance)

model_performance <- model_performance[order(-model_performance$Accuracy), ]

best_model_name <- model_performance$Model[1]
best_accuracy <- model_performance$Accuracy[1]

cat("\n=== 最佳模型选择 ===\n")
cat("最佳模型:", best_model_name, "\n")
cat("内部验证集准确率:", round(best_accuracy, 4), "\n")

get_model_by_name <- function(model_name) {
  switch(model_name,
         "随机森林" = final_RF_model,
         "逻辑回归" = final_model,
         "SVM" = linear_svm_model,
         "朴素贝叶斯" = nb_model,
         "KNN" = knn_model,
         stop("未知模型名称: ", model_name)
  )
}

best_model_obj <- get_model_by_name(best_model_name)

cat("最佳模型类型:", class(best_model_obj)[1], "\n")

saveRDS(best_model_obj, "best_classification_model.rds")
saveRDS(selected_features, "selected_features.rds")

model_metadata <- list(
  model_name = best_model_name,
  accuracy = best_accuracy,
  selected_features = selected_features,
  model_class = class(best_model_obj)[1],
  timestamp = Sys.time()
)

saveRDS(model_metadata, "model_metadata.rds")

saveRDS(best_model_obj, "best_classification_model.rds")
saveRDS(selected_features, "selected_features.rds")

clinical_predictor_simple <- function(new_data, 
                                      model = best_model_obj,
                                      features = selected_features) {
  
  cat("=== PCOS心脏代谢亚类预测 ===\n")
  
  missing_features <- setdiff(features, colnames(new_data))
  if (length(missing_features) > 0) {
    stop("缺少特征：", paste(missing_features, collapse = ", "))
  }
  
  input_data <- new_data[, features, drop = FALSE]
  
  predicted_labels_X <- predict(model, newdata = input_data)
  predicted_probs <- predict(model, newdata = input_data, type = "prob")
  
  class_mapping <- c("X1" = 1, "X2" = 2, "X3" = 3)
  predicted_class <- class_mapping[predicted_labels_X]

  class_desc <- ifelse(predicted_class == 1, "SHP",
                       ifelse(predicted_class == 2, "HCRP", "MP"))
  
  hcrp_prob <- if ("X2" %in% colnames(predicted_probs)) {
    predicted_probs[, "X2"]
  } else {
    rep(NA, nrow(new_data))
  }
  
  results <- data.frame(
    Patient_ID = if ("Patient_ID" %in% colnames(new_data)) 
      new_data$Patient_ID else paste0("P", seq_len(nrow(new_data))),
    Predicted_Class = class_desc,
    Probability_HCRP = round(hcrp_prob, 4),
    stringsAsFactors = FALSE
  )
  
  cat("预测完成！\n")
  cat("样本数：", nrow(results), "\n")
  
  hcrp_count <- sum(results$Predicted_Class == "HCRP")
  cat(sprintf("HCRP亚类：%d例 (%.1f%%)\n", 
              hcrp_count, hcrp_count/nrow(results)*100))
  
  high_risk <- sum(results$Probability_HCRP >= 0.5, na.rm = TRUE)
  cat(sprintf("HCRP概率≥0.5：%d例\n", high_risk))
  
  return(results)
}
test_data<-read.csv("newdata1.csv",header = T)

simple_results <- clinical_predictor_simple(test_data)

print(simple_results)

write.csv(simple_results, "hcrp_predictions.csv", row.names = FALSE)

