################################################################################
#                                                                              #
#                    MODÉLISATION - RÉGRESSION LOGISTIQUE                      #
#              Telco Customer Churn - Pipeline Complet de ML                   #
#                   Train → Evaluate → Save → Inference                        #
#                                                                              #
################################################################################

################################################################################
# TABLE DES MATIÈRES
################################################################################
# 
# 1. CONFIGURATION ET CHARGEMENT DES DONNÉES
# 2. ENTRAÎNEMENT DU MODÈLE DE RÉGRESSION LOGISTIQUE
#    2.1 Modèle de Base (sans régularisation)
#    2.2 Modèle avec Régularisation Ridge
#    2.3 Modèle avec Régularisation Lasso
# 3. PRÉDICTIONS SUR LES ENSEMBLES (TRAIN/VAL/TEST)
# 4. ÉVALUATION DES PERFORMANCES
#    4.1 Matrice de Confusion
#    4.2 Métriques de Performance
#    4.3 Courbe ROC et AUC
#    4.4 Courbe Precision-Recall
#    4.5 Feature Importance
# 5. SAUVEGARDE DU MODÈLE ET DES TRANSFORMERS
#    5.1 Sauvegarde du Modèle
#    5.2 Sauvegarde des Métadonnées
# 6. CHARGEMENT DU MODÈLE (SIMULATION PRODUCTION)
# 7. INFÉRENCE SUR NOUVELLES DONNÉES
#    7.1 Fonction d'Inférence Complète
#    7.2 Exemple d'Utilisation
# 8. INTERPRÉTATION DU MODÈLE
#
################################################################################


################################################################################
# 1. CONFIGURATION ET CHARGEMENT DES DONNÉES
################################################################################

cat("\n================================================================================\n")
cat("                     DÉMARRAGE DE LA MODÉLISATION                            \n")
cat("================================================================================\n\n")

library(dplyr)
library(caret)
library(ggplot2)
library(pROC)
library(glmnet)
library(ROCR)
library(gridExtra)
library(broom)

set.seed(123)

cat("Step 1: Chargement des données preprocessées\n")

train_set <- read.csv("../preparedData/train_set.csv")
validation_set <- read.csv("../preparedData/validation_set.csv")
test_set <- read.csv("../preparedData/test_set.csv")

train_set$churn <- as.factor(train_set$churn)
validation_set$churn <- as.factor(validation_set$churn)
test_set$churn <- as.factor(test_set$churn)

X_train <- train_set %>% select(-churn)
y_train <- train_set$churn

X_validation <- validation_set %>% select(-churn)
y_validation <- validation_set$churn

X_test <- test_set %>% select(-churn)
y_test <- test_set$churn

cat("  Données chargées avec succès\n")
cat("    Train:", nrow(train_set), "observations,", ncol(X_train), "features\n")
cat("    Validation:", nrow(validation_set), "observations\n")
cat("    Test:", nrow(test_set), "observations\n")
cat("    Distribution Churn (Train):", paste(names(table(y_train)), "=", table(y_train), collapse = ", "), "\n")


################################################################################
# 2. ENTRAÎNEMENT DU MODÈLE DE RÉGRESSION LOGISTIQUE
################################################################################

cat("\nStep 2: Entraînement des modèles\n")

X_train_matrix <- as.matrix(X_train)
X_val_matrix <- as.matrix(X_validation)
X_test_matrix <- as.matrix(X_test)


################################################################################
# 2.1 MODÈLE DE BASE (SANS RÉGULARISATION)
################################################################################

cat("\n  Modèle 1: Régression Logistique Standard (lambda = 0)\n")

logistic_model_base <- glmnet(
  X_train_matrix, 
  y_train, 
  family = "binomial",
  alpha = 0,
  lambda = 0
)

cat("    Modèle entraîné\n")


################################################################################
# 2.2 MODÈLE AVEC RÉGULARISATION RIDGE
################################################################################

cat("\n  Modèle 2: Régression Logistique Ridge (alpha = 0)\n")

cv_ridge <- cv.glmnet(
  X_train_matrix, 
  y_train, 
  family = "binomial",
  alpha = 0,
  type.measure = "auc",
  nfolds = 5
)

best_lambda_ridge <- cv_ridge$lambda.min
cat("    Meilleur lambda (Ridge):", round(best_lambda_ridge, 6), "\n")

logistic_model_ridge <- glmnet(
  X_train_matrix, 
  y_train, 
  family = "binomial",
  alpha = 0,
  lambda = best_lambda_ridge
)

cat("    Modèle Ridge entraîné\n")


################################################################################
# 2.3 MODÈLE AVEC RÉGULARISATION LASSO
################################################################################

cat("\n  Modèle 3: Régression Logistique Lasso (alpha = 1)\n")

cv_lasso <- cv.glmnet(
  X_train_matrix, 
  y_train, 
  family = "binomial",
  alpha = 1,
  type.measure = "auc",
  nfolds = 5
)

best_lambda_lasso <- cv_lasso$lambda.min
cat("    Meilleur lambda (Lasso):", round(best_lambda_lasso, 6), "\n")

logistic_model_lasso <- glmnet(
  X_train_matrix, 
  y_train, 
  family = "binomial",
  alpha = 1,
  lambda = best_lambda_lasso
)

cat("    Modèle Lasso entraîné\n")

n_features_lasso <- sum(coef(logistic_model_lasso) != 0) - 1
cat("    Features sélectionnées par Lasso:", n_features_lasso, "/", ncol(X_train), "\n")


################################################################################
# 3. PRÉDICTIONS SUR LES ENSEMBLES
################################################################################

cat("\nStep 3: Génération des prédictions\n")

SELECTED_MODEL <- logistic_model_base

cat("  Modèle sélectionné: Régression Logistique Standard\n")


################################################################################
# 3.1 PRÉDICTIONS (PROBABILITÉS)
################################################################################

pred_train_prob <- predict(
  SELECTED_MODEL, 
  newx = X_train_matrix,
  type = "response"
)[, 1]

pred_validation_prob <- predict(
  SELECTED_MODEL, 
  newx = X_val_matrix,
  type = "response"
)[, 1]

pred_test_prob <- predict(
  SELECTED_MODEL, 
  newx = X_test_matrix,
  type = "response"
)[, 1]

cat("  Prédictions (probabilités) générées\n")
cat("    Exemple de probabilités (Test):", round(head(pred_test_prob, 3), 3), "\n")


################################################################################
# 3.2 CONVERSION EN CLASSES (SEUIL = 0.5)
################################################################################

THRESHOLD <- 0.5

pred_train_class <- factor(
  ifelse(pred_train_prob > THRESHOLD, "Yes", "No"),
  levels = c("No", "Yes")
)

pred_validation_class <- factor(
  ifelse(pred_validation_prob > THRESHOLD, "Yes", "No"),
  levels = c("No", "Yes")
)

pred_test_class <- factor(
  ifelse(pred_test_prob > THRESHOLD, "Yes", "No"),
  levels = c("No", "Yes")
)

cat("  Prédictions (classes) générées avec seuil =", THRESHOLD, "\n")


################################################################################
# 4. ÉVALUATION DES PERFORMANCES
################################################################################

cat("\nStep 4: Évaluation des performances\n")


################################################################################
# 4.1 MATRICE DE CONFUSION
################################################################################

cat("\n  4.1 Matrices de Confusion\n")

cm_train <- confusionMatrix(pred_train_class, y_train, positive = "Yes")
cat("\n    TRAIN:\n")
print(cm_train$table)

cm_validation <- confusionMatrix(pred_validation_class, y_validation, positive = "Yes")
cat("\n    VALIDATION:\n")
print(cm_validation$table)

cm_test <- confusionMatrix(pred_test_class, y_test, positive = "Yes")
cat("\n    TEST:\n")
print(cm_test$table)

cm_test_df <- as.data.frame(cm_test$table)

p_cm <- ggplot(cm_test_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white", size = 1.5) +
  geom_text(aes(label = Freq), color = "white", size = 8, fontface = "bold") +
  scale_fill_gradient(low = "#3498db", high = "#e74c3c") +
  labs(
    title = "Matrice de Confusion - Test Set",
    subtitle = paste0("Accuracy: ", round(cm_test$overall['Accuracy'], 3)),
    x = "Prédiction",
    y = "Réalité"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    axis.text = element_text(size = 11),
    legend.position = "none"
  )
print(p_cm)
ggsave("../figures/confusion_matrix_test.png", p_cm, width = 7, height = 6)
cat("    Graphique sauvegardé: confusion_matrix_test.png\n")


################################################################################
# 4.2 MÉTRIQUES DE PERFORMANCE
################################################################################

cat("\n  4.2 Métriques de Performance\n")

extract_metrics <- function(cm) {
  data.frame(
    Accuracy = cm$overall['Accuracy'],
    Precision = cm$byClass['Pos Pred Value'],
    Recall = cm$byClass['Sensitivity'],
    Specificity = cm$byClass['Specificity'],
    F1_Score = cm$byClass['F1'],
    Balanced_Accuracy = cm$byClass['Balanced Accuracy']
  )
}

metrics_train <- extract_metrics(cm_train)
metrics_validation <- extract_metrics(cm_validation)
metrics_test <- extract_metrics(cm_test)

metrics_comparison <- rbind(
  cbind(Set = "Train", metrics_train),
  cbind(Set = "Validation", metrics_validation),
  cbind(Set = "Test", metrics_test)
)

rownames(metrics_comparison) <- NULL

cat("\n")
print(metrics_comparison %>% mutate(across(where(is.numeric), ~ round(.x, 4))))

write.csv(
  metrics_comparison, 
  "../outputs/performance_metrics.csv", 
  row.names = FALSE
)
cat("\n    Métriques sauvegardées: performance_metrics.csv\n")


################################################################################
# 4.3 COURBE ROC ET AUC
################################################################################

cat("\n  4.3 Courbes ROC et AUC\n")

roc_train <- roc(y_train, pred_train_prob, levels = c("No", "Yes"), direction = "<")
roc_validation <- roc(y_validation, pred_validation_prob, levels = c("No", "Yes"), direction = "<")
roc_test <- roc(y_test, pred_test_prob, levels = c("No", "Yes"), direction = "<")

auc_train <- auc(roc_train)
auc_validation <- auc(roc_validation)
auc_test <- auc(roc_test)

cat("    AUC Train:", round(auc_train, 4), "\n")
cat("    AUC Validation:", round(auc_validation, 4), "\n")
cat("    AUC Test:", round(auc_test, 4), "\n")

roc_data <- data.frame(
  FPR = c(1 - roc_train$specificities, 
          1 - roc_validation$specificities, 
          1 - roc_test$specificities),
  TPR = c(roc_train$sensitivities, 
          roc_validation$sensitivities, 
          roc_test$sensitivities),
  Set = rep(c("Train", "Validation", "Test"), 
            c(length(roc_train$specificities), 
              length(roc_validation$specificities), 
              length(roc_test$specificities)))
)

p_roc <- ggplot(roc_data, aes(x = FPR, y = TPR, color = Set)) +
  geom_line(size = 1.3) +
  geom_abline(linetype = "dashed", color = "gray50", size = 0.8) +
  labs(
    title = "Courbe ROC - Régression Logistique",
    subtitle = paste0("AUC - Train: ", round(auc_train, 3), 
                      " | Validation: ", round(auc_validation, 3),
                      " | Test: ", round(auc_test, 3)),
    x = "Taux de Faux Positifs (1 - Specificité)",
    y = "Taux de Vrais Positifs (Sensibilité / Recall)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    legend.position = "bottom",
    legend.title = element_blank()
  ) +
  scale_color_manual(values = c("Train" = "#3498db", 
                                "Validation" = "#f39c12", 
                                "Test" = "#e74c3c"))

print(p_roc)
ggsave("../figures/roc_curve.png", p_roc, width = 9, height = 7)
cat("    Graphique sauvegardé: roc_curve.png\n")


################################################################################
# 4.4 COURBE PRECISION-RECALL
################################################################################

cat("\n  4.4 Courbes Precision-Recall\n")

calc_pr <- function(y_true, y_pred_prob) {
  pred_obj <- prediction(y_pred_prob, y_true)
  perf <- performance(pred_obj, "prec", "rec")
  
  data.frame(
    Recall = perf@x.values[[1]],
    Precision = perf@y.values[[1]]
  )
}

pr_train <- calc_pr(y_train, pred_train_prob)
pr_validation <- calc_pr(y_validation, pred_validation_prob)
pr_test <- calc_pr(y_test, pred_test_prob)

pr_data <- rbind(
  cbind(pr_train, Set = "Train"),
  cbind(pr_validation, Set = "Validation"),
  cbind(pr_test, Set = "Test")
)

p_pr <- ggplot(pr_data, aes(x = Recall, y = Precision, color = Set)) +
  geom_line(size = 1.3, na.rm = TRUE) +
  labs(
    title = "Courbe Precision-Recall",
    subtitle = "Trade-off entre Précision et Rappel",
    x = "Recall (Sensibilité)",
    y = "Precision (Valeur Prédictive Positive)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    legend.position = "bottom",
    legend.title = element_blank()
  ) +
  scale_color_manual(values = c("Train" = "#3498db", 
                                "Validation" = "#f39c12", 
                                "Test" = "#e74c3c")) +
  ylim(0, 1) +
  xlim(0, 1)

print(p_pr)
ggsave("../figures/precision_recall_curve.png", p_pr, width = 9, height = 7)
cat("    Graphique sauvegardé: precision_recall_curve.png\n")


################################################################################
# 4.5 FEATURE IMPORTANCE
################################################################################

cat("\n  4.5 Feature Importance\n")

coefficients <- coef(SELECTED_MODEL) %>%
  as.matrix() %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Feature") %>%
  rename(Coefficient = s0) %>%
  filter(Feature != "(Intercept)") %>%
  mutate(
    Abs_Coefficient = abs(Coefficient),
    Direction = ifelse(Coefficient > 0, "Positive", "Negative")
  ) %>%
  arrange(desc(Abs_Coefficient))

top_features <- coefficients %>% head(15)

cat("    Top 5 features les plus importantes:\n")
print(top_features %>% head(5) %>% select(Feature, Coefficient, Direction))

p_importance <- ggplot(top_features, aes(x = reorder(Feature, Abs_Coefficient), 
                                         y = Coefficient, 
                                         fill = Direction)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Top 15 Features - Importance",
    subtitle = "Basé sur les coefficients du modèle",
    x = "Feature",
    y = "Coefficient"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    legend.position = "bottom"
  ) +
  scale_fill_manual(values = c("Positive" = "#27ae60", "Negative" = "#e74c3c"))

print(p_importance)
ggsave("../figures/feature_importance.png", p_importance, width = 10, height = 7)
cat("    Graphique sauvegardé: feature_importance.png\n")

write.csv(
  coefficients, 
  "../outputs/model_coefficients.csv", 
  row.names = FALSE
)
cat("    Coefficients sauvegardés: model_coefficients.csv\n")


################################################################################
# 5. SAUVEGARDE DU MODÈLE ET DES TRANSFORMERS
################################################################################

cat("\nStep 5: Sauvegarde du modèle et des artifacts\n")


################################################################################
# 5.1 SAUVEGARDE DU MODÈLE
################################################################################

cat("\n  5.1 Sauvegarde du modèle principal\n")

saveRDS(SELECTED_MODEL, "C:/Users/pc/Desktop/Master WISD/Statistiques/logistic_model.rds")
cat("    Modèle sauvegardé: logistic_model.rds\n")

saveRDS(logistic_model_ridge, "../models/logistic_model_ridge.rds")
saveRDS(logistic_model_lasso, "../models/logistic_model_lasso.rds")
cat("    Modèles alternatifs sauvegardés (Ridge, Lasso)\n")


################################################################################
# 5.2 SAUVEGARDE DES MÉTADONNÉES
################################################################################

cat("\n  5.2 Sauvegarde des métadonnées du modèle\n")

model_metadata <- list(
  model_name = "Logistic Regression - Telco Churn",
  model_type = "Régression Logistique (glmnet)",
  created_at = Sys.time(),
  r_version = R.version.string,
  
  hyperparameters = list(
    alpha = 0,
    lambda = 0,
    family = "binomial",
    threshold = THRESHOLD
  ),
  
  performance = list(
    train_accuracy = cm_train$overall['Accuracy'],
    val_accuracy = cm_validation$overall['Accuracy'],
    test_accuracy = cm_test$overall['Accuracy'],
    train_auc = as.numeric(auc_train),
    val_auc = as.numeric(auc_validation),
    test_auc = as.numeric(auc_test),
    test_precision = cm_test$byClass['Pos Pred Value'],
    test_recall = cm_test$byClass['Sensitivity'],
    test_f1 = cm_test$byClass['F1']
  ),
  
  data_info = list(
    n_features = ncol(X_train),
    feature_names = colnames(X_train),
    n_train = nrow(X_train),
    n_validation = nrow(X_validation),
    n_test = nrow(X_test),
    target_variable = "churn",
    class_balance_train = prop.table(table(y_train))
  ),
  
  preprocessing_files = list(
    scaler = "scaler_model.rds",
    dummy_encoder = "dummy_model.rds"
  )
)

saveRDS(model_metadata, "../models/model_metadata.rds")
cat("    Métadonnées sauvegardées: model_metadata.rds\n")


################################################################################
# 6. CHARGEMENT DU MODÈLE (SIMULATION PRODUCTION)
################################################################################

cat("\nStep 6: Simulation du chargement en production\n")

loaded_model <- readRDS("../models/logistic_model.rds")
cat("  Modèle chargé depuis le disque\n")

loaded_metadata <- readRDS("../models/model_metadata.rds")
cat("  Métadonnées chargées\n")
cat("    Modèle créé le:", format(loaded_metadata$created_at, "%Y-%m-%d %H:%M:%S"), "\n")
cat("    AUC Test:", round(loaded_metadata$performance$test_auc, 4), "\n")
cat("    Features:", loaded_metadata$data_info$n_features, "\n")


################################################################################
# 7. INFÉRENCE SUR NOUVELLES DONNÉES
################################################################################

cat("\nStep 7: Fonction d'inférence pour nouvelles données\n")


################################################################################
# 7.1 FONCTION D'INFÉRENCE COMPLÈTE
################################################################################

predict_churn <- function(new_data_raw, 
                          model,
                          scaler_model_path = "../models/scaler_model.rds",
                          dummy_model_path = "../models/dummy_model.rds",
                          threshold = 0.5,
                          return_proba = FALSE) {
  
  cat("  Début de l'inférence...\n")
  
  cat("    Preprocessing des données...\n")
  
  scaler_model <- readRDS(scaler_model_path)
  dummy_model <- readRDS(dummy_model_path)
  
  categorical_vars <- c("gender", "seniorcitizen", "partner", "dependents", 
                        "phoneservice", "multiplelines", "internetservice",
                        "onlinesecurity", "onlinebackup", "deviceprotection",
                        "techsupport", "streamingtv", "streamingmovies",
                        "contract", "paperlessbilling", "paymentmethod")
  
  numerical_vars <- c("tenure", "monthlycharges", "totalcharges")
  
  new_data_clean <- new_data_raw %>%
    select(-customerID) %>%
    rename_with(tolower) %>%
    mutate(
      seniorcitizen = as.factor(seniorcitizen),
      totalcharges = as.numeric(totalcharges),
      totalcharges = replace_na(totalcharges, 0)
    ) %>%
    distinct()
  
  new_data_clean <- new_data_clean %>%
    mutate(across(all_of(categorical_vars), as.factor))
  
  new_dummy <- predict(dummy_model, 
                       newdata = new_data_clean %>% select(all_of(categorical_vars))) %>%
    as.data.frame()
  
  new_scaled <- predict(scaler_model, 
                        new_data_clean %>% select(all_of(numerical_vars))) %>%
    as.data.frame()
  
  new_data_preprocessed <- bind_cols(new_scaled, new_dummy)
  
  cat("    Données preprocessées:", nrow(new_data_preprocessed), "lignes\n")
  
  cat("    Génération des prédictions...\n")
  
  new_data_matrix <- as.matrix(new_data_preprocessed)
  
  predictions_proba <- predict(model, 
                               newx = new_data_matrix,
                               type = "response")[, 1]
  
  if (return_proba) {
    cat("  Inférence terminée (probabilités)\n")
    return(predictions_proba)
  } else {
    predictions_class <- factor(
      ifelse(predictions_proba > threshold, "Yes", "No"),
      levels = c("No", "Yes")
    )
    cat("  Inférence terminée (classes)\n")
    return(list(
      class = predictions_class,
      probability = predictions_proba
    ))
  }
}


################################################################################
# 7.2 EXEMPLE D'UTILISATION
################################################################################

cat("\n  7.2 Exemple d'inférence sur le test set\n")

test_raw_example <- read.csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

test_sample <- test_raw_example %>% sample_n(5)

predictions_example <- predict_churn(
  test_sample, 
  loaded_model,
  threshold = 0.5,
  return_proba = FALSE
)

cat("\n  Résultats de l'inférence:\n")
results_df <- data.frame(
  CustomerID = test_sample$customerID,
  Predicted_Class = predictions_example$class,
  Probability_Churn = round(predictions_example$probability, 3)
)
print(results_df)


################################################################################
# 8. INTERPRÉTATION DU MODÈLE
################################################################################

cat("\nStep 8: Interprétation du modèle\n")

cat("\n  8.1 Résumé des performances:\n")
cat("    Accuracy (Test):", round(cm_test$overall['Accuracy'], 4), "\n")
cat("    Precision (Test):", round(cm_test$byClass['Pos Pred Value'], 4), "\n")
cat("    Recall (Test):", round(cm_test$byClass['Sensitivity'], 4), "\n")
cat("    F1-Score (Test):", round(cm_test$byClass['F1'], 4), "\n")
cat("    AUC (Test):", round(auc_test, 4), "\n")

cat("\n  8.2 Diagnostic:\n")
if (auc_test > 0.8) {
  cat("    Excellent modèle (AUC > 0.8)\n")
} else if (auc_test > 0.7) {
  cat("    Modèle acceptable (0.7 < AUC < 0.8)\n")
} else {
  cat("    Modèle faible (AUC < 0.7)\n")
}

auc_diff <- auc_train - auc_test
if (auc_diff > 0.05) {
  cat("    Attention: Possible overfitting (AUC Train - Test =", round(auc_diff, 3), ")\n")
} else {
  cat("    Pas d'overfitting détecté\n")
}

cat("\n  8.3 Top 3 features qui prédisent le churn:\n")
print(coefficients %>% head(3) %>% select(Feature, Coefficient))


################################################################################
# 9. RÉSUMÉ FINAL
################################################################################

cat("\n")
cat("================================================================================\n")
cat("                          MODÉLISATION TERMINÉE                              \n")
cat("================================================================================\n")
cat("\n")
cat("FICHIERS SAUVEGARDÉS:\n")
cat("  - Modèle: logistic_model.rds\n")
cat("  - Métadonnées: model_metadata.rds\n")
cat("  - Métriques: performance_metrics.csv\n")
cat("  - Coefficients: model_coefficients.csv\n")
cat("  - Graphiques: confusion_matrix_test.png, roc_curve.png,\n")
cat("                precision_recall_curve.png, feature_importance.png\n")
cat("\n")


################################################################################
# FIN DU SCRIPT
################################################################################