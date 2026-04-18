################################################################################
#                                                                              #
#                    PREPROCESSING DES DONNÉES                
#              Telco Customer Churn - Préparation pour Modélisation            #
#                                                                              #
################################################################################

################################################################################
# TABLE DES MATIÈRES
################################################################################
# 
# 1. CONFIGURATION ET CHARGEMENT
# 2. NETTOYAGE BASIQUE (AVANT SPLIT) 
# 3. DATA SPLITTING (TRAIN / VALIDATION / TEST) 
# 4. PREPROCESSING SUR TRAIN (FIT) PUIS APPLICATION (TRANSFORM) 
#    4.1 One-Hot Encoding
#    4.2 Scaling des Variables Numériques
# 5. VÉRIFICATION DES ENSEMBLES
# 6. SAUVEGARDE DES DONNÉES PREPROCESSÉES
#
################################################################################


################################################################################
# 1. CONFIGURATION ET CHARGEMENT
################################################################################

library(dplyr)
library(caret)        # Pour createDataPartition et preprocessing
library(recipes)      # Pour pipeline de preprocessing moderne
library(rsample)      # Pour split avancé
library(ggplot2)
library(tidyverse)

set.seed(123)


################################################################################
# 2. NETTOYAGE BASIQUE (AVANT SPLIT) 
################################################################################


df <- read.csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2.1 Nettoyage basique (pas de statistiques calculées)
df <- df %>% 
  select(-customerID) %>%                          # Supprimer ID
  rename_with(tolower) %>%                         # Noms en minuscules
  mutate(
    seniorcitizen = as.factor(seniorcitizen),     # Conversion basique
    totalcharges = as.numeric(totalcharges)       # Conversion basique
  ) %>%
  distinct()                                       # Supprimer doublons

# 2.2 Gérer les NA de manière simple (on remplit avec 0 )

# Remplacer par 0 (logique métier: pas de charges si pas de tenure)
df <- df %>%
  mutate(totalcharges = replace_na(totalcharges, 0))

# 2.3 Identifier les variables
categorical_vars <- c("gender", "seniorcitizen", "partner", "dependents", 
                      "phoneservice", "multiplelines", "internetservice",
                      "onlinesecurity", "onlinebackup", "deviceprotection",
                      "techsupport", "streamingtv", "streamingmovies",
                      "contract", "paperlessbilling", "paymentmethod")

numerical_vars <- c("tenure", "monthlycharges", "totalcharges")

# 2.4 Convertir les variables catégorielles en facteurs 
df <- df %>%
  mutate(across(all_of(categorical_vars), as.factor))

# 2.5 Convertir churn en facteur (variable cible)
df <- df %>%
  mutate(churn = as.factor(churn))

# Vérification
cat("Dimensions après nettoyage:", dim(df), "\n")
cat("Variables numériques:", length(numerical_vars), "\n")
cat("Variables catégorielles:", length(categorical_vars), "\n")
cat("NA dans totalcharges:", sum(is.na(df$totalcharges)), "\n")


################################################################################
# 3. DATA SPLITTING (TRAIN / VALIDATION / TEST) 
################################################################################


# 3.1 Split 70% Train / 15% Validation / 15% Test
set.seed(123)
train_index <- createDataPartition(df$churn, p = 0.70, list = FALSE)

train_raw <- df[train_index, ]
temp_set <- df[-train_index, ]

# 3.2 Split Validation / Test (50/50 du temp_set)
set.seed(123)
val_index <- createDataPartition(temp_set$churn, p = 0.50, list = FALSE)

validation_raw <- temp_set[val_index, ]
test_raw <- temp_set[-val_index, ]

# Vérification des proportions

cat("  Train:", nrow(train_raw), "(", round(nrow(train_raw)/nrow(df)*100, 1), "%)\n")
cat("  Validation:", nrow(validation_raw), "(", round(nrow(validation_raw)/nrow(df)*100, 1), "%)\n")
cat("  Test:", nrow(test_raw), "(", round(nrow(test_raw)/nrow(df)*100, 1), "%)\n")

# Vérification de la distribution de churn

cat("  Train:", round(prop.table(table(train_raw$churn))*100, 1), "%\n")
cat("  Validation:", round(prop.table(table(validation_raw$churn))*100, 1), "%\n")
cat("  Test:", round(prop.table(table(test_raw$churn))*100, 1), "%\n")


################################################################################
# 4. PREPROCESSING SUR TRAIN (FIT) PUIS APPLICATION (TRANSFORM) 
################################################################################
# MAINTENANT ON FAIT LE PREPROCESSING EN 3 ÉTAPES:
# 1. FIT sur TRAIN uniquement
# 2. TRANSFORM sur TRAIN
# 3. TRANSFORM sur VALIDATION et TEST (avec les paramètres du TRAIN)


################################################################################
# 4.1 ONE-HOT ENCODING
################################################################################

# 4.1.1 FIT: Créer le modèle d'encoding sur TRAIN uniquement
dummy_model <- dummyVars(~ ., 
                         data = train_raw %>% select(all_of(categorical_vars)), 
                         fullRank = TRUE)

# 4.1.2 TRANSFORM: Appliquer sur TRAIN
train_dummy <- predict(dummy_model, 
                       newdata = train_raw %>% select(all_of(categorical_vars))) %>%
  as.data.frame()

# 4.1.3 TRANSFORM: Appliquer sur VALIDATION (avec le modèle du TRAIN)
validation_dummy <- predict(dummy_model, 
                            newdata = validation_raw %>% select(all_of(categorical_vars))) %>%
  as.data.frame()

# 4.1.4 TRANSFORM: Appliquer sur TEST (avec le modèle du TRAIN)
test_dummy <- predict(dummy_model, 
                      newdata = test_raw %>% select(all_of(categorical_vars))) %>%
  as.data.frame()

cat("  encoding terminé -", ncol(train_dummy), "features créées\n")


################################################################################
# 4.2 SCALING DES VARIABLES NUMÉRIQUES
################################################################################


# 4.2.1 FIT: Calculer les paramètres (mean, sd) sur TRAIN uniquement
preproc_model <- preProcess(
  train_raw %>% select(all_of(numerical_vars)),
  method = c("center", "scale")  # center = moyenne 0, scale = écart-type 1
)

# Afficher les paramètres appris

cat("    Moyennes:", round(preproc_model$mean, 2), "\n")
cat("    Écarts-types:", round(preproc_model$std, 2), "\n")

# 4.2.2 TRANSFORM: Appliquer sur TRAIN
train_scaled <- predict(preproc_model, 
                        train_raw %>% select(all_of(numerical_vars))) %>%
  as.data.frame()

# 4.2.3 TRANSFORM: Appliquer sur VALIDATION (avec les stats du TRAIN)
validation_scaled <- predict(preproc_model, 
                             validation_raw %>% select(all_of(numerical_vars))) %>%
  as.data.frame()

# 4.2.4 TRANSFORM: Appliquer sur TEST (avec les stats du TRAIN)
test_scaled <- predict(preproc_model, 
                       test_raw %>% select(all_of(numerical_vars))) %>%
  as.data.frame()




################################################################################
# 4.3 COMBINER TOUTES LES FEATURES
################################################################################


# Train
train_set <- bind_cols(
  train_scaled,
  train_dummy,
  train_raw %>% select(churn)
)

# Validation
validation_set <- bind_cols(
  validation_scaled,
  validation_dummy,
  validation_raw %>% select(churn)
)

# Test
test_set <- bind_cols(
  test_scaled,
  test_dummy,
  test_raw %>% select(churn)
)


cat("    Train:", dim(train_set), "\n")
cat("    Validation:", dim(validation_set), "\n")
cat("    Test:", dim(test_set), "\n")


################################################################################
# 5. VÉRIFICATION DES ENSEMBLES
################################################################################


# 5.1 Vérifier que le scaling est correct sur TRAIN
cat("\nVérification du scaling sur TRAIN (doit être ~0 et ~1):\n")
train_set %>% 
  select(all_of(numerical_vars)) %>%
  summarise(across(everything(), list(mean = ~round(mean(.), 6), sd = ~round(sd(.), 3)))) %>%
  print()

# 5.2 Vérifier le scaling sur VALIDATION (ne sera PAS exactement 0 et 1)
cat("\nScaling sur VALIDATION (différent de TRAIN, c'est NORMAL):\n")
validation_set %>% 
  select(all_of(numerical_vars)) %>%
  summarise(across(everything(), list(mean = ~round(mean(.), 3), sd = ~round(sd(.), 3)))) %>%
  print()

# 5.3 Distribution de la variable cible

cat("  Train:\n")
print(table(train_set$churn))
cat("  Validation:\n")
print(table(validation_set$churn))
cat("  Test:\n")
print(table(test_set$churn))

# 5.4 Vérifier qu'il n'y a pas de NA

cat("  Train:", sum(is.na(train_set)), "\n")
cat("  Validation:", sum(is.na(validation_set)), "\n")
cat("  Test:", sum(is.na(test_set)), "\n")


################################################################################
# 6. SAUVEGARDE DES DONNÉES PREPROCESSÉES
################################################################################


# 6.1 Sauvegarder les datasets
write.csv(train_set, "../preparedData/train_set.csv", row.names = FALSE)
write.csv(validation_set, "../preparedData/validation_set.csv", row.names = FALSE)
write.csv(test_set, "../preparedData/test_set.csv", row.names = FALSE)



################################################################################
# 7. SÉPARATION X ET Y POUR MODÉLISATION
################################################################################

# Train
X_train <- train_set %>% select(-churn)
y_train <- train_set$churn

# Validation
X_validation <- validation_set %>% select(-churn)
y_validation <- validation_set$churn

# Test
X_test <- test_set %>% select(-churn)
y_test <- test_set$churn


cat("  X_train:", dim(X_train), "| y_train:", length(y_train), "\n")
cat("  X_validation:", dim(X_validation), "| y_validation:", length(y_validation), "\n")
cat("  X_test:", dim(X_test), "| y_test:", length(y_test), "\n")


################################################################################
# VÉRIFICATION VISUELLE DE LA DISTRIBUTION
################################################################################

library(ggplot2)

# Créer un dataframe pour la comparaison
comparison_df <- data.frame(
  Set = rep(c("Train", "Validation", "Test"), each = 2),
  Churn = rep(c("No", "Yes"), 3),
  Count = c(
    sum(train_set$churn == "No"), sum(train_set$churn == "Yes"),
    sum(validation_set$churn == "No"), sum(validation_set$churn == "Yes"),
    sum(test_set$churn == "No"), sum(test_set$churn == "Yes")
  )
)

comparison_df <- comparison_df %>%
  group_by(Set) %>%
  mutate(Proportion = Count / sum(Count))

# Graphique
p <- ggplot(comparison_df, aes(x = Set, y = Proportion, fill = Churn)) +
  geom_bar(stat = "identity", position = "fill") +
  scale_fill_manual(values = c("No" = "#2ecc71", "Yes" = "#e74c3c")) +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Distribution de Churn dans les Ensembles",
    subtitle = "Vérification de la stratification",
    x = "Ensemble",
    y = "Proportion"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 10)
  )

ggsave("../figures/churn_distribution.png", p, width = 8, height = 5)



################################################################################
# SAUVEGARDE DES MODÈLES DE PREPROCESSING
################################################################################

# Sauvegarder le modèle de scaling
saveRDS(preproc_model, "../models/scaler_model.rds")

# Sauvegarder le modèle d'encoding
saveRDS(dummy_model, "../models/dummy_model.rds")

cat("\n✓ Modèles de preprocessing sauvegardés:\n")
cat("  - scaler_model.rds\n")
cat("  - dummy_model.rds\n")


################################################################################
# FIN DU PREPROCESSING 
################################################################################

