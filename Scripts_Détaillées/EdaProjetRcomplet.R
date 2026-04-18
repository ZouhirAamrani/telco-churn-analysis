################################################################################
#                                                                              #
#           ANALYSE EXPLORATOIRE DES DONNÉES (EDA) - COMPLÈTE                  #
#                    Telco Customer Churn Dataset                              #
#                                                                              #
#                                                                              #
################################################################################

################################################################################
# TABLE DES MATIÈRES
################################################################################
# 
# 1.  CONFIGURATION DE L'ENVIRONNEMENT
# 2.  CHARGEMENT DES DONNÉES
# 3.  APERÇU INITIAL
# 4.  NETTOYAGE DES DONNÉES
# 5.  ANALYSE UNIVARIÉE - VARIABLE CIBLE
# 6.  ANALYSE UNIVARIÉE - VARIABLES CATÉGORIELLES
# 7.  ANALYSE UNIVARIÉE - VARIABLES NUMÉRIQUES
# 8.  ANALYSE BIVARIÉE - CHURN VS CATÉGORIELLES
# 9.  ANALYSE BIVARIÉE - CHURN VS NUMÉRIQUES
# 10. MATRICE DE CORRÉLATION
# 11. IMPORTANCE DES VARIABLES (MUTUAL INFORMATION)
# 12. IMPORTANCE DES VARIABLES (CORRÉLATION)
# 13. ANALYSE DÉTAILLÉE PAR VARIABLE NUMÉRIQUE
# 14. VISUALISATIONS COMPARATIVES
# 15. ANALYSE MULTIVARIÉE
# 16. INSIGHTS ET CONCLUSIONS
#
################################################################################


################################################################################
# 1. CONFIGURATION DE L'ENVIRONNEMENT
################################################################################

# Installation des packages (décommenter si nécessaire)
# install.packages(c("dplyr", "ggplot2", "readr", "tidyverse", "skimr",
#                    "DataExplorer", "corrplot", "gridExtra", "infotheo"))

# Chargement des bibliothèques
library(dplyr)
library(ggplot2)
library(readr)
library(tidyverse)
library(skimr)
library(DataExplorer)
library(corrplot)
library(gridExtra)
library(infotheo)

# Configuration globale
options(scipen = 999)
set.seed(123)

# Création du dossier pour les graphiques
output_dir <- "../graphsEDA"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  cat("Dossier créé:", output_dir, "\n")
}

# Compteur pour les graphiques
graph_counter <- 1


################################################################################
# 2. CHARGEMENT DES DONNÉES
################################################################################

df <- read.csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')


################################################################################
# 3. APERÇU INITIAL
################################################################################

# Vue d'ensemble
t(head(df, 4))
t(tail(df, 4))

# Dimensions et structure
dim(df)
colnames(df)
str(df)

# Résumés statistiques
summary(df)
skim(df)


################################################################################
# 4. NETTOYAGE DES DONNÉES
################################################################################

# 4.1 Suppression de customerID
df <- df %>% select(-customerID)

# 4.2 Conversion de SeniorCitizen en facteur
df <- df %>% 
  mutate(SeniorCitizen = as.factor(SeniorCitizen))

# 4.3 Standardisation des noms de colonnes
df <- df %>% rename_with(tolower)

# 4.4 Vérification des valeurs manquantes
colSums(is.na(df))

# 4.5 Analyse de TotalCharges
summary(df$totalcharges)
sum(is.na(df$totalcharges))

# Valeurs les plus fréquentes
df %>%
  count(totalcharges) %>%
  slice_max(n, n = 1)

# Lignes avec tenure = 0 et NA
df %>% 
  filter(tenure == 0 & is.na(totalcharges))

# 4.6 Imputation par la médiane
df <- df %>%
  mutate(totalcharges = replace_na(totalcharges, median(totalcharges, na.rm = TRUE)))

sum(is.na(df$totalcharges))

# 4.7 Vérification et suppression des doublons
sum(duplicated(df))
df <- df %>% distinct()
sum(duplicated(df))


################################################################################
# 5. ANALYSE UNIVARIÉE - VARIABLE CIBLE
################################################################################

# Distribution du Churn
table(df$churn)
prop.table(table(df$churn)) * 100

# Visualisation
churn_table <- table(df$churn)
churn_prop <- prop.table(churn_table) * 100

p <- ggplot(df, aes(x = churn, fill = churn)) +
  geom_bar(alpha = 0.8) +
  geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -0.5) +
  scale_fill_manual(values = c("No" = "#2ecc71", "Yes" = "#e74c3c")) +
  labs(
    title = "Distribution du Churn",
    subtitle = paste0("Taux de churn: ", round(churn_prop[2], 1), "%"),
    x = "Churn", 
    y = "Nombre de clients"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 12)
  )

print(p)
ggsave(file.path(output_dir, sprintf("%02d_churn_distribution.png", graph_counter)), 
       p, width = 8, height = 5, dpi = 300)
graph_counter <- graph_counter + 1

# Conversion en valeur numérique pour analyses
df <- df %>%
  mutate(churn_numeric = as.integer(churn == "Yes"))


################################################################################
# 6. ANALYSE UNIVARIÉE - VARIABLES CATÉGORIELLES
################################################################################

# Définition des variables catégorielles
categorical <- c("gender", "seniorcitizen", "partner", "dependents", 
                 "phoneservice", "multiplelines", "internetservice",
                 "onlinesecurity", "onlinebackup", "deviceprotection",
                 "techsupport", "streamingtv", "streamingmovies",
                 "contract", "paperlessbilling", "paymentmethod")

# Nombre de valeurs uniques
df %>%
  summarise(across(all_of(categorical), n_distinct))

# Valeurs uniques
lapply(df[categorical], unique)

# Fréquences pour chaque variable
for (var in categorical) {
  table(df[[var]])
}


################################################################################
# 7. ANALYSE UNIVARIÉE - VARIABLES NUMÉRIQUES
################################################################################

numerical <- c("tenure", "monthlycharges", "totalcharges")

# Statistiques descriptives
df %>%
  select(all_of(numerical)) %>%
  summary()

# Distribution - Histogrammes
for (var in numerical) {
  p <- ggplot(df, aes(x = .data[[var]])) +
    geom_histogram(bins = 30, fill = "#3498db", alpha = 0.7, color = "black") +
    labs(
      title = paste("Distribution de", var),
      x = var,
      y = "Fréquence"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  print(p)
  ggsave(file.path(output_dir, sprintf("%02d_hist_%s.png", graph_counter, var)), 
         p, width = 8, height = 5, dpi = 300)
  graph_counter <- graph_counter + 1
}

# Boxplots
for (var in numerical) {
  p <- ggplot(df, aes(y = .data[[var]])) +
    geom_boxplot(fill = "#3498db", alpha = 0.7) +
    labs(
      title = paste("Boxplot de", var),
      y = var
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  print(p)
  ggsave(file.path(output_dir, sprintf("%02d_boxplot_%s.png", graph_counter, var)), 
         p, width = 8, height = 5, dpi = 300)
  graph_counter <- graph_counter + 1
}


################################################################################
# 8. ANALYSE BIVARIÉE - CHURN VS VARIABLES CATÉGORIELLES
################################################################################

# Calcul du taux de churn global (baseline)
global_mean <- mean(df$churn_numeric, na.rm = TRUE)
global_mean

# Analyse détaillée pour chaque variable catégorielle
for (feature in categorical) {
  
  df_group <- df %>%
    group_by(!!sym(feature)) %>%
    summarise(
      n = n(),
      mean_churn = mean(churn_numeric, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      diff = mean_churn - global_mean,
      risk_ratio = mean_churn / global_mean,
      churn_pct = round(mean_churn * 100, 2)
    ) %>%
    arrange(desc(risk_ratio))
  
  print(df_group)
}

# Visualisation 1: Distribution par catégorie (barplot groupé)
for (feature in categorical) {
  
  p <- ggplot(df, aes(x = !!sym(feature), fill = churn)) +
    geom_bar(position = "dodge", alpha = 0.8) +
    scale_fill_manual(
      values = c("No" = "#2ecc71", "Yes" = "#e74c3c")
    ) +
    labs(
      title = paste("Distribution par", feature),
      x = feature,
      y = "Nombre de clients",
      fill = "Churn"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "top"
    )
  
  print(p)
  ggsave(file.path(output_dir, sprintf("%02d_barplot_grouped_%s.png", graph_counter, feature)), 
         p, width = 10, height = 6, dpi = 300)
  graph_counter <- graph_counter + 1
}

# Visualisation 2: Taux de churn par catégorie (barplot simple)
for (feature in categorical) {
  
  df_group <- df %>%
    group_by(!!sym(feature)) %>%
    summarise(mean = mean(churn_numeric, na.rm = TRUE), .groups = "drop")
  
  p <- ggplot(df_group, aes(x = !!sym(feature), y = mean)) +
    geom_col(fill = "#3498db", alpha = 0.8) +
    geom_hline(yintercept = global_mean, linewidth = 1.2, 
               color = "#e74c3c", linetype = "dashed") +
    annotate("text", x = 1, y = global_mean + 0.03, 
             label = "Taux global", color = "#e74c3c", fontface = "bold") +
    scale_y_continuous(labels = scales::percent) +
    labs(
      title = paste("Taux de Churn par", feature),
      x = feature,
      y = "Taux de Churn"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  print(p)
  ggsave(file.path(output_dir, sprintf("%02d_churn_rate_%s.png", graph_counter, feature)), 
         p, width = 10, height = 6, dpi = 300)
  graph_counter <- graph_counter + 1
}

# Visualisation 3: Barplot empilé (proportions)
for (feature in categorical) {
  
  p <- ggplot(df, aes(x = !!sym(feature), fill = churn)) +
    geom_bar(position = "fill", alpha = 0.8) +
    scale_fill_manual(
      values = c("No" = "#2ecc71", "Yes" = "#e74c3c")
    ) +
    scale_y_continuous(labels = scales::percent) +
    labs(
      title = paste("Proportion de Churn par", feature),
      x = feature,
      y = "Proportion",
      fill = "Churn"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "top"
    )
  
  print(p)
  ggsave(file.path(output_dir, sprintf("%02d_proportion_%s.png", graph_counter, feature)), 
         p, width = 10, height = 6, dpi = 300)
  graph_counter <- graph_counter + 1
}


################################################################################
# 9. ANALYSE BIVARIÉE - CHURN VS VARIABLES NUMÉRIQUES
################################################################################

# Statistiques par groupe de churn
df %>%
  group_by(churn) %>%
  summarise(
    across(all_of(numerical),
           list(mean = mean, median = median, sd = sd),
           .names = "{.col}_{.fn}")
  )

# Boxplots par churn
for (var in numerical) {
  
  p <- ggplot(df, aes(x = churn, y = .data[[var]], fill = churn)) +
    geom_boxplot(alpha = 0.7) +
    scale_fill_manual(
      values = c("No" = "#2ecc71", "Yes" = "#e74c3c")
    ) +
    labs(
      title = paste(var, "par Churn"),
      x = "Churn",
      y = var
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      legend.position = "none"
    )
  
  print(p)
  ggsave(file.path(output_dir, sprintf("%02d_boxplot_churn_%s.png", graph_counter, var)), 
         p, width = 8, height = 5, dpi = 300)
  graph_counter <- graph_counter + 1
}

# Violin plots
for (var in numerical) {
  
  p <- ggplot(df, aes(x = churn, y = .data[[var]], fill = churn)) +
    geom_violin(alpha = 0.7) +
    geom_boxplot(width = 0.1, fill = "white", alpha = 0.5) +
    scale_fill_manual(
      values = c("No" = "#2ecc71", "Yes" = "#e74c3c")
    ) +
    labs(
      title = paste("Distribution de", var, "par Churn"),
      x = "Churn",
      y = var
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      legend.position = "none"
    )
  
  print(p)
  ggsave(file.path(output_dir, sprintf("%02d_violin_%s.png", graph_counter, var)), 
         p, width = 8, height = 5, dpi = 300)
  graph_counter <- graph_counter + 1
}

# Density plots
for (var in numerical) {
  
  p <- ggplot(df, aes(x = .data[[var]], fill = churn)) +
    geom_density(alpha = 0.5) +
    scale_fill_manual(
      values = c("No" = "#2ecc71", "Yes" = "#e74c3c")
    ) +
    labs(
      title = paste("Densité de", var, "par Churn"),
      x = var,
      y = "Densité",
      fill = "Churn"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      legend.position = "top"
    )
  
  print(p)
  ggsave(file.path(output_dir, sprintf("%02d_density_%s.png", graph_counter, var)), 
         p, width = 8, height = 5, dpi = 300)
  graph_counter <- graph_counter + 1
}


################################################################################
# 10. MATRICE DE CORRÉLATION
################################################################################

# 10.1 Corrélation entre variables numériques
cor_numeric <- cor(df %>% select(all_of(numerical)), use = "complete.obs")
cor_numeric

# Visualisation avec corrplot
png(file.path(output_dir, sprintf("%02d_corrplot_numeric.png", graph_counter)), 
    width = 800, height = 600, res = 100)
corrplot(cor_numeric, 
         method = "color", 
         type = "upper",
         addCoef.col = "black",
         tl.col = "black",
         tl.srt = 45,
         title = "Matrice de Corrélation - Variables Numériques",
         mar = c(0, 0, 2, 0),
         number.cex = 1.2)
dev.off()
graph_counter <- graph_counter + 1

# 10.2 Corrélation incluant churn
cor_with_churn <- cor(df %>% select(all_of(numerical), churn_numeric), 
                      use = "complete.obs")
cor_with_churn

png(file.path(output_dir, sprintf("%02d_corrplot_with_churn.png", graph_counter)), 
    width = 800, height = 600, res = 100)
corrplot(cor_with_churn, 
         method = "color",
         addCoef.col = "black",
         tl.col = "black",
         tl.srt = 45,
         title = "Matrice de Corrélation (avec Churn)",
         mar = c(0, 0, 2, 0),
         number.cex = 1)
dev.off()
graph_counter <- graph_counter + 1

# 10.3 Heatmap avec ggplot2
cor_df <- as.data.frame(as.table(cor_with_churn))
names(cor_df) <- c("Var1", "Var2", "Correlation")

p <- ggplot(cor_df, aes(x = Var1, y = Var2, fill = Correlation)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(Correlation, 2)), color = "black", size = 4) +
  scale_fill_gradient2(
    low = "#e74c3c", 
    mid = "white", 
    high = "#2ecc71",
    midpoint = 0,
    limit = c(-1, 1)
  ) +
  labs(
    title = "Heatmap de Corrélation",
    x = "",
    y = ""
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

print(p)
ggsave(file.path(output_dir, sprintf("%02d_heatmap_correlation.png", graph_counter)), 
       p, width = 8, height = 6, dpi = 300)
graph_counter <- graph_counter + 1


################################################################################
# 11. IMPORTANCE DES VARIABLES (MUTUAL INFORMATION)
################################################################################

# Calcul de l'information mutuelle pour variables catégorielles
mi_results <- sapply(categorical, function(feature) {
  mutinformation(df[[feature]], df$churn)
})

# Dataframe des résultats
df_mi <- data.frame(
  Feature = names(mi_results),
  MI = mi_results,
  row.names = NULL
) %>%
  arrange(desc(MI))

# Top 10
head(df_mi, 10)

# Bottom 5
tail(df_mi, 5)

# Visualisation
p <- ggplot(df_mi, aes(x = reorder(Feature, MI), y = MI)) +
  geom_bar(stat = "identity", fill = "#9b59b6", alpha = 0.8) +
  coord_flip() +
  labs(
    title = "Importance des Variables (Mutual Information)",
    x = "Variables",
    y = "Mutual Information"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14)
  )

print(p)
ggsave(file.path(output_dir, sprintf("%02d_mutual_information.png", graph_counter)), 
       p, width = 10, height = 8, dpi = 300)
graph_counter <- graph_counter + 1


################################################################################
# 12. IMPORTANCE DES VARIABLES (CORRÉLATION)
################################################################################

# Corrélation avec churn pour variables numériques
correlations <- sapply(numerical, function(var) {
  cor(df[[var]], df$churn_numeric, use = "complete.obs")
})

correlations

# Dataframe
df_cor <- data.frame(
  Feature = names(correlations),
  Correlation = correlations,
  AbsCorrelation = abs(correlations)
) %>%
  arrange(desc(AbsCorrelation))

df_cor

# Visualisation
p <- ggplot(df_cor, aes(x = reorder(Feature, AbsCorrelation), y = Correlation)) +
  geom_bar(stat = "identity", 
           aes(fill = Correlation > 0), 
           alpha = 0.8) +
  coord_flip() +
  scale_fill_manual(
    values = c("TRUE" = "#e74c3c", "FALSE" = "#2ecc71"),
    labels = c("Négative", "Positive")
  ) +
  labs(
    title = "Corrélation avec Churn (Variables Numériques)",
    x = "Variables",
    y = "Coefficient de Corrélation",
    fill = "Type"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    legend.position = "bottom"
  )

print(p)
ggsave(file.path(output_dir, sprintf("%02d_correlation_with_churn.png", graph_counter)), 
       p, width = 8, height = 6, dpi = 300)
graph_counter <- graph_counter + 1


################################################################################
# 13. ANALYSE DÉTAILLÉE PAR VARIABLE NUMÉRIQUE
################################################################################

# 13.1 TENURE
# Catégorisation
t1 <- mean(df$churn_numeric[df$tenure >= 1 & df$tenure <= 2], na.rm = TRUE)
t2 <- mean(df$churn_numeric[df$tenure >= 3 & df$tenure <= 12], na.rm = TRUE)
t3 <- mean(df$churn_numeric[df$tenure > 12], na.rm = TRUE)

c(t1, t2, t3)

df_tenure <- data.frame(
  tenure_cat = factor(c('1-2', '3-12', '+12'), levels = c('1-2', '3-12', '+12')),
  churn_rate = c(t1, t2, t3)
)

p <- ggplot(df_tenure, aes(x = tenure_cat, y = churn_rate, fill = tenure_cat)) +
  geom_col(alpha = 0.8) +
  scale_fill_brewer(palette = "Greens") +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Taux de Churn par Catégorie de Tenure",
    x = "Tenure (mois)",
    y = "Taux de Churn"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

print(p)
ggsave(file.path(output_dir, sprintf("%02d_tenure_categories.png", graph_counter)), 
       p, width = 8, height = 5, dpi = 300)
graph_counter <- graph_counter + 1

# 13.2 MONTHLY CHARGES
mc1 <- mean(df$churn_numeric[df$monthlycharges <= 20], na.rm = TRUE)
mc2 <- mean(df$churn_numeric[df$monthlycharges > 20 & df$monthlycharges <= 50], na.rm = TRUE)
mc3 <- mean(df$churn_numeric[df$monthlycharges > 50], na.rm = TRUE)

c(mc1, mc2, mc3)

df_monthly <- data.frame(
  charges_cat = factor(c('0-20', '21-50', '+50'), levels = c('0-20', '21-50', '+50')),
  churn_rate = c(mc1, mc2, mc3)
)

p <- ggplot(df_monthly, aes(x = charges_cat, y = churn_rate, fill = charges_cat)) +
  geom_col(alpha = 0.8) +
  scale_fill_brewer(palette = "Blues") +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Taux de Churn par Charges Mensuelles",
    x = "Charges Mensuelles ($)",
    y = "Taux de Churn"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

print(p)
ggsave(file.path(output_dir, sprintf("%02d_monthly_charges_categories.png", graph_counter)), 
       p, width = 8, height = 5, dpi = 300)
graph_counter <- graph_counter + 1

# 13.3 TOTAL CHARGES
tc1 <- mean(df$churn_numeric[df$totalcharges <= 1000], na.rm = TRUE)
tc2 <- mean(df$churn_numeric[df$totalcharges > 1000 & df$totalcharges <= 5000], na.rm = TRUE)
tc3 <- mean(df$churn_numeric[df$totalcharges > 5000], na.rm = TRUE)

c(tc1, tc2, tc3)

df_total <- data.frame(
  charges_cat = factor(c('0-1000', '1000-5000', '+5000'), 
                       levels = c('0-1000', '1000-5000', '+5000')),
  churn_rate = c(tc1, tc2, tc3)
)

p <- ggplot(df_total, aes(x = charges_cat, y = churn_rate, fill = charges_cat)) +
  geom_col(alpha = 0.8) +
  scale_fill_brewer(palette = "Oranges") +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Taux de Churn par Charges Totales",
    x = "Charges Totales ($)",
    y = "Taux de Churn"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

print(p)
ggsave(file.path(output_dir, sprintf("%02d_total_charges_categories.png", graph_counter)), 
       p, width = 8, height = 5, dpi = 300)
graph_counter <- graph_counter + 1


################################################################################
# 14. VISUALISATIONS COMPARATIVES
################################################################################

# 14.1 Comparaison des taux de churn (top variables)
top_vars <- c("contract", "internetservice", "techsupport", "onlinesecurity")

plots_list <- list()

for (i in seq_along(top_vars)) {
  var <- top_vars[i]
  
  df_group <- df %>%
    group_by(!!sym(var)) %>%
    summarise(mean = mean(churn_numeric, na.rm = TRUE), .groups = "drop")
  
  p <- ggplot(df_group, aes(x = !!sym(var), y = mean, fill = !!sym(var))) +
    geom_col(alpha = 0.8) +
    geom_hline(yintercept = global_mean, linetype = "dashed", color = "red") +
    scale_y_continuous(labels = scales::percent) +
    labs(
      title = var,
      x = "",
      y = "Taux"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "none"
    )
  
  plots_list[[i]] <- p
}

p_combined <- do.call(grid.arrange, c(plots_list, ncol = 2))

# Sauvegarder la grille de graphiques
ggsave(file.path(output_dir, sprintf("%02d_top_variables_comparison.png", graph_counter)), 
       p_combined, width = 12, height = 10, dpi = 300)
graph_counter <- graph_counter + 1


################################################################################
# 15. ANALYSE MULTIVARIÉE
################################################################################

# 15.1 Churn par Contract et InternetService
df %>%
  group_by(contract, internetservice) %>%
  summarise(
    n = n(),
    churn_rate = mean(churn_numeric, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(churn_rate))

p <- ggplot(df, aes(x = contract, fill = churn)) +
  geom_bar(position = "fill") +
  facet_wrap(~ internetservice) +
  scale_fill_manual(
    values = c("No" = "#2ecc71", "Yes" = "#e74c3c")
  ) +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Taux de Churn par Contrat et Service Internet",
    x = "Type de Contrat",
    y = "Proportion",
    fill = "Churn"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

print(p)
ggsave(file.path(output_dir, sprintf("%02d_contract_internetservice.png", graph_counter)), 
       p, width = 12, height = 6, dpi = 300)
graph_counter <- graph_counter + 1

# 15.2 Tenure groups
df <- df %>%
  mutate(tenure_group = cut(tenure, 
                            breaks = c(0, 12, 24, 48, 72, Inf),
                            labels = c("0-12 mois", "12-24 mois", "24-48 mois", 
                                       "48-72 mois", "72+ mois")))

df %>%
  group_by(tenure_group) %>%
  summarise(
    n = n(),
    churn_rate = mean(churn_numeric, na.rm = TRUE),
    .groups = "drop"
  )

p <- ggplot(df, aes(x = tenure_group, fill = churn)) +
  geom_bar(position = "fill", alpha = 0.8) +
  scale_fill_manual(
    values = c("No" = "#2ecc71", "Yes" = "#e74c3c")
  ) +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Taux de Churn par Groupe de Tenure",
    x = "Groupe de Tenure",
    y = "Proportion",
    fill = "Churn"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

print(p)
ggsave(file.path(output_dir, sprintf("%02d_tenure_groups.png", graph_counter)), 
       p, width = 10, height = 6, dpi = 300)
graph_counter <- graph_counter + 1

# 15.3 Scatter plots (variables numériques)
p <- ggplot(df, aes(x = tenure, y = monthlycharges, color = churn)) +
  geom_point(alpha = 0.5) +
  scale_color_manual(values = c("No" = "#2ecc71", "Yes" = "#e74c3c")) +
  labs(
    title = "Tenure vs Monthly Charges par Churn",
    x = "Tenure (mois)",
    y = "Charges Mensuelles ($)",
    color = "Churn"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

print(p)
ggsave(file.path(output_dir, sprintf("%02d_scatter_tenure_monthly.png", graph_counter)), 
       p, width = 10, height = 6, dpi = 300)
graph_counter <- graph_counter + 1

p <- ggplot(df, aes(x = tenure, y = totalcharges, color = churn)) +
  geom_point(alpha = 0.5) +
  scale_color_manual(values = c("No" = "#2ecc71", "Yes" = "#e74c3c")) +
  labs(
    title = "Tenure vs Total Charges par Churn",
    x = "Tenure (mois)",
    y = "Charges Totales ($)",
    color = "Churn"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

print(p)
ggsave(file.path(output_dir, sprintf("%02d_scatter_tenure_total.png", graph_counter)), 
       p, width = 10, height = 6, dpi = 300)
graph_counter <- graph_counter + 1


################################################################################
# 16. INSIGHTS ET CONCLUSIONS
################################################################################

cat("\n\n")
cat("================================================================================\n")
cat("                    ANALYSE TERMINÉE AVEC SUCCÈS                               \n")
cat("================================================================================\n")
cat("\n")
cat("Nombre total de graphiques générés:", graph_counter - 1, "\n")
cat("Dossier de sauvegarde:", output_dir, "\n")
cat("\n")
cat("INSIGHTS CLÉS :\n")
cat("\n")
cat("1. GENDER - Pas de différence significative entre hommes et femmes\n")
cat("2. SENIOR CITIZEN - Les seniors ont un taux de churn significativement plus élevé\n")
cat("3. PARTNER & DEPENDENTS - Les personnes avec un partenaire churnent moins\n")
cat("4. PHONE SERVICE - Impact minimal sur le churn\n")
cat("5. INTERNET SERVICE - Fiber optic: taux de churn élevé\n")
cat("6. SERVICES ADDITIONNELS - Les clients sans ces services churnent plus\n")
cat("7. CONTRACT - Variable la plus importante! Month-to-month: 42%, Two year: 3%\n")
cat("8. PAYMENT METHOD - Electronic check: taux élevé\n")
cat("9. TENURE - Corrélation négative forte (-0.35)\n")
cat("10. MONTHLY CHARGES - Corrélation positive (0.19)\n")
cat("11. TOTAL CHARGES - Corrélation négative\n")
cat("\n")
cat("================================================================================\n")


################################################################################
# FIN DE L'ANALYSE
################################################################################

