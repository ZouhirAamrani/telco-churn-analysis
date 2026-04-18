# Analyse Complète du Churn — Telco Customer Dataset

> Prédiction du désabonnement client par classification supervisée et régression logistique  
> **Faculté des Sciences Dhar El Mahraz — Université Sidi Mohamed Ben Abdallah**  
> Année universitaire 2024-2025

---

## Auteurs

| Nom | Rôle |
|-----|------|
| Aamrani Zouhir | Analyse & Modélisation |
| Ighachouten Anass | Analyse & Modélisation |

**Encadrant :** Abdelkamel ALJ

---

## Description du projet

Ce projet vise à prédire le **churn client** (désabonnement) dans le secteur des télécommunications à l'aide d'un modèle de **régression logistique**. Il s'appuie sur le dataset public *Telco Customer Churn* d'IBM Watson Analytics (7 043 observations, 21 variables).

L'objectif est double :
- **Statistique** : construire un modèle prédictif fiable (AUC-ROC > 0.75, Recall > 0.70)
- **Business** : identifier les facteurs clés du churn et formuler des recommandations actionnables

---
## Base de données

La base de données provient de :
- [Telco Customer Churn sur Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)

---
## Structure du projet

```
projet/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset source
├── models/
│   ├── logistic_model.rds                       # Modèle entraîné
│   ├── scaler_model.rds                         # Scaler (Z-score)
│   └── dummy_model.rds                          # Encodeur One-Hot
├── outputs/
│   ├── performance_metrics.csv                  # Métriques de performance
│   └── model_coefficients.csv                   # Coefficients & feature importance
├── figures/
│   ├── churn_distribution.png
│   ├── roc_curve.png
│   └── feature_importance.png
├── telco_churn_analysis.Rmd                     # Code source principal
└── README.md
```

---

## Environnement technique

- **Langage :** R 4.3.1
- **Packages principaux :**

| Package | Usage |
|---------|-------|
| `tidyverse` | Manipulation des données |
| `caret` | Preprocessing & validation |
| `glmnet` | Régression logistique |
| `pROC` / `ROCR` | Courbes ROC & AUC |
| `ggplot2` | Visualisations |
| `corrplot` | Heatmap de corrélation |

---

## Lancer le projet

```r
# 1. Installer les dépendances
packages <- c("dplyr", "ggplot2", "readr", "tidyverse", "skimr",
              "DataExplorer", "corrplot", "gridExtra", "infotheo",
              "caret", "glmnet", "pROC", "ROCR", "broom")
install.packages(packages)

# 2. Ouvrir et exécuter le fichier principal
# Dans RStudio : ouvrir telco_churn_analysis.Rmd puis "Knit" ou "Run All"
```

---

## Résultats du modèle (Test Set)

| Métrique | Résultat | Objectif | Statut |
|----------|----------|----------|--------|
| AUC-ROC | **0.8389** | > 0.75 | Atteint |
| Accuracy | **79.20%** | — | Bonne |
| F1-Score | **59.24%** | > 0.60 | Proche |
| Recall | **56.80%** | > 0.70 | À améliorer |
| Écart Train/Test | **1.25%** | < 5% | Pas d'overfitting |

---

## Insights clés

- **Type de contrat** : les clients *month-to-month* churned à 42.7% vs 2.8% pour les contrats 2 ans (×15)
- **Service Fiber optic** : taux de churn de 41.9% — problème de qualité ou de prix perçu
- **Ancienneté (tenure)** : les 12 premiers mois sont la période critique ; chaque mois supplémentaire réduit le risque de 59%
- **Services additionnels** : l'absence de TechSupport ou OnlineSecurity double le risque de churn
- **Méthode de paiement** : *Electronic check* associé à un churn de 45.3% vs ~15% pour les autres méthodes

---

## Recommandations stratégiques

1. **Proposer des contrats longue durée** avec réductions (-15% à -25%) et avantages exclusifs
2. **Auditer et améliorer** la qualité du service Fiber optic
3. **Renforcer l'onboarding** des nouveaux clients (0–12 mois) : welcome pack, suivi proactif
4. **Bundler les services additionnels** (sécurité, backup, support) pour augmenter la valeur perçue
5. **Encourager la migration** vers des méthodes de paiement automatiques (-$5/mois d'incentive)

---

## Limitations

- Dataset fictif (IBM) — validation sur données réelles recommandée
- Régression logistique : relations supposées linéaires ; des modèles non-linéaires (Random Forest, XGBoost) pourraient améliorer le Recall
- Absence de données de satisfaction client, historique de réclamations et variables temporelles

---

## Références principales

- Hastie, Tibshirani & Friedman (2009) — *The Elements of Statistical Learning*
- James et al. (2013) — *An Introduction to Statistical Learning with Applications in R*
- Verbeke et al. (2012) — *New insights into churn prediction in the telecommunication sector*
- Fawcett (2006) — *An introduction to ROC analysis*
