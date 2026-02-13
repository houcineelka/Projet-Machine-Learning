# Projet Machine Learning — Prédiction des Maladies Cardiaques

## Description

Ce projet applique des techniques de Machine Learning sur le dataset **Cleveland Heart Disease** afin de prédire la présence d'une maladie cardiaque. Il couvre l'ensemble du pipeline ML : exploration des données, ingénierie des features, clustering, classification, régression, tuning des hyperparamètres et suivi des expériences avec MLflow.

---

## Dataset

**Fichier :** `heart.csv` / `heart_clean.csv`

| Feature | Description |
|---|---|
| `age` | Âge du patient |
| `sex` | Sexe (1 = homme, 0 = femme) |
| `cp` | Type de douleur thoracique (0–3) |
| `trestbps` | Pression artérielle au repos (mm Hg) |
| `chol` | Cholestérol sérique (mg/dl) |
| `fbs` | Glycémie à jeun > 120 mg/dl (1 = vrai) |
| `restecg` | Résultats ECG au repos (0–2) |
| `thalach` | Fréquence cardiaque maximale atteinte |
| `exang` | Angine induite par l'exercice (1 = oui) |
| `oldpeak` | Dépression ST induite par l'exercice |
| `slope` | Pente du segment ST à l'effort |
| `ca` | Nombre de vaisseaux colorés (0–3) |
| `thal` | Thalassémie (1 = normal, 2 = défaut fixe, 3 = défaut réversible) |
| `target` | **Variable cible** (1 = maladie, 0 = sain) |

---

## Structure du Projet

```
projet/
├── 01_EDA_Analysis.ipynb               # Analyse exploratoire des données
├── 02_Feature_Engineering.ipynb        # Ingénierie et sélection de features
├── 03_Clustering_Analysis.ipynb        # Analyse par clustering
├── 04_Classification_Models.ipynb      # Modèles de classification
├── 05_Regression_Models.ipynb          # Modèles de régression
├── 06_Hyperparameter_Tuning.ipynb      # Optimisation des hyperparamètres
├── 07_Final_Comparison_MLflow.ipynb    # Comparaison finale & MLflow
├── heart.csv                           # Dataset brut
├── heart_clean.csv                     # Dataset nettoyé
├── mlflow_utils.py                     # Utilitaires MLflow
├── models/                             # Modèles entraînés (.pkl)
├── models_optimized/                   # Modèles après tuning (.pkl)
├── images/                             # Visualisations générées
├── data_transformed/                   # Données transformées
└── mlruns/                             # Logs MLflow
```

---

## Étapes du Pipeline

### 1. Analyse Exploratoire (EDA)
- Statistiques descriptives
- Distribution de la variable cible
- Analyse des corrélations
- Détection des valeurs aberrantes et des valeurs manquantes
- Visualisations : heatmap, pairplot, boxplots, violin plots

### 2. Ingénierie des Features
Méthodes de sélection de features appliquées :

| Méthode | Features sélectionnées |
|---|---|
| **RFE** | `sex`, `cp`, `thalach`, `exang`, `oldpeak`, `ca`, `thal` |
| **SBS** | `sex`, `cp`, `restecg`, `oldpeak`, `slope`, `ca`, `thal` |
| **LASSO** | 12 features |
| **PCA** | 12 composantes principales |
| **LDA** | 1 discriminant linéaire |
| **t-SNE** | 2 composantes (visualisation) |

**Features consensus** (sélectionnées par toutes les méthodes) : `sex`, `cp`, `oldpeak`, `ca`, `thal`

### 3. Clustering
- **K-Means**
- **DBSCAN**
- **Clustering Hiérarchique**
- **Clustering Spectral**

### 4. Classification
Modèles entraînés :
- Logistic Regression, Decision Tree, Random Forest, Extra Trees
- KNN, Naive Bayes, Ridge Classifier
- SVM (Linear, RBF, Polynomial)
- Gradient Boosting, AdaBoost, LightGBM, XGBoost
- Stacking Ensemble

### 5. Régression
Modèles entraînés :
- Ridge, Lasso, Random Forest, Gradient Boosting, SVR (RBF), XGBoost

### 6. Optimisation des Hyperparamètres
Modèles optimisés :
- `Random_Forest_optimized.pkl`
- `SVM_optimized.pkl`
- `XGBoost_optimized.pkl`

### 7. Suivi avec MLflow
Toutes les expériences sont tracées via **MLflow** (métriques, paramètres, artefacts) dans le dossier `mlruns/`.

---

## Prérequis

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm mlflow joblib
```

---

## Utilisation

Exécuter les notebooks dans l'ordre :

```bash
jupyter notebook 01_EDA_Analysis.ipynb
jupyter notebook 02_Feature_Engineering.ipynb
jupyter notebook 03_Clustering_Analysis.ipynb
jupyter notebook 04_Classification_Models.ipynb
jupyter notebook 05_Regression_Models.ipynb
jupyter notebook 06_Hyperparameter_Tuning.ipynb
jupyter notebook 07_Final_Comparison_MLflow.ipynb
```

Pour visualiser les expériences MLflow :

```bash
mlflow ui
```

---

## Auteur

Projet réalisé dans le cadre du Master ISI — Module Machine Learning.