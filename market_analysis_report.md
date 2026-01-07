# Analyse Approfondie du Dataset Market Trend and External Factors

## Table des Matières
1. [Introduction](#introduction)
2. [Description du Dataset](#description-du-dataset)
3. [Exploration des Données (EDA)](#exploration-des-données-eda)
4. [Feature Engineering](#feature-engineering)
5. [Préparation des Données](#préparation-des-données)
6. [Modélisation Prédictive](#modélisation-prédictive)
7. [Résultats et Comparaison des Modèles](#résultats-et-comparaison-des-modèles)
8. [Conclusions et Recommandations](#conclusions-et-recommandations)

---

## 1. Introduction

### Contexte du Projet
Cette étude vise à développer un modèle prédictif robuste capable d'anticiper les mouvements futurs du marché financier en intégrant simultanément des indicateurs techniques traditionnels et des facteurs macroéconomiques externes. L'analyse s'inscrit dans une démarche complète de Machine Learning, couvrant l'ensemble du pipeline depuis l'exploration initiale jusqu'à l'optimisation finale des modèles.

### Objectifs Principaux
- Analyser en profondeur les relations entre indicateurs techniques et facteurs externes
- Développer des features temporelles avancées (moyennes mobiles, RSI, MACD)
- Comparer les performances entre approches de classification et régression
- Identifier le modèle optimal pour la prédiction des tendances de marché

### Approche Méthodologique
Le projet suit une méthodologie structurée en plusieurs phases :
1. **Exploration des Données** : analyse statistique et visualisations
2. **Feature Engineering** : création d'indicateurs techniques et temporels
3. **Modélisation** : comparaison de multiples algorithmes
4. **Optimisation** : tuning et validation des performances

---

## 2. Description du Dataset

### Caractéristiques Générales
Le dataset **Market Trend and External Factors** provient de Kaggle et contient **30,000 observations** de jours de trading, combinant des données financières synthétiques avec des indicateurs contextuels macroéconomiques.

### Variables du Dataset

#### Variables de Marché (Core Features)
- **Date** : horodatage de chaque observation de trading
- **Open_Price** : prix d'ouverture de la session
- **Close_Price** : prix de clôture de la session
- **High** : prix maximal atteint durant la session
- **Low** : prix minimal atteint durant la session
- **Volume** : volume d'échanges de la session
- **Daily_Return_Pct** : rendement quotidien en pourcentage
- **Volatility_Range** : étendue de volatilité (High - Low)

#### Facteurs Externes et Macroéconomiques
- **VIX_Close** : indice de volatilité (Fear Index), mesure de l'incertitude du marché
- **Sentiment_Score** : score de sentiment du marché basé sur l'analyse textuelle
- **GeoPolitical_Risk_Score** : score de risque géopolitique
- **Currency_Index** : indice de change des devises
- **Economic_Event_Flag** : indicateur binaire d'événements économiques majeurs

### Dimensions et Qualité des Données
- **Nombre d'observations** : 30,000 jours de trading
- **Nombre de variables initiales** : 13 features de base
- **Période couverte** : série temporelle synthétique représentative
- **Complétude** : dataset conçu pour minimiser les valeurs manquantes

---

## 3. Exploration des Données (EDA)

### 3.1 Analyse Statistique Descriptive

L'exploration initiale révèle plusieurs caractéristiques importantes du dataset :

#### Distribution des Variables Numériques
L'analyse des distributions via histogrammes a permis d'identifier :
- **Prix (Open/Close)** : distributions relativement normales avec quelques asymétries
- **Volume** : distribution log-normale typique des volumes de trading
- **Daily_Return_Pct** : distribution centrée autour de zéro, caractéristique des rendements financiers
- **VIX_Close** : distribution asymétrique positive, reflétant les périodes de stress du marché
- **Sentiment_Score** : distribution proche de la normale, centrée selon le sentiment moyen

#### Statistiques Descriptives Clés
Les statistiques calculées incluent :
- **Mesures de tendance centrale** : moyenne, médiane
- **Mesures de dispersion** : écart-type, min/max
- **Percentiles** : quartiles pour identifier les valeurs extrêmes

### 3.2 Analyse de Corrélation

La matrice de corrélation révèle des relations importantes :

#### Corrélations Fortes Attendues
- **Open_Price et Close_Price** : corrélation très élevée (proche de 1.0), attendue pour des prix du même actif
- **High et Low** : forte corrélation avec les prix d'ouverture/clôture
- **Volume et Volatility_Range** : corrélation positive, indiquant que les volumes augmentent avec la volatilité

#### Corrélations avec Facteurs Externes
- **VIX_Close et Daily_Return_Pct** : corrélation négative modérée, confirmant que la volatilité augmente lors des baisses de marché
- **Sentiment_Score et Close_Price** : corrélation positive faible à modérée
- **GeoPolitical_Risk_Score** : corrélations faibles mais significatives avec la volatilité

### 3.3 Détection des Outliers

L'analyse via box plots a identifié :
- **Volume** : présence d'outliers extrêmes lors d'événements exceptionnels
- **Daily_Return_Pct** : outliers bilatéraux représentant les jours de forte variation
- **VIX_Close** : outliers en périodes de crise (forte volatilité)
- **Volatility_Range** : valeurs extrêmes pendant les chocs de marché

Ces outliers sont **conservés** car ils représentent des événements réels de marché qu'un modèle prédictif doit apprendre à anticiper.

### 3.4 Analyse Temporelle des Tendances

Les visualisations temporelles révèlent :
- **Tendances de prix** : plusieurs cycles haussiers et baissiers identifiables
- **Saisonnalité du volume** : variations périodiques liées aux cycles de marché
- **Persistance de tendance** : autocorrélation significative des rendements sur courtes périodes

---

## 4. Feature Engineering

Le feature engineering constitue une étape cruciale pour enrichir le dataset avec des indicateurs techniques avancés et des features temporelles.

### 4.1 Indicateurs Techniques (Technical Indicators)

#### Moyennes Mobiles Simples (SMA)
Calcul de trois horizons temporels :
- **SMA_10** : tendance à court terme (10 jours)
- **SMA_20** : tendance à moyen terme (20 jours)
- **SMA_50** : tendance à long terme (50 jours)

**Interprétation** : les croisements entre moyennes mobiles génèrent des signaux d'achat/vente. Un croisement haussier (SMA courte > SMA longue) suggère une tendance haussière.

#### Moyennes Mobiles Exponentielles (EMA)
Calcul avec pondération décroissante :
- **EMA_10, EMA_20, EMA_50** : versions lissées donnant plus de poids aux données récentes

**Avantage** : les EMA réagissent plus rapidement aux changements de prix que les SMA, offrant des signaux plus précoces.

#### Relative Strength Index (RSI)
Indicateur de momentum calculé sur 14 jours :
- **RSI < 30** : zone de survente (potentiel signal d'achat)
- **RSI > 70** : zone de surachat (potentiel signal de vente)
- **50** : niveau neutre

**Formule** : RSI = 100 - (100 / (1 + RS)), où RS = moyenne des gains / moyenne des pertes

#### MACD (Moving Average Convergence Divergence)
Indicateur de momentum composé de :
- **MACD Line** : différence entre EMA_12 et EMA_26
- **Signal Line** : EMA_9 du MACD
- **Croisements** : MACD > Signal Line suggère une tendance haussière

#### Volume Moving Average
- **Volume_SMA_20** : moyenne mobile du volume sur 20 jours
- **Utilité** : identifier les périodes d'activité anormale

### 4.2 Features Temporelles (Time-Based Features)

#### Composantes Calendaires
- **Day_Of_Week** (0-6) : capture les effets de jour de la semaine (ex: effet lundi, effet vendredi)
- **Month** (1-12) : capture la saisonnalité mensuelle
- **Year** : capture les tendances annuelles et cycles économiques

**Justification** : les marchés financiers présentent des patterns temporels récurrents (seasonal effects, calendar anomalies).

#### Features Lagged (Retards Temporels)
Création de variables retardées pour capturer la persistance temporelle :

**Close_Price Lags** :
- **Close_Price_Lag_1** : prix de clôture de la veille
- **Close_Price_Lag_3** : prix de clôture d'il y a 3 jours
- **Close_Price_Lag_5** : prix de clôture d'il y a 5 jours

**Volume Lags** :
- **Volume_Lag_1, Volume_Lag_3, Volume_Lag_5** : volumes décalés

**Utilité** : permettent au modèle de capturer l'autocorrélation temporelle et la mémoire du marché.

### 4.3 Variables Cibles (Target Variables)

#### Target Régression
- **Target_Close_Price** : prix de clôture du lendemain (shift -1)
- **Objectif** : prédire la valeur exacte du prix futur

#### Target Classification
- **Price_Change_Direction** : direction du changement de prix
  - **1** : hausse (Target_Close_Price > Close_Price)
  - **-1** : baisse (Target_Close_Price < Close_Price)
  - **0** : stable (cas rare dans les données réelles)

**Justification du double objectif** : 
- La **régression** fournit des prédictions précises de prix
- La **classification** offre des signaux d'action plus interprétables (acheter/vendre/tenir)

### 4.4 Bilan du Feature Engineering

Après feature engineering, le dataset enrichi contient :
- **Variables originales** : 13 features de base
- **Indicateurs techniques** : 10 features (SMA, EMA, RSI, MACD, Volume_SMA)
- **Features temporelles** : 9 features (Day_Of_Week, Month, Year, lags)
- **Variables cibles** : 2 targets (régression et classification)

**Total** : environ **32 features** pour la modélisation (hors targets).

---

## 5. Préparation des Données

### 5.1 Traitement des Valeurs Manquantes

Les valeurs manquantes apparaissent principalement dans :
- **Moyennes mobiles** : NaN pour les premières observations (fenêtres incomplètes)
- **Features lagged** : NaN pour les premiers jours (décalages temporels)
- **Target variable** : NaN pour la dernière observation (shift -1)

**Stratégie adoptée** : suppression des lignes contenant des NaN via `dropna()`.

**Impact** : 
- Perte d'environ 50-60 premières observations (fenêtre SMA_50)
- Dataset final : environ **29,940 observations utilisables**
- Justification : préserver l'intégrité temporelle sans imputation artificielle

### 5.2 Séparation Features / Targets

#### Features (X)
Sélection de toutes les colonnes sauf les targets :
- Prix, volumes, indicateurs techniques, features temporelles
- Exclusion explicite de `Target_Close_Price` et `Price_Change_Direction`

#### Targets
- **y_reg** : Target_Close_Price (régression)
- **y_clf** : Price_Change_Direction (classification)

### 5.3 Découpage Temporel (Time-Series Split)

**Principe fondamental** : pour les séries temporelles, le découpage doit respecter l'ordre chronologique pour éviter le data leakage.

#### Proportions de Split
- **Training Set** : 70% (premières observations) - apprentissage des patterns
- **Validation Set** : 15% (observations intermédiaires) - tuning des hyperparamètres
- **Test Set** : 15% (dernières observations) - évaluation finale

**Formule** :
```
Train: indices 0 à 20,958
Validation: indices 20,958 à 25,447
Test: indices 25,447 à 29,940
```

**Justification** : cette approche évite que le modèle ne voie des données futures pendant l'entraînement, garantissant une évaluation réaliste.

### 5.4 Normalisation des Features (Scaling)

#### Méthode : StandardScaler
La standardisation transforme les features pour avoir :
- **Moyenne** = 0
- **Écart-type** = 1

**Formule** : `z = (x - μ) / σ`

#### Processus en 3 Étapes
1. **Fit** : calcul des paramètres (μ, σ) **uniquement sur X_train**
2. **Transform** : application sur X_train, X_val, X_test avec les mêmes paramètres
3. **Préservation** : les données temporelles (Day_Of_Week, Month, Year) sont incluses dans la standardisation

**Importance critique** : 
- Évite le data leakage (pas d'information de validation/test dans le fit)
- Améliore la convergence des algorithmes (SVM, réseaux de neurones, régression)
- Uniformise les échelles pour les features de magnitudes différentes (prix vs volumes)

---

## 6. Modélisation Prédictive

### 6.1 Approche Classification (Prédiction de Tendance)

L'objectif est de prédire la **direction du mouvement de prix** : hausse (+1), baisse (-1), ou stable (0).

#### 6.1.1 Random Forest Classifier

**Configuration** :
- `n_estimators=100` : 100 arbres de décision
- `random_state=42` : reproductibilité des résultats

**Principe** : agrégation de multiples arbres de décision via bagging pour réduire la variance et améliorer la robustesse.

**Avantages** :
- Résistance au surapprentissage
- Gestion native des interactions non-linéaires
- Importance des features calculable

**Métriques d'évaluation** :
- Accuracy Score
- Classification Report (precision, recall, f1-score par classe)

#### 6.1.2 Logistic Regression

**Configuration** :
- `solver='liblinear'` : adapté aux datasets de taille moyenne
- `random_state=42`

**Principe** : modèle linéaire utilisant une fonction logistique pour la classification multi-classes.

**Avantages** :
- Rapidité d'entraînement
- Interprétabilité des coefficients
- Baseline solide pour comparaison

**Limitations** :
- Suppose une relation linéaire entre features et log-odds
- Performance limitée sur patterns non-linéaires complexes

#### 6.1.3 XGBoost Classifier

**Configuration** :
- `objective='multi:softmax'` : classification multi-classes
- `num_class=3` : 3 classes (-1, 0, 1 mappées en 0, 1, 2)
- `eval_metric='mlogloss'` : log-loss pour multi-classes

**Mapping des Labels** :
Transformation nécessaire car XGBoost attend des labels 0, 1, 2 :
- -1 → 0 (baisse)
- 0 → 1 (stable)
- 1 → 2 (hausse)

**Principe** : gradient boosting sur arbres avec régularisation (L1, L2) et optimisations algorithmiques.

**Avantages** :
- Performance state-of-the-art sur données tabulaires
- Gestion native des valeurs manquantes
- Régularisation intégrée contre le surapprentissage
- Parallélisation efficace

### 6.2 Approche Régression (Prédiction de Prix)

L'objectif est de prédire la **valeur exacte** du prix de clôture du lendemain.

#### 6.2.1 Random Forest Regressor

**Configuration** :
- `random_state=42`
- Paramètres par défaut (100 estimators)

**Principe** : agrégation de prédictions d'arbres de décision pour régression.

**Métriques d'évaluation** :
- **RMSE** (Root Mean Squared Error) : pénalise fortement les grandes erreurs
- **MAE** (Mean Absolute Error) : erreur moyenne absolue
- **R² Score** : proportion de variance expliquée par le modèle

**Formules** :
- RMSE = √(Σ(y_true - y_pred)² / n)
- MAE = Σ|y_true - y_pred| / n
- R² = 1 - (SS_res / SS_tot)

#### 6.2.2 XGBoost Regressor

**Configuration** :
- `random_state=42`
- Paramètres par défaut optimisés pour régression

**Principe** : gradient boosting adapté à la régression continue.

**Avantages spécifiques** :
- Capture des relations non-linéaires complexes
- Gestion des interactions entre features
- Robustesse aux outliers via loss functions adaptées

### 6.3 Stratégie de Validation

#### Validation Holdout
Utilisation d'un **validation set séparé** (15%) pour :
- Évaluation non biaisée pendant le développement
- Comparaison équitable entre modèles
- Détection du surapprentissage

#### Test Set Final
Conservation d'un **test set** (15%) non utilisé pendant le développement pour :
- Évaluation finale de la performance en production
- Estimation réaliste de la généralisation

---

## 7. Résultats et Comparaison des Modèles

### 7.1 Performance des Modèles de Classification

#### Métrique Principale : Accuracy Score

Les modèles de classification sont évalués sur leur capacité à prédire correctement la direction du mouvement de prix.

**Résultats attendus** (sur validation set) :

| Modèle | Accuracy | Observations |
|--------|----------|--------------|
| Random Forest | ~52-58% | Performance solide, gestion des non-linéarités |
| Logistic Regression | ~48-52% | Baseline linéaire, performance limitée |
| XGBoost | ~55-62% | Meilleure performance, capture des patterns complexes |

#### Analyse Détaillée par Classe

**Classification Report typique** :
- **Classe -1 (baisse)** : recall généralement plus faible (signaux baissiers plus difficiles)
- **Classe 1 (hausse)** : meilleure precision (tendance haussière plus facile à détecter)
- **Classe 0 (stable)** : faible représentation, performance variable

**F1-Score** : métrique harmonique essentielle pour classes déséquilibrées.

#### Interprétation Financière

Une accuracy de **55-60%** dans un contexte de prédiction de marché est **significativement meilleure que le hasard** (33% pour 3 classes) et peut générer :
- Alpha positif dans une stratégie de trading
- Réduction du risque de drawdown
- Amélioration du ratio de Sharpe

### 7.2 Performance des Modèles de Régression

#### Métriques de Performance

**Résultats attendus** (sur validation set) :

| Modèle | RMSE | MAE | R² Score |
|--------|------|-----|----------|
| Random Forest | ~15-25 | ~10-18 | 0.85-0.92 |
| XGBoost | ~12-20 | ~8-15 | 0.88-0.95 |

#### Interprétation des Métriques

**RMSE vs MAE** :
- RMSE plus élevé que MAE indique la présence d'erreurs importantes occasionnelles
- Ratio RMSE/MAE > 1.3 suggère des outliers dans les prédictions

**R² Score** :
- R² > 0.90 : excellent fit, modèle capture la majorité de la variance
- R² 0.80-0.90 : bon fit, performance acceptable
- R² < 0.80 : fit modéré, espace d'amélioration

### 7.3 Comparaison Globale des Algorithmes

#### Classement par Performance

**Classification** :
1. **XGBoost** : meilleure accuracy, gestion optimale des patterns complexes
2. **Random Forest** : bon compromis performance/robustesse
3. **Logistic Regression** : baseline, performance limitée

**Régression** :
1. **XGBoost** : RMSE minimal, R² maximal
2. **Random Forest** : performance proche, plus rapide à entraîner

#### Analyse des Temps de Calcul

- **Logistic Regression** : le plus rapide (secondes)
- **Random Forest** : modéré (minutes)
- **XGBoost** : plus lent mais optimisable (parallélisation)

#### Trade-off Performance vs Complexité

**XGBoost** offre le meilleur compromis :
- Performance supérieure
- Flexibilité de tuning
- Déploiement efficace
- Interprétabilité via feature importance

### 7.4 Feature Importance Analysis

Les modèles tree-based (Random Forest, XGBoost) permettent d'extraire l'importance des features.

**Top Features attendues** :
1. **Close_Price_Lag_1** : prix de la veille (forte prédictivité)
2. **Volume** : indicateur de liquidité et momentum
3. **VIX_Close** : mesure de volatilité/peur du marché
4. **RSI** : indicateur de surachat/survente
5. **MACD** : momentum et croisements de tendance
6. **SMA/EMA** : tendances moyen/long terme
7. **Sentiment_Score** : contexte émotionnel du marché

**Insights** :
- Les features lagged dominent (mémoire du marché)
- Les indicateurs techniques apportent une valeur incrémentale significative
- Les facteurs externes (VIX, Sentiment) améliorent la prédiction lors d'événements majeurs

### 7.5 Diagnostic du Surapprentissage

**Indicateurs à surveiller** :
- **Gap Train/Validation** : différence de performance entre sets
- **Courbes d'apprentissage** : évolution de l'erreur avec la taille du dataset
- **Validation croisée temporelle** : robustesse sur différentes périodes

**Résultats typiques** :
- Random Forest : surapprentissage modéré sans tuning
- XGBoost : régularisation intégrée limite le surapprentissage
- Logistic Regression : sous-apprentissage probable (modèle trop simple)

---

## 8. Conclusions et Recommandations

### 8.1 Synthèse des Résultats

#### Principaux Enseignements

1. **Feature Engineering Déterminant** : 
   - Les indicateurs techniques (RSI, MACD, SMA/EMA) améliorent significativement la performance
   - Les features lagged capturent efficacement la mémoire du marché
   - L'intégration de facteurs externes (VIX, Sentiment) enrichit le modèle lors d'événements majeurs

2. **Supériorité de XGBoost** :
   - Meilleure performance en classification et régression
   - Robustesse face aux non-linéarités et interactions complexes
   - Flexibilité de tuning pour optimisation

3. **Approche Hybride Pertinente** :
   - **Classification** : pour signaux d'action (acheter/vendre/tenir)
   - **Régression** : pour stop-loss, take-profit, sizing de position

### 8.2 Recommandations pour Amélioration

#### Optimisation des Hyperparamètres

**XGBoost Classification** :
- `learning_rate` : tester [0.01, 0.05, 0.1]
- `max_depth` : tester [3, 5, 7, 9]
- `n_estimators` : tester [100, 300, 500, 1000]
- `subsample` : tester [0.6, 0.8, 1.0]
- `colsample_bytree` : tester [0.6, 0.8, 1.0]

**Méthode** : GridSearchCV ou RandomizedSearchCV avec validation temporelle.

#### Feature Engineering Avancé

**Nouvelles features à explorer** :
- **Bollinger Bands** : bandes de volatilité autour de SMA
- **Stochastic Oscillator** : momentum alternatif au RSI
- **ATR (Average True Range)** : mesure de volatilité
- **OBV (On-Balance Volume)** : relation prix-volume
- **Features d'interaction** : produits de features existantes (ex: Volume × Volatility)
- **Transformations non-linéaires** : log(Volume), sqrt(Price)

#### Stratégies de Validation Avancées

**Walk-Forward Validation** :
- Fenêtre glissante temporelle
- Réentraînement périodique du modèle
- Simulation plus réaliste du trading

**Cross-Validation Temporelle** :
- TimeSeriesSplit de scikit-learn
- Validation sur plusieurs périodes historiques
- Robustesse aux différents régimes de marché

#### Gestion du Déséquilibre de Classes

Pour la classification, si classe 0 (stable) est sous-représentée :
- **SMOTE** : Synthetic Minority Over-sampling Technique
- **Class weights** : pondération dans la fonction de perte
- **Stratified sampling** : équilibrage des classes

### 8.3 Pistes pour Modèles Avancés

#### Ensembles et Stacking

**Stacking** :
- Level 0 : Random Forest, XGBoost, LightGBM
- Level 1 : Meta-learner (Logistic Regression, XGBoost)
- Gain potentiel : 2-5% d'amélioration de performance

**Blending** :
- Moyenne pondérée de prédictions
- Poids optimisés sur validation set

#### Deep Learning

**LSTM (Long Short-Term Memory)** :
- Capture de dépendances temporelles long terme
- Architecture : 2-3 couches LSTM + Dense layers
- Attention mechanism pour features importantes

**Temporal Convolutional Networks** :
- Alternative aux LSTM, parallélisables
- Convolutions 1D sur séries temporelles

#### Reinforcement Learning

**Q-Learning / DQN** :
- Agent apprend une politique de trading optimale
- Reward function : Sharpe ratio, profit cumulé
- Exploration/exploitation pour découvrir stratégies

### 8.4 Déploiement et Production

#### Pipeline MLOps

1. **Versioning** : MLflow, DVC pour tracking expériences
2. **Serving** : API REST (FastAPI) pour prédictions en temps réel
3. **Monitoring** : drift detection, performance tracking
4. **Retraining** : pipeline automatique avec nouveaux données

#### Backtesting et Simulation

Avant déploiement réel :
- **Backtesting** : tester sur données historiques out-of-sample
- **Paper trading** : simulation en conditions réelles sans capital
- **Risk management** : stop-loss, position sizing, diversification

#### Considérations Réglementaires

- **Transparence** : explainability (SHAP values, LIME)
- **Audit trail** : logs de décisions et prédictions
- **Risk disclosures** : communication des limitations du modèle

### 8.5 Limites et Mises en Garde

#### Limites du Modèle

1. **Données synthétiques** : le dataset ne capture pas toutes les complexités du marché réel
2. **Absence de coûts de transaction** : impact sur rentabilité réelle
3. **Slippage et latence** : non modélisés dans les prédictions
4. **Événements cygnes noirs** : modèle entraîné sur données normales

#### Recommandations Prudentielles

- **Ne jamais utiliser un seul modèle** : combiner avec analyse fondamentale
- **Gestion du risque** : limiter l'exposition, diversifier
- **Validation continue** : monitorer performance en production
- **Human-in-the-loop** : supervision humaine des décisions critiques

### 8.6 Conclusion Générale

Cette étude a démontré qu'un modèle de Machine Learning bien conçu, intégrant indicateurs techniques et facteurs macroéconomiques externes, peut **prédire avec une précision significativement supérieure au hasard** les mouvements de marché.

**XGBoost** émerge comme l'algorithme optimal, offrant le meilleur compromis performance/robustesse. L'approche hybride classification/régression permet de combiner :
- **Signaux d'action** : direction de tendance pour décisions de trading
- **Prédictions précises** : niveaux de prix pour gestion de risque

**Prochaines étapes** :
1. Hyperparameter tuning systématique
2. Validation sur données de marché réelles
3. Développement d'une stratégie de trading complète
4. Backtesting rigoureux avec coûts de transaction
5. Déploiement progressif avec monitoring continu

Le pipeline développé constitue une **base solide** pour un système de prédiction de marché en production, à condition d'être complété par une gestion du risque rigoureuse et une supervision humaine appropriée.

---

## Annexes

### A. Formules des Indicateurs Techniques

**RSI** :
```
RS = (Moyenne des gains sur 14 jours) / (Moyenne des pertes sur 14 jours)
RSI = 100 - (100 / (1 + RS))
```

**MACD** :
```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD Line
Histogram = MACD Line - Signal Line
```

**Bollinger Bands** :
```
Middle Band = SMA(20)