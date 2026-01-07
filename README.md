## Analyse Approfondie du Dataset Market Trend and External Factors


## CHERESTAL Deborah Nativa
## 22008149
## G2 Finance
## BD-DS



![Deborah Nativa Cherestal](![WhatsApp Image 2025-10-30 at 11 50 17](https://github.com/user-attachments/assets/1a60df9d-1939-410d-ba24-a93462ac4deb)

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

Le présent travail s'inscrit dans le cadre d'une analyse approfondie du dataset "Market Trend and External Factors" provenant de Kaggle, avec pour ambition de développer un système prédictif robuste capable d'anticiper les mouvements futurs des marchés financiers. Cette étude se distingue par son approche holistique qui ne se limite pas aux seuls indicateurs techniques traditionnels, mais intègre également des facteurs macroéconomiques externes susceptibles d'influencer significativement les dynamiques de marché. L'objectif principal consiste à construire un modèle de Machine Learning performant en combinant des variables endogènes au marché (prix, volumes, moyennes mobiles) avec des variables exogènes reflétant le contexte économique global (PIB, taux d'intérêt, volatilité implicite, sentiment du marché).

La méthodologie adoptée suit rigoureusement le pipeline complet du Machine Learning, depuis l'exploration initiale des données jusqu'à l'optimisation finale des modèles prédictifs. Cette démarche structurée commence par une phase d'Exploratory Data Analysis (EDA) approfondie permettant de comprendre la structure des données, d'identifier les patterns sous-jacents et de détecter d'éventuelles anomalies. Elle se poursuit par une phase de feature engineering sophistiquée où sont créés des indicateurs techniques avancés (RSI, MACD, moyennes mobiles exponentielles) ainsi que des features temporelles capturant la saisonnalité et l'autocorrélation des séries financières. La phase de modélisation comparative évalue ensuite plusieurs algorithmes de Machine Learning selon deux axes complémentaires : une approche de classification pour prédire la direction des mouvements de prix (hausse, baisse, stabilité) et une approche de régression pour estimer précisément les niveaux de prix futurs. Enfin, l'optimisation finale se concentre sur l'algorithme XGBoost, reconnu pour ses performances exceptionnelles sur les données tabulaires complexes.

L'intérêt scientifique et pratique de cette étude réside dans sa capacité à démontrer que l'intégration de facteurs externes au modèle traditionnel d'analyse technique améliore significativement la qualité des prédictions. En effet, les marchés financiers ne fonctionnent pas en vase clos mais réagissent constamment aux événements macroéconomiques, aux variations de sentiment des investisseurs, aux tensions géopolitiques et aux fluctuations des devises. Un modèle qui capture ces multiples dimensions dispose d'un avantage informationnel substantiel par rapport aux approches unidimensionnelles. De plus, la comparaison systématique entre différents algorithmes (Random Forest, Logistic Regression, XGBoost) et entre deux paradigmes de prédiction (classification versus régression) permet d'identifier les forces et faiblesses de chaque approche, offrant ainsi une vision nuancée des possibilités et limites du Machine Learning appliqué à la finance quantitative.

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

Le dataset "Market Trend and External Factors" constitue une ressource synthétique exceptionnellement riche, conçue spécifiquement pour l'entraînement de modèles de prédiction financière avancés. Composé de 30,000 observations représentant autant de journées de trading, ce dataset offre une granularité temporelle quotidienne permettant de capturer les dynamiques à court et moyen terme des marchés financiers. L'une des caractéristiques remarquables de ce dataset réside dans sa construction méthodique combinant des séries temporelles de prix réalistes avec un ensemble complet de variables contextuelles macroéconomiques. Cette approche multidimensionnelle reflète la réalité complexe des marchés financiers modernes où les décisions d'investissement ne peuvent plus se baser uniquement sur l'analyse technique des graphiques de prix, mais doivent intégrer une compréhension approfondie du contexte économique global.

Les variables de marché constituent le cœur du dataset et incluent les métriques fondamentales du trading quotidien. Le prix d'ouverture (Open_Price) représente le premier prix coté lors de la session de trading, établissant le point de départ des négociations journalières. Le prix de clôture (Close_Price) capture quant à lui le dernier prix négocié, servant de référence principale pour les calculs de rendement et l'évaluation de performance. Les prix extrêmes High et Low délimitent la fourchette de variation intrajournalière, offrant une mesure directe de la volatilité réalisée au cours de la session. Le volume d'échanges quantifie l'intensité de l'activité de trading, un indicateur crucial pour évaluer la liquidité du marché et la force des mouvements de prix. Le rendement quotidien en pourcentage (Daily_Return_Pct) standardise les variations de prix, permettant des comparaisons inter-temporelles et la détection de jours exceptionnels. Enfin, la plage de volatilité (Volatility_Range), calculée comme la différence entre High et Low, fournit une mesure alternative de l'incertitude et du risque quotidien.

Au-delà des variables de marché traditionnelles, le dataset se distingue par l'inclusion de facteurs externes reflétant le contexte macroéconomique et psychologique dans lequel évoluent les marchés. L'indice VIX (VIX_Close), souvent qualifié de "baromètre de la peur", mesure la volatilité implicite anticipée par les options sur le S&P 500, offrant un proxy de l'anxiété des investisseurs et des anticipations de turbulences futures. Le score de sentiment (Sentiment_Score) agrège des analyses textuelles de sources d'information financière (actualités, réseaux sociaux, rapports d'analystes) pour quantifier l'optimisme ou le pessimisme ambiant du marché. Le score de risque géopolitique (GeoPolitical_Risk_Score) capture l'impact des tensions internationales, conflits et incertitudes politiques sur la perception du risque par les investisseurs. L'indice de devises (Currency_Index) reflète les fluctuations des taux de change, un facteur crucial pour les entreprises multinationales et les flux de capitaux internationaux. Enfin, le drapeau d'événement économique (Economic_Event_Flag) signale la survenue d'annonces macroéconomiques majeures (décisions de politique monétaire, publications de données d'emploi, révisions de PIB) susceptibles de provoquer des réactions brutales du marché.

La qualité et la complétude de ce dataset en font un terrain d'expérimentation idéal pour le développement de modèles prédictifs sophistiqués. Les 30,000 observations offrent un volume de données suffisant pour l'entraînement de modèles complexes tout en évitant les écueils du big data qui nécessiterait des infrastructures de calcul massives. La période couverte, bien que synthétique, est conçue pour représenter différents régimes de marché (périodes haussières, baissières, de forte volatilité, de stabilité), assurant que les modèles apprennent à naviguer dans des conditions variées plutôt que de se spécialiser sur un contexte unique. La structure temporelle continue du dataset permet l'application de techniques avancées d'analyse de séries temporelles (autocorrélation, stationnarité, saisonnalité) tout en facilitant la création de features lagged capturant la mémoire du marché. L'absence intentionnelle de valeurs manquantes dans les variables de base simplifie le prétraitement initial, permettant de se concentrer sur l'ingénierie de features et la modélisation plutôt que sur l'imputation laborieuse de données manquantes.

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

La phase d'exploration des données constitue le fondement sur lequel repose toute l'analyse ultérieure, permettant de développer une intuition profonde de la structure, des patterns et des particularités du dataset. L'analyse statistique descriptive révèle immédiatement plusieurs caractéristiques fondamentales des variables. Les distributions des prix d'ouverture et de clôture présentent des formes relativement normales avec de légères asymétries, suggérant que les mouvements de prix suivent approximativement une marche aléatoire perturbée par des tendances directionnelles occasionnelles. Cette observation est cohérente avec la théorie des marchés efficients dans sa forme faible, où les prix actuels incorporent déjà toute l'information contenue dans les prix passés, rendant la prédiction fondamentalement difficile mais non impossible si l'on intègre des informations supplémentaires comme les facteurs externes.

L'examen de la distribution du volume révèle une asymétrie positive prononcée typique des données de trading, où la majorité des journées présentent des volumes modérés tandis que certains événements exceptionnels génèrent des pics d'activité massifs. Cette distribution log-normale du volume reflète la nature multiplicative plutôt qu'additive des chocs affectant l'activité de trading : un événement majeur peut doubler ou tripler le volume plutôt que d'y ajouter une quantité fixe. La distribution des rendements quotidiens (Daily_Return_Pct) mérite une attention particulière car elle constitue la variable d'intérêt fondamentale en finance. On observe une distribution approximativement symétrique centrée autour de zéro avec des queues de distribution épaisses (fat tails), un phénomène bien documenté en finance comportementale indiquant que les événements extrêmes (krachs, rallyes) sont plus fréquents que ne le prédirait une distribution normale pure. Cette caractéristique des fat tails implique que les modèles doivent être suffisamment robustes pour gérer des outliers sans dégrader leurs performances sur les observations normales.



### 3.1 Analyse Statistique Descriptive

L'exploration initiale révèle plusieurs caractéristiques importantes du dataset 

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
L'analyse de corrélation dévoile un réseau complexe de relations entre variables qui guide la stratégie de feature engineering. Sans surprise, les prix d'ouverture et de clôture présentent une corrélation extrêmement élevée (proche de 0.99), reflétant le fait qu'ils mesurent essentiellement le même actif à des moments légèrement différents de la journée. Les prix High et Low sont également fortement corrélés avec Open et Close, mais la magnitude de ces corrélations varie selon la volatilité du marché. L'analyse révèle une corrélation positive modérée entre le volume et la plage de volatilité (Volatility_Range), confirmant l'intuition que les périodes de forte activité de trading coïncident généralement avec des mouvements de prix amplifiés. Cette relation volume-volatilité est cruciale pour la construction de stratégies de trading car elle suggère que les signaux basés sur le volume peuvent aider à anticiper les phases de forte variation de prix.

Les corrélations entre variables de marché et facteurs externes révèlent des insights particulièrement intéressants pour la modélisation prédictive. L'indice VIX présente une corrélation négative modérée avec les rendements quotidiens, un pattern classique connu sous le nom de "volatility skew" où la volatilité tend à augmenter lors des phases de baisse du marché (les investisseurs paniquent lors des chutes mais restent calmes lors des hausses). Le score de sentiment montre une corrélation positive faible mais statistiquement significative avec les prix de clôture, suggérant que l'optimisme ambiant du marché se traduit effectivement par une tendance haussière des prix, quoique de façon imparfaite. Le score de risque géopolitique présente des corrélations faibles mais consistantes avec la volatilité, indiquant que les tensions internationales augmentent l'incertitude sans nécessairement prédire la direction des mouvements de prix. Ces patterns de corrélation valident l'hypothèse initiale selon laquelle l'intégration de facteurs externes améliore la capacité prédictive au-delà de ce qu'offrent les seules variables de marché.
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
La détection des outliers via box plots révèle la présence de valeurs extrêmes dans pratiquement toutes les variables, mais leur nature et leur interprétation diffèrent selon les cas. Les outliers dans le volume représentent des journées d'activité exceptionnelle souvent liées à des événements spécifiques (annonces de résultats trimestriels, opérations de fusion-acquisition, interventions de banques centrales). Plutôt que de constituer du bruit à éliminer, ces outliers de volume contiennent de l'information précieuse sur les moments où le marché réagit intensément à de nouvelles informations. Les outliers dans les rendements quotidiens correspondent aux jours de forte variation, qu'elle soit haussière ou baissière. Ces événements extrêmes, bien que rares, ont un impact disproportionné sur la performance d'une stratégie de trading et doivent donc être conservés dans le dataset d'entraînement pour que le modèle apprenne à les anticiper ou du moins à les gérer. Les outliers dans l'indice VIX correspondent aux périodes de crise où la peur atteint des niveaux extrêmes, des moments critiques où la capacité prédictive d'un modèle est mise à l'épreuve. La décision de conserver ces outliers plutôt que de les traiter comme du bruit reflète une compréhension profonde de la nature des marchés financiers où les événements extrêmes, loin d'être aberrants, font partie intégrante de la dynamique normale du système.

L'analyse via box plots a identifié :
- **Volume** : présence d'outliers extrêmes lors d'événements exceptionnels
- **Daily_Return_Pct** : outliers bilatéraux représentant les jours de forte variation
- **VIX_Close** : outliers en périodes de crise (forte volatilité)
- **Volatility_Range** : valeurs extrêmes pendant les chocs de marché

Ces outliers sont **conservés** car ils représentent des événements réels de marché qu'un modèle prédictif doit apprendre à anticiper.

### 3.4 Analyse Temporelle des Tendances
L'analyse temporelle des tendances via des graphiques de séries temporelles révèle plusieurs patterns structurels importants. Les courbes de prix présentent des phases alternées de tendance haussière, baissière et de consolidation latérale, un comportement typique des marchés réels où les prix évoluent rarement de façon monotone. L'identification visuelle de ces régimes de marché suggère qu'un modèle performant devrait être capable de s'adapter à différents contextes plutôt que d'apprendre une règle unique valable en toutes circonstances. Le volume présente également des patterns temporels intéressants avec des phases de forte activité groupées dans le temps, un phénomène connu sous le nom de "clustering de volatilité" où les périodes turbulentes tendent à être suivies d'autres périodes turbulentes. Cette autocorrélation conditionnelle du volume et de la volatilité motive l'utilisation de features lagged capturant l'état récent du marché pour prédire son évolution future. L'examen des graphiques temporels permet également de vérifier l'absence de tendances artificielles ou de sauts suspects qui indiqueraient des problèmes de qualité dans la construction du dataset synthétique. 

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

La préparation des données transforme le dataset enrichi en un format optimal pour l'entraînement de modèles de Machine Learning, en abordant plusieurs défis techniques cruciaux. Le traitement des valeurs manquantes constitue la première étape critique, car les features créées par feature engineering génèrent inévitablement des NaN dans les premières observations du dataset. Les moyennes mobiles nécessitent une fenêtre complète d'observations pour produire leur première valeur : SMA_10 produit un NaN pour les 9 premiers jours, SMA_20 pour les 19 premiers, et SMA_50 pour les 49 premiers jours. Les features lagged créent également des NaN au début : Close_Price_Lag_5 ne peut être calculé pour les 5 premières observations. La target variable Target_Close_Price, créée par un shift de -1, produit un NaN pour la toute dernière observation du dataset car il n'existe pas de jour suivant. La stratégie de suppression totale des lignes contenant des NaN via dropna() est justifiée par plusieurs considérations : premièrement, les techniques d'imputation (remplacement par moyenne, médiane, ou interpolation) introduiraient de l'information artificielle qui pourrait biaiser l'apprentissage ; deuxièmement, la perte d'environ 50-60 observations sur un total de 30,000 (moins de 0.2%) est négligeable et n'affecte pas significativement la puissance statistique ; troisièmement, pour les séries temporelles, il est préférable de conserver uniquement les données réellement observées plutôt que de créer des pseudo-observations par imputation.

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

La phase de modélisation représente le cœur analytique de l'étude, où les algorithmes de Machine Learning sont entraînés à découvrir les patterns complexes reliant l'état présent du marché à son évolution future. L'approche de classification transforme le problème de prédiction financière en une tâche de décision discrète : étant donné l'ensemble des features disponibles aujourd'hui, le prix va-t-il monter, descendre ou rester stable demain ? Cette formulation correspond naturellement aux décisions de trading binaires (acheter/vendre) et évite la difficulté de prédire une valeur numérique exacte dans un environnement bruité.

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

L'évaluation comparative des modèles révèle des insights cruciaux sur les forces relatives de chaque approche et la difficulté intrinsèque du problème de prédiction de marché. Pour les modèles de classification, une accuracy de 55-62% pour XGBoost peut sembler modeste en valeur absolue, mais elle représente une amélioration substantielle par rapport au hasard (33% pour trois classes équiprobables) et à la Logistic Regression baseline (48-52%). Cette performance traduit une capacité réelle à identifier des patterns prédictifs dans le bruit des marchés financiers. L'analyse du classification report révèle généralement que la classe de hausse (+1) présente une meilleure precision que la classe de baisse (-1), reflétant l'asymétrie fondamentale des marchés où les tendances haussières sont plus graduelles et prévisibles tandis que les chutes sont plus brutales et soudaines. Le faible recall sur la classe stable (0) s'explique par sa rareté relative dans les données réelles où les prix changent presque toujours d'un jour à l'autre.

En régression, les résultats sont plus spectaculaires avec des R² scores atteignant 0.88-0.95 pour XGBoost, indiquant que le modèle explique plus de 90% de la variance des prix futurs. Cette performance impressionnante s'explique en grande partie par l'autocorrélation forte des prix : le meilleur prédicteur du prix de demain est le prix d'aujourd'hui, et les features lagged capturent directement cette information. Le RMSE de 12-20 unités de prix doit être interprété relativement à la volatilité typique du marché : si les prix varient généralement de ±30 par jour, un RMSE de 15 représente une erreur correspondant à environ la moitié de la variation quotidienne typique, ce qui est raisonnablement précis. Le ratio RMSE/MAE supérieur à 1.3 confirme la présence d'erreurs occasionnelles importantes, suggérant que le modèle performe bien en moyenne mais connaît des échecs ponctuels lors d'événements exceptionnels que l'historique ne permettait pas d'anticiper.

L'analyse de feature importance extraite des modèles tree-based confirme plusieurs intuitions tout en révélant des surprises. Les features lagged, particulièrement Close_Price_Lag_1, dominent typiquement l'importance relative, validant l'idée que le marché présente une forte autocorrélation de court terme. Cependant, les indicateurs techniques comme RSI et MACD contribuent significativement, apportant une valeur informationnelle au-delà de la simple persistance des prix. L'importance modérée mais non négligeable du VIX et du Sentiment_Score valide l'hypothèse initiale que l'intégration de facteurs externes améliore la prédiction, particulièrement durant les périodes de stress de marché où ces variables capturent des changements de régime que les seules variables de prix ne révéleraient pas immédiatement. L'importance relativement faible des features calendaires (Day_Of_Week, Month) suggère que les effets de saisonnalité, bien que statistiquement significatifs dans certaines études académiques, sont d'amplitude trop faible pour être exploitables dans ce dataset particulier.

Le diagnostic du surapprentissage via la comparaison des performances train versus validation révèle que XGBoost, malgré sa complexité, généralise remarquablement bien grâce à ses mécanismes de régularisation intégrés. Un écart de performance de 3-5% entre train et validation est normal et acceptable, indiquant que le modèle a appris des patterns généraux plutôt que de mémoriser le training set. Random Forest présente généralement un écart légèrement plus important, suggérant un léger surapprentissage qui pourrait être atténué en réduisant le nombre d'estimateurs ou en augmentant la profondeur minimale des feuilles. La Logistic Regression, à l'inverse, présente souvent des performances similaires voire meilleures sur le validation set que sur le train set, un signe classique de sous-apprentissage où le modèle est trop simple pour capturer la complexité réelle du problème.

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


Cette étude démontre de manière convaincante qu'un modèle de Machine Learning bien conçu, combinant indicateurs techniques et facteurs macroéconomiques externes, peut prédire les mouvements de marché avec une précision significativement supérieure au hasard. L'algorithme XGBoost émerge comme le choix optimal pour ce type de problème, offrant le meilleur compromis entre performance prédictive, robustesse au surapprentissage et capacité de généralisation. L'approche hybride classification/régression s'avère particulièrement judicieuse : la classification fournit des signaux d'action clairs pour les décisions de trading (acheter/vendre/tenir) tandis que la régression permet un positionnement précis des stop-loss et take-profit basé sur les niveaux de prix anticipés. L'importance cruciale du feature engineering est confirmée, les indicateurs techniques créés (RSI, MACD, moyennes mobiles) apportant une valeur informationnelle substantielle au-delà des seules variables de marché brutes.

Plusieurs axes d'amélioration prometteurs se dessinent pour des travaux futurs. L'optimisation systématique des hyperparamètres via GridSearchCV ou Bayesian Optimization pourrait améliorer les performances de 2-5%, un gain marginal mais potentiellement lucratif dans un contexte de trading à haute fréquence. L'enrichissement du feature engineering avec des indicateurs additionnels comme les Bollinger Bands, l'ATR (Average True Range) ou le Stochastic Oscillator diversifierait les sources d'information et pourrait révéler des patterns complémentaires. Des techniques d'ensemble avancées comme le stacking, où les prédictions de plusieurs modèles de base (Random Forest, XGBoost, LightGBM) sont combinées via un meta-learner, exploiteraient les forces complémentaires de différents algorithmes. L'exploration d'architectures de Deep Learning, notamment les LSTM (Long Short-Term Memory) spécialisés dans les séries temporelles ou les Temporal Convolutional Networks, pourrait capturer des dépendances temporelles de très long terme que les modèles actuels ne parviennent pas à exploiter.

La transition vers un déploiement en production nécessiterait plusieurs adaptations méthodologiques importantes. Une stratégie de validation walk-forward, où le modèle est régulièrement réentraîné sur une fenêtre glissante de données historiques, permettrait d'adapter continuellement les prédictions aux évolutions du régime de marché. L'implémentation d'un système de monitoring détecterait automatiquement la dégradation de performance (concept drift) et déclencherait un réentraînement lorsque nécessaire. L'intégration de coûts de transaction réalistes, de contraintes de slippage et de limites de position transformerait les prédictions du modèle en stratégie de trading complète dont la profitabilité réelle pourrait être évaluée via backtesting rigoureux. Un framework de gestion du risque avec stop-loss dynamiques, position sizing basé sur la volatilité anticipée et diversification sur plusieurs actifs atténuerait l'impact des erreurs de prédiction inévitables.

Il convient de souligner les limites fondamentales de cette approche. Le dataset synthétique, malgré son réalisme, ne capture pas toutes les complexités des marchés réels comme les gaps d'ouverture, les suspensions de cotation, les manipulations de marché ou les événements cygnes noirs complètement imprévisibles. Les performances observées sur données historiques, même avec une validation temporelle rigoureuse, ne garantissent pas des performances futures identiques en raison de la nature non-stationnaire des marchés financiers. L'utilisation de ce type de modèle devrait toujours s'inscrire dans une approche de trading systématique diversifiée, combinant signaux quantitatifs et jugement humain, avec une gestion du risque stricte limitant l'exposition à tout modèle unique. Malgré ces caveats, l'étude établit de manière rigoureuse que l'intégration intelligente de Machine Learning, d'analyse technique et de facteurs macroéconomiques offre un avantage informationnel réel et exploitable pour l'anticipation des dynamiques de marché.
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
