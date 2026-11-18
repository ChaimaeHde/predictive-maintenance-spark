# Maintenance Prédictive Industrielle — Spark MLlib & HDFS

Mini-projet 04 du module Big Data (ENSAM Rabat, filière INDIA-SD).

Objectif : prédire la probabilité de panne d'un équipement industriel dans les **24 heures** à partir de données de capteurs stockées sur **HDFS**, avec un pipeline **PySpark ML** distribué (feature engineering, Random Forest, Gradient-Boosted Trees), afin d'alimenter le plan de maintenance préventive d'une GMAO.

## Architecture

```
generate_telemetry.py ──► HDFS /maintenance/raw (JSON Lines)
                               │
                               ▼
              Spark : Imputer ─► Rolling stats 6h/12h/24h (SQLTransformer)
                               │
                               ▼
        VectorAssembler ─► StandardScaler ─► GBT / Random Forest
                               │   (CrossValidator, PR-AUC)
                               ▼
        HDFS /maintenance/models/gbt_maintenance.model (PipelineModel)
                               │
                               ▼
              Inférence batch ─► prédictions pour la GMAO
```

## Données

Données de télémétrie synthétiques, une mesure par heure et par machine :

```json
{"device_id": "DEV-8890", "temperature": 92.1, "vibration_level": 18.5, "pressure": 4.2,
 "operating_hours": 3450, "failure_in_24h": 1, "timestamp": "2026-10-06T08:00:00Z"}
```

- 50 machines × 180 jours = 216 000 enregistrements
- Classe positive (`failure_in_24h = 1`) : environ 3,3 % → données déséquilibrées
- Environ 2 % de valeurs manquantes par capteur
- Dégradation progressive simulée avant chaque panne, plus quelques fausses alertes

Les données ne sont pas versionnées : `generate_telemetry.py` les régénère à l'identique (seed fixe).

## Structure du dépôt

| Fichier | Description | État |
|---|---|---|
| `generate_telemetry.py` | Génération des données synthétiques | ✅ |
| `features.py` | Schéma, imputation (médiane), rolling mean/std 6h-12h-24h | ✅ |
| `explore_and_prepare.py` | Exploration des données + calcul et écriture des features sur HDFS | ✅ |
| `train_predictive_maintenance.py` | Split temporel, pondération des classes, CrossValidator (PR-AUC), évaluation, sauvegarde du PipelineModel | ✅ |
| `inference_gmao.py` | Rechargement du PipelineModel et prédictions pour la GMAO (CSV sur HDFS) | ✅ |
| `threshold.txt` | Seuil d'alerte retenu à l'entraînement, lu par `inference_gmao.py` | ✅ |
| `model_evaluation_report.pdf` | Rapport : matrices de confusion, précision, rappel, courbes ROC/PR | ✅ |

## Résultats (test : septembre 2026)

| Modèle | Seuil | PR-AUC | ROC-AUC | Précision | Rappel | F1 |
|---|---|---|---|---|---|---|
| **GBT pondéré (retenu)** | 0,75 | 0,843 | 0,964 | **0,748** | 0,827 | **0,785** |
| Random Forest pondéré | 0,50 | **0,852** | **0,976** | 0,668 | **0,841** | 0,745 |
| GBT sur-échantillonné | 0,75 | 0,834 | 0,973 | 0,723 | **0,841** | 0,778 |

- Meilleur GBT (validation croisée 3 plis, PR-AUC) : `maxDepth=7`, `maxIter=50`, `stepSize=0.1`.
- Seuil choisi sur août (modèle entraîné d'avril à juillet) en minimisant `5 × pannes manquées + fausses alertes`.
- Évaluation par panne, script d'inférence exécuté chaque heure : **42 pannes sur 42 signalées à l'avance**, avance médiane de **21,5 h** (minimum 9 h).
- Variable la plus utile : la vibration, en particulier son écart-type sur 24 h.

Une PR-AUC sans compétence vaudrait 0,031 (proportion de pannes). Les données étant synthétiques, ces scores ne sont pas transposables tels quels à une usine réelle.

## Environnement

| Outil | Version |
|---|---|
| Java | 17 |
| Hadoop (HDFS pseudo-distribué) | 3.5.0 |
| Python | 3.12 |
| PySpark | 4.2 – 4.3 |
| pandas, numpy, matplotlib, scikit-learn | — |

Sous Windows : `winutils.exe` et `hadoop.dll` dans `%HADOOP_HOME%\bin`, et `PYSPARK_PYTHON=python`.

## Exécution

```bash
# 1. Démarrer HDFS
start-dfs.cmd            # Windows (start-dfs.sh sous Linux)
jps                      # NameNode + DataNode

# 2. Générer les données et les charger sur HDFS
pip install pandas numpy pyspark
python generate_telemetry.py
hdfs dfs -mkdir -p /maintenance/raw
hdfs dfs -put telemetry.json /maintenance/raw/

# 3. Exploration + feature engineering
python explore_and_prepare.py
hdfs dfs -ls /maintenance/features

# 4. Entraînement, validation croisée, évaluation, sauvegarde du modèle
pip install matplotlib scikit-learn pyarrow
python train_predictive_maintenance.py --quick   # test rapide (grille réduite)
python train_predictive_maintenance.py           # grille complète
hdfs dfs -ls /maintenance/models

# 5. Prédictions pour la GMAO (24 dernières heures par défaut)
python inference_gmao.py
hdfs dfs -ls -R /maintenance/predictions
```

Les figures et tableaux de résultats sont écrits dans `outputs/` (non versionné).

## Prédictions transmises à la GMAO

`inference_gmao.py` recharge le `PipelineModel`, score un lot de mesures brutes et écrit deux fichiers CSV (séparateur `;`) dans `/maintenance/predictions/batch=<date>/` :

| Fichier | Contenu |
|---|---|
| `predictions` | `device_id ; timestamp ; proba_panne ; alerte ; seuil ; date_prediction` (une ligne par machine et par heure) |
| `alertes_machines` | Une ligne par machine concernée : probabilité à la dernière mesure et maximale, heures en alerte, première alerte, priorité, action. C'est le fichier importé par la GMAO : ordre de travail si la dernière mesure dépasse le seuil (priorité HAUTE ou MOYENNE), simple surveillance si l'alerte est retombée. |

Le modèle calcule lui-même les statistiques glissantes : pour scorer une mesure, il a besoin des **24 heures précédentes** de la même machine. Le script charge donc ce contexte depuis `/maintenance/raw` avant de scorer le lot.

## Méthodologie (résumé)

- **Imputation distribuée** par la médiane, apprise sur le jeu d'entraînement uniquement.
- **Fenêtres glissantes temporelles** par machine (6h, 12h, 24h) : moyenne et écart-type de chaque capteur, calculés uniquement sur le passé pour éviter toute fuite d'information.
- **Split temporel** train/test (pas de split aléatoire, à cause des fenêtres glissantes).
- **Déséquilibre** traité par pondération des classes (`weightCol`).
- **Optimisation** par `CrossValidator` + `ParamGridBuilder` sur la **PR-AUC**, plus adaptée que l'accuracy aux classes rares.
- **Seuil d'alerte** choisi sur août (modèle entraîné d'avril à juillet) en minimisant un coût métier : une panne ratée coûte 5 fausses alertes.

## Auteures

- Chaimae Haddouche
- Loubna Haouach
