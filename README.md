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
| `train_predictive_maintenance.py` | Pipeline complet, gestion du déséquilibre, CrossValidator, sauvegarde | ⏳ |
| `inference_gmao.py` | Rechargement du PipelineModel et prédictions pour la GMAO | ⏳ |
| `model_evaluation_report.pdf` | Matrice de confusion, précision, rappel, courbes ROC/PR | ⏳ |

## Environnement

| Outil | Version |
|---|---|
| Java | 17 |
| Hadoop (HDFS pseudo-distribué) | 3.5.0 |
| Python | 3.12 |
| PySpark | 4.x |
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
```

## Méthodologie (résumé)

- **Imputation distribuée** par la médiane, apprise sur le jeu d'entraînement uniquement.
- **Fenêtres glissantes temporelles** par machine (6h, 12h, 24h) : moyenne et écart-type de chaque capteur, calculés uniquement sur le passé pour éviter toute fuite d'information.
- **Split temporel** train/test (pas de split aléatoire, à cause des fenêtres glissantes).
- **Déséquilibre** traité par pondération des classes (`weightCol`).
- **Optimisation** par `CrossValidator` + `ParamGridBuilder` sur la **PR-AUC**, plus adaptée que l'accuracy aux classes rares.

## Auteures

- Chaimae Haddouche
- Loubna Haouach
