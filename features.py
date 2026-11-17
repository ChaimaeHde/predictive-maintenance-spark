"""
features.py
Étapes de feature engineering réutilisables dans le pipeline final
(train_predictive_maintenance.py) :
  1. Imputer (médiane) sur les capteurs
  2. SQLTransformer : moyennes et écarts-types glissants sur 6h, 12h, 24h
     par machine, en n'utilisant que le passé (pas de fuite d'information)

Ces deux étapes sont des stages Spark ML : elles seront donc sauvegardées
dans le PipelineModel avec le modèle.
"""
from pyspark.ml.feature import Imputer, SQLTransformer
from pyspark.sql.types import (DoubleType, IntegerType, LongType, StringType,
                               StructField, StructType)

SENSORS = ["temperature", "vibration_level", "pressure"]
WINDOWS_H = [6, 12, 24]
LABEL_COL = "failure_in_24h"

# Schéma explicite : lecture plus rapide et types garantis
TELEMETRY_SCHEMA = StructType([
    StructField("device_id", StringType()),
    StructField("temperature", DoubleType()),
    StructField("vibration_level", DoubleType()),
    StructField("pressure", DoubleType()),
    StructField("operating_hours", LongType()),
    StructField(LABEL_COL, IntegerType()),
    StructField("timestamp", StringType()),
])

IMPUTED = [f"{c}_imp" for c in SENSORS]

ROLLING_COLS = [
    f"{c}_{stat}_{h}h" for c in SENSORS for h in WINDOWS_H for stat in ("mean", "std")
]

# Colonnes d'entrée du VectorAssembler
FEATURE_COLS = IMPUTED + ["operating_hours"] + ROLLING_COLS


def build_imputer():
    """Imputation distribuée par la médiane (calculée sur le train uniquement)."""
    return Imputer(strategy="median", inputCols=SENSORS, outputCols=IMPUTED)


def build_rolling_sql():
    """
    Fenêtres glissantes temporelles par machine.
    RANGE BETWEEN X PRECEDING AND CURRENT ROW sur le timestamp en secondes :
    la fenêtre couvre les X dernières secondes, même s'il manque des mesures.
    """
    exprs = []
    for c in SENSORS:
        for h in WINDOWS_H:
            w = (f"(PARTITION BY device_id ORDER BY ts_sec "
                 f"RANGE BETWEEN {h * 3600} PRECEDING AND CURRENT ROW)")
            exprs.append(f"AVG({c}_imp) OVER {w} AS {c}_mean_{h}h")
            # stddev = NULL quand la fenêtre ne contient qu'une ligne -> 0
            exprs.append(f"COALESCE(STDDEV({c}_imp) OVER {w}, 0.0) AS {c}_std_{h}h")

    statement = (
        "SELECT *, " + ",\n       ".join(exprs) + "\n"
        "FROM (SELECT *, CAST(CAST(timestamp AS TIMESTAMP) AS BIGINT) AS ts_sec "
        "FROM __THIS__)"
    )
    return SQLTransformer(statement=statement)


def feature_stages():
    """Stages à placer en tête du Pipeline : [Imputer, SQLTransformer]."""
    return [build_imputer(), build_rolling_sql()]
