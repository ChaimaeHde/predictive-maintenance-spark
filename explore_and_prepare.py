"""
explore_and_prepare.py  (Personne A — étapes 2 à 4)
  2. Lecture des données depuis HDFS + exploration (chiffres pour le rapport)
  3. Imputation distribuée (médiane)
  4. Fenêtres glissantes 6h / 12h / 24h par machine
Sortie : jeu de features en Parquet sur HDFS (/maintenance/features)
         + fichiers CSV d'exploration dans ./outputs/ pour le rapport

Usage :
    python explore_and_prepare.py
    python explore_and_prepare.py --input file:///C:/chemin/telemetry.json --output file:///C:/tmp/features
"""
import argparse
import os

from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from features import FEATURE_COLS, LABEL_COL, SENSORS, TELEMETRY_SCHEMA, feature_stages

HDFS = "hdfs://localhost:9000"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=f"{HDFS}/maintenance/raw/")
    p.add_argument("--output", default=f"{HDFS}/maintenance/features")
    p.add_argument("--outdir", default="outputs", help="dossier local des CSV d'exploration")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    spark = (SparkSession.builder
             .appName("PredictiveMaintenance-Prepare")
             .master("local[*]")
             .config("spark.driver.memory", "4g")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.shuffle.partitions", "16")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # ---------- Étape 2 : lecture + exploration ----------
    df = spark.read.schema(TELEMETRY_SCHEMA).json(args.input)
    df = df.withColumn("ts", F.to_timestamp("timestamp")).cache()

    n = df.count()
    print("\n=== Schéma ===")
    df.printSchema()
    print(f"Lignes : {n:,} | Machines : {df.select('device_id').distinct().count()}")
    df.select(F.min("ts").alias("debut"), F.max("ts").alias("fin")).show(truncate=False)

    print("=== Valeurs manquantes par colonne ===")
    nulls = df.select([F.round(F.avg(F.col(c).isNull().cast("int")) * 100, 2).alias(c)
                       for c in SENSORS + ["operating_hours", LABEL_COL]])
    nulls.show()

    print("=== Répartition des classes ===")
    classes = (df.groupBy(LABEL_COL).count()
                 .withColumn("pourcentage", F.round(F.col("count") / n * 100, 2))
                 .orderBy(LABEL_COL))
    classes.show()

    print("=== Statistiques des capteurs par classe ===")
    by_class = df.groupBy(LABEL_COL).agg(
        *[F.round(F.avg(c), 2).alias(f"moy_{c}") for c in SENSORS],
        *[F.round(F.stddev(c), 2).alias(f"std_{c}") for c in SENSORS],
    ).orderBy(LABEL_COL)
    by_class.show(truncate=False)

    # Export pour le rapport
    nulls.toPandas().to_csv(f"{args.outdir}/valeurs_manquantes.csv", index=False)
    classes.toPandas().to_csv(f"{args.outdir}/repartition_classes.csv", index=False)
    by_class.toPandas().to_csv(f"{args.outdir}/stats_par_classe.csv", index=False)

    # ---------- Étapes 3 & 4 : imputation + rolling ----------
    prep = Pipeline(stages=feature_stages()).fit(df)
    feats = prep.transform(df)

    # Vérifications
    print("=== Nulls restants dans les features (doit être 0) ===")
    feats.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in FEATURE_COLS[:4]]).show()

    print("=== Exemple : une machine, colonnes température ===")
    (feats.filter(F.col("device_id") == "DEV-8800")
          .orderBy("ts")
          .select("timestamp", "temperature", "temperature_imp",
                  "temperature_mean_6h", "temperature_std_6h", "temperature_mean_24h", LABEL_COL)
          .show(8, truncate=False))

    print("=== Signal : moyenne des rolling features par classe ===")
    signal = feats.groupBy(LABEL_COL).agg(
        *[F.round(F.avg(c), 2).alias(c) for c in
          ["temperature_mean_6h", "temperature_std_6h", "vibration_level_mean_6h", "pressure_mean_6h"]]
    ).orderBy(LABEL_COL)
    signal.show(truncate=False)
    signal.toPandas().to_csv(f"{args.outdir}/signal_rolling_par_classe.csv", index=False)

    # Sauvegarde (Parquet, partitionné par mois) pour la Personne B
    out_cols = ["device_id", "timestamp", "ts"] + FEATURE_COLS + [LABEL_COL]
    (feats.select(*out_cols)
          .withColumn("month", F.date_format("ts", "yyyy-MM"))
          .write.mode("overwrite").partitionBy("month").parquet(args.output))
    print(f"\nFeatures écrites dans : {args.output}  ({len(FEATURE_COLS)} features)")

    spark.stop()


if __name__ == "__main__":
    main()
