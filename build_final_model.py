"""
build_final_model.py

Reconstruit le modele final (gbt_maintenance.model) sans relancer toute la
validation croisee (environ 3 h). Il reprend directement les meilleurs
hyperparametres trouves par train_predictive_maintenance.py :
    GBT maxDepth=7, maxIter=50, stepSize=0.1, ponderation des classes, graine 42.
Le modele obtenu est identique a celui de l'entrainement complet
(test septembre : PR-AUC 0,843 ; 829 pannes detectees, 280 fausses alertes a 0,75).

Usage :
    python build_final_model.py
    python build_final_model.py --base file:///C:/tmp/maintenance
"""
import argparse

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from features import FEATURE_COLS, LABEL_COL, TELEMETRY_SCHEMA, feature_stages

TEST_START = "2026-09-01"
BEST = dict(maxDepth=7, maxIter=50, stepSize=0.1)
THRESHOLD = 0.75


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="hdfs://localhost:9000")
    args = p.parse_args()

    spark = (SparkSession.builder.appName("PredictiveMaintenance-FinalModel")
             .master("local[*]")
             .config("spark.driver.memory", "4g")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    raw = (spark.read.schema(TELEMETRY_SCHEMA).json(f"{args.base}/maintenance/raw/")
           .withColumn("ts", F.to_timestamp("timestamp")))
    train_raw = raw.filter(F.col("ts") < TEST_START)

    # Meme preparation que train_predictive_maintenance.py
    prep = Pipeline(stages=feature_stages()).fit(train_raw)
    feats = prep.transform(raw).cache()
    train = feats.filter(F.col("ts") < TEST_START)
    test = feats.filter(F.col("ts") >= TEST_START)

    n = train.count()
    n_pos = train.filter(F.col(LABEL_COL) == 1).count()
    w = (n - n_pos) / n_pos
    train_w = (train.withColumn("weight", F.when(F.col(LABEL_COL) == 1, F.lit(w)).otherwise(F.lit(1.0)))
               .coalesce(8).cache())
    print(f"train = {n:,} lignes, poids classe 1 = {w:.2f}")

    gbt = GBTClassifier(labelCol=LABEL_COL, featuresCol="features", weightCol="weight", seed=42, **BEST)
    model = Pipeline(stages=[VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw"),
                             StandardScaler(inputCol="features_raw", outputCol="features"),
                             gbt]).fit(train_w)

    # Controle sur septembre : on doit retrouver les chiffres du rapport
    pred = model.transform(test).withColumn("p1", vector_to_array("probability")[1])
    pr_auc = BinaryClassificationEvaluator(labelCol=LABEL_COL, metricName="areaUnderPR").evaluate(pred)
    lab, alert = F.col(LABEL_COL), F.col("p1") >= THRESHOLD
    c = pred.agg(F.sum(((lab == 1) & alert).cast("int")).alias("tp"),
                 F.sum(((lab == 0) & alert).cast("int")).alias("fp")).first()
    print(f"controle septembre : PR-AUC = {pr_auc:.3f} (attendu 0,843) | "
          f"pannes detectees = {c.tp} (attendu 829) | fausses alertes = {c.fp} (attendu 280)")

    path = f"{args.base}/maintenance/models/gbt_maintenance.model"
    PipelineModel(stages=prep.stages + model.stages).write().overwrite().save(path)
    print(f"modele sauvegarde : {path}")
    spark.stop()


if __name__ == "__main__":
    main()
