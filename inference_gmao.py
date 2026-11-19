"""
inference_gmao.py  (Personne A - etape 11)

Recharge le PipelineModel sauvegarde sur HDFS et produit les predictions
destinees a la GMAO pour un lot de mesures (par defaut : les 24 dernieres heures).

Point important : le modele calcule lui-meme les moyennes et ecarts-types
glissants sur 6/12/24 h. Pour scorer une mesure, il lui faut donc les 24 heures
qui la precedent pour la meme machine. Le script charge ce contexte, score
l'ensemble, puis ne garde que les lignes du lot.

Sorties (CSV, separateur ';', une ligne d'en-tete) :
  <output>/batch=<date>/predictions      une ligne par machine et par heure
  <output>/batch=<date>/alertes_machines  une ligne par machine concernee : ordre de
                                          travail si la derniere mesure depasse le seuil,
                                          surveillance si l'alerte est retombee

Usage :
    python inference_gmao.py                                   # 24 dernieres heures
    python inference_gmao.py --start 2026-09-20T00:00:00Z --end 2026-09-21T00:00:00Z
    python inference_gmao.py --incoming /maintenance/incoming/  # nouveaux fichiers
    python inference_gmao.py --base file:///C:/tmp/maintenance  # sans HDFS
"""
import argparse
import os
import time

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from features import LABEL_COL, TELEMETRY_SCHEMA

HISTORY_H = 24  # plus grande fenetre glissante du modele


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="hdfs://localhost:9000")
    p.add_argument("--model", default="/maintenance/models/gbt_maintenance.model")
    p.add_argument("--history", default="/maintenance/raw/",
                   help="mesures deja recues (fournit les 24 h de contexte)")
    p.add_argument("--incoming", default=None,
                   help="nouvelles mesures a scorer (optionnel, sinon on rejoue l'historique)")
    p.add_argument("--start", default=None, help="debut du lot (UTC, inclus)")
    p.add_argument("--end", default=None, help="fin du lot (UTC, exclu)")
    p.add_argument("--threshold", type=float, default=None,
                   help="seuil d'alerte (defaut : threshold.txt, sinon 0.5)")
    p.add_argument("--output", default="/maintenance/predictions")
    return p.parse_args()


def read_threshold(value):
    if value is not None:
        return value
    if os.path.exists("threshold.txt"):
        return float(open("threshold.txt").read().strip())
    return 0.5


def load_model(spark, path):
    """Charge le PipelineModel via la JVM.
    PipelineModel.load() construit ses chemins avec os.path.join : sous Windows il
    ecrit 'model\\metadata' (antislash), introuvable sur HDFS. La JVM, elle,
    utilise toujours des '/'."""
    return PipelineModel._from_java(spark._jvm.org.apache.spark.ml.PipelineModel.load(path))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    args = parse_args()
    thr = read_threshold(args.threshold)

    spark = (SparkSession.builder.appName("PredictiveMaintenance-Inference")
             .master("local[*]")
             .config("spark.driver.memory", "4g")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # ---------- 1. Modele ----------
    model = load_model(spark, f"{args.base}{args.model}")
    log(f"modele charge : {args.base}{args.model} ({len(model.stages)} etapes) | seuil = {thr}")

    # ---------- 2. Donnees : historique + nouvelles mesures ----------
    read = lambda path: (spark.read.schema(TELEMETRY_SCHEMA).json(f"{args.base}{path}")
                         .withColumn("ts", F.to_timestamp("timestamp")))
    data = read(args.history)
    if args.incoming:
        data = data.unionByName(read(args.incoming)).dropDuplicates(["device_id", "timestamp"])

    # Lot a scorer : par defaut les 24 dernieres heures disponibles
    if args.end:
        end = F.to_timestamp(F.lit(args.end))
    else:
        last = data.agg(F.max("ts")).first()[0]
        end = F.lit(last) + F.expr("INTERVAL 1 HOUR")
    start = F.to_timestamp(F.lit(args.start)) if args.start else end - F.expr(f"INTERVAL {HISTORY_H} HOURS")
    context_start = start - F.expr(f"INTERVAL {HISTORY_H} HOURS")

    # Lot + 24 h de contexte par machine (necessaires aux fenetres glissantes)
    ctx = data.filter((F.col("ts") >= context_start) & (F.col("ts") < end))

    # ---------- 3. Prediction (imputation + fenetres + scaler + GBT) ----------
    pred = (model.transform(ctx)
            .filter(F.col("ts") >= start)
            .withColumn("proba_panne", F.round(vector_to_array("probability")[1], 4))
            .withColumn("alerte", (F.col("proba_panne") >= thr).cast("int"))
            .withColumn("seuil", F.lit(thr))
            .withColumn("date_prediction", F.date_format(F.current_timestamp(), "yyyy-MM-dd'T'HH:mm:ss'Z'"))
            .cache())

    fmt = lambda c: F.date_format(c, "yyyy-MM-dd'T'HH:mm'Z'")  # en UTC, sans conversion Python
    lot = pred.agg(fmt(F.min("ts")).alias("debut"), fmt(F.max("ts")).alias("fin"),
                   F.count("*").alias("n"), F.countDistinct("device_id").alias("machines")).first()
    log(f"lot score : {lot.debut} -> {lot.fin} | {lot.n:,} mesures | {lot.machines} machines")

    predictions = pred.select("device_id", "timestamp", "proba_panne", "alerte",
                              "seuil", "date_prediction").orderBy("device_id", "timestamp")

    # Une ligne par machine concernee. La decision porte sur l'etat a la DERNIERE mesure :
    #   - derniere mesure au-dessus du seuil -> ordre de travail (priorite HAUTE si >= 0.9)
    #   - alerte plus tot dans le lot mais redescendue -> surveillance, sans ordre de travail
    alertes = (pred.groupBy("device_id")
               .agg(F.max_by("proba_panne", "ts").alias("proba_derniere_mesure"),
                    F.max("proba_panne").alias("proba_max"),
                    F.sum("alerte").alias("heures_en_alerte"),
                    F.min(F.when(F.col("alerte") == 1, F.col("timestamp"))).alias("premiere_alerte"),
                    F.max("timestamp").alias("derniere_mesure"))
               .filter(F.col("heures_en_alerte") > 0)
               .withColumn("priorite", F.when(F.col("proba_derniere_mesure") >= 0.9, "HAUTE")
                                        .when(F.col("proba_derniere_mesure") >= thr, "MOYENNE")
                                        .otherwise("SURVEILLANCE"))
               .withColumn("action", F.when(F.col("priorite") == "SURVEILLANCE",
                                            "Surveillance renforcee, pas d'ordre de travail")
                                      .otherwise("Ordre de travail : inspection sous 24 h"))
               .orderBy(F.desc("proba_derniere_mesure")))

    # ---------- 4. Ecriture pour la GMAO ----------
    batch = lot.fin[:13]
    out = f"{args.base}{args.output}/batch={batch}"
    for name, df in (("predictions", predictions), ("alertes_machines", alertes)):
        (df.coalesce(1).write.mode("overwrite")
           .option("header", True).option("sep", ";").csv(f"{out}/{name}"))
    log(f"ecrit : {out}/predictions et {out}/alertes_machines")

    print("\n=== Machines signalees a la GMAO ===")
    alertes.show(20, truncate=False)
    n_ot = alertes.filter(F.col("priorite") != "SURVEILLANCE").count()
    log(f"{n_ot} ordre(s) de travail, {alertes.count() - n_ot} machine(s) en surveillance "
        f"sur {lot.machines}")

    # Controle (demo uniquement) : si les etiquettes sont connues, on compare la decision
    # prise a la derniere mesure avec ce qui s'est reellement passe dans les 24 h suivantes
    if pred.filter(F.col(LABEL_COL).isNotNull()).limit(1).count():
        last = pred.groupBy("device_id").agg(F.max_by("alerte", "ts").alias("a"),
                                             F.max_by(LABEL_COL, "ts").alias("y"))
        c = last.agg(F.sum(((F.col("a") == 1) & (F.col("y") == 1)).cast("int")).alias("ok"),
                     F.sum(((F.col("a") == 1) & (F.col("y") == 0)).cast("int")).alias("fausses"),
                     F.sum(((F.col("a") == 0) & (F.col("y") == 1)).cast("int")).alias("ratees")).first()
        log(f"controle (etiquettes connues) : {c.ok} ordre(s) justifie(s), "
            f"{c.fausses} fausse(s) alerte(s), {c.ratees} panne(s) non signalee(s)")

    spark.stop()


if __name__ == "__main__":
    main()
