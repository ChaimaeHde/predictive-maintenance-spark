"""
train_predictive_maintenance.py  (Personne B - etapes 5 a 10)

Entraine, optimise et evalue Random Forest et GBT pour predire une panne dans
les 24 h, puis sauvegarde le PipelineModel complet (prep + modele) sur HDFS.

  5. Split temporel : train = avril-aout, test = septembre
  6. Desequilibre : ponderation (weightCol) + comparaison avec sur-echantillonnage
  7. Pipelines : VectorAssembler -> StandardScaler -> classifieur
  8. CrossValidator + ParamGridBuilder sur la PR-AUC
  9. Evaluation (PR-AUC, ROC-AUC, matrices de confusion, courbes, importances)
 10. Sauvegarde du PipelineModel (Imputer, SQLTransformer, ..., GBT) sur HDFS

Choix de conception :
  - Imputer ajuste sur le TRAIN uniquement ; les fenetres glissantes sont
    ensuite calculees sur toutes les donnees (elles ne regardent que le passe),
    puis on coupe par date : ainsi les premieres heures de septembre ont leur
    historique d'aout.
  - Les features sont calculees AVANT le CrossValidator (sinon les fenetres
    seraient recalculees sur des lignes tirees au hasard).
  - Le seuil de decision est choisi sur AOUT avec un modele entraine sur
    avril-juillet seulement (pas sur le test), en minimisant un cout metier
    (une panne ratee coute COST_FN fois une fausse alerte).

Usage :
    python train_predictive_maintenance.py --quick      # grille reduite, test rapide
    python train_predictive_maintenance.py              # grille complete
    python train_predictive_maintenance.py --base file:///C:/tmp/maintenance --quick
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.metrics import precision_recall_curve, roc_curve

from features import FEATURE_COLS, LABEL_COL, TELEMETRY_SCHEMA, feature_stages

TEST_START = "2026-09-01"   # test = septembre
VAL_START = "2026-08-01"    # validation (choix du seuil) = aout
COST_FN = 5.0               # une panne ratee coute 5 fausses alertes
SEED = 42
THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
REPORT_THRESHOLDS = (0.3, 0.5, 0.7)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="hdfs://localhost:9000",
                   help="racine du systeme de fichiers (HDFS ou file:///...)")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--quick", action="store_true", help="grille reduite, 2 folds")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--parallelism", type=int, default=1)
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def confusion(df, thr):
    """Matrice de confusion au seuil thr (calculee par Spark)."""
    y = (F.col("p1") >= thr).cast("int")
    lab = F.col(LABEL_COL)
    r = df.agg(
        F.sum(((lab == 1) & (y == 1)).cast("int")).alias("tp"),
        F.sum(((lab == 0) & (y == 1)).cast("int")).alias("fp"),
        F.sum(((lab == 1) & (y == 0)).cast("int")).alias("fn"),
        F.sum(((lab == 0) & (y == 0)).cast("int")).alias("tn")).first()
    tp, fp, fn, tn = (int(r[k] or 0) for k in ("tp", "fp", "fn", "tn"))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(threshold=thr, tp=tp, fp=fp, fn=fn, tn=tn,
                precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4))


def score(model, df):
    return (model.transform(df)
            .withColumn("p1", vector_to_array("probability")[1])
            .select(LABEL_COL, "rawPrediction", "p1").cache())


def best_threshold(pred_val):
    """Seuil minimisant COST_FN*FN + FP sur la validation (aout)."""
    # un seul passage Spark pour tous les seuils
    lab = F.col(LABEL_COL)
    aggs = []
    for i, t in enumerate(THRESHOLDS):
        y = (F.col("p1") >= t).cast("int")
        aggs += [F.sum(((lab == 1) & (y == 1)).cast("int")).alias(f"tp{i}"),
                 F.sum(((lab == 0) & (y == 1)).cast("int")).alias(f"fp{i}"),
                 F.sum(((lab == 1) & (y == 0)).cast("int")).alias(f"fn{i}"),
                 F.sum(((lab == 0) & (y == 0)).cast("int")).alias(f"tn{i}")]
    r = pred_val.agg(*aggs).first()
    rows = []
    for i, t in enumerate(THRESHOLDS):
        tp, fp, fn, tn = (int(r[f"{k}{i}"] or 0) for k in ("tp", "fp", "fn", "tn"))
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        rows.append(dict(threshold=t, tp=tp, fp=fp, fn=fn, tn=tn,
                         precision=round(prec, 4), recall=round(rec, 4),
                         f1=round(2 * prec * rec / max(prec + rec, 1e-9), 4)))
    for r in rows:
        r["cost"] = COST_FN * r["fn"] + r["fp"]
    best = min(rows, key=lambda r: r["cost"])
    return best["threshold"], pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_curves(pdfs, figdir):
    plt.figure(figsize=(6, 5))
    for name, pdf in pdfs.items():
        fpr, tpr, _ = roc_curve(pdf[LABEL_COL], pdf["p1"])
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("Taux de faux positifs"); plt.ylabel("Rappel (TPR)")
    plt.title("Courbes ROC (test : septembre)"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{figdir}/roc_curves.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 5))
    for name, pdf in pdfs.items():
        p, r, _ = precision_recall_curve(pdf[LABEL_COL], pdf["p1"])
        plt.plot(r, p, label=name)
    base = next(iter(pdfs.values()))[LABEL_COL].mean()
    plt.axhline(base, color="k", ls="--", lw=0.8, label=f"hasard ({base:.3f})")
    plt.xlabel("Rappel"); plt.ylabel("Precision")
    plt.title("Courbes Precision-Rappel (test : septembre)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{figdir}/pr_curves.png", dpi=150); plt.close()


def plot_confusions(cms, thr, figdir):
    fig, axes = plt.subplots(1, len(cms), figsize=(5.2 * len(cms), 4.5))
    axes = np.atleast_1d(axes)
    for ax, (name, c) in zip(axes, cms.items()):
        m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(m, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{m[i, j]:,}", ha="center", va="center",
                        color="white" if m[i, j] > m.max() / 2 else "black", fontsize=13)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pas d'alerte", "Alerte"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Pas de panne", "Panne"])
        ax.set_xlabel("Predit"); ax.set_ylabel("Reel")
        ax.set_title(f"{name} (seuil {thr})")
    plt.tight_layout(); plt.savefig(f"{figdir}/confusion_matrices.png", dpi=150); plt.close()


def plot_importance(imp_df, figdir):
    top = imp_df.head(10).iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top["feature"], top["importance"], color="#3b6fb6")
    plt.xlabel("Importance"); plt.title("Top 10 des features (GBT)")
    plt.tight_layout(); plt.savefig(f"{figdir}/feature_importance.png", dpi=150); plt.close()


# --------------------------------------------------------------------------
def main():
    args = parse_args()
    figdir = f"{args.outdir}/figures"
    os.makedirs(figdir, exist_ok=True)
    model_path = f"{args.base}/maintenance/models/gbt_maintenance.model"

    spark = (SparkSession.builder.appName("PredictiveMaintenance-Train")
             .master("local[*]")
             .config("spark.driver.memory", "4g")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.ui.showConsoleProgress", "false")
             .config("spark.local.dir", os.environ.get("SPARK_LOCAL_DIR", "C:/bigdata/spark-tmp"))
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # ---------- Etape 5 : lecture + split temporel ----------
    raw = (spark.read.schema(TELEMETRY_SCHEMA).json(f"{args.base}/maintenance/raw/")
           .withColumn("ts", F.to_timestamp("timestamp")))
    train_raw = raw.filter(F.col("ts") < TEST_START)

    # Imputer (medianes) appris sur le train uniquement ; fenetres sur tout l'historique
    prep_model = Pipeline(stages=feature_stages()).fit(train_raw)
    feats = prep_model.transform(raw).cache()
    train = feats.filter(F.col("ts") < TEST_START)
    test = feats.filter(F.col("ts") >= TEST_START).cache()
    n_train, n_test = train.count(), test.count()
    log(f"train={n_train:,} lignes | test={n_test:,} lignes (attendu 183 600 / 32 400)")

    # ---------- Etape 6 : desequilibre ----------
    n_pos = train.filter(F.col(LABEL_COL) == 1).count()
    w = (n_train - n_pos) / n_pos
    log(f"classe 1 : {n_pos:,} ({n_pos / n_train:.2%}) -> poids = {w:.2f}")
    add_w = lambda df: df.withColumn(
        "weight", F.when(F.col(LABEL_COL) == 1, F.lit(w)).otherwise(F.lit(1.0)))
    # peu de partitions + cache : l'entrainement des arbres est bien plus rapide en local
    train_w = add_w(train).coalesce(8).cache()
    train_w.count()

    # ---------- Etape 7 : pipelines ----------
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features")
    gbt = GBTClassifier(labelCol=LABEL_COL, featuresCol="features", weightCol="weight", seed=SEED)
    rf = RandomForestClassifier(labelCol=LABEL_COL, featuresCol="features", weightCol="weight", seed=SEED)
    gbt_pipe = Pipeline(stages=[assembler, scaler, gbt])
    rf_pipe = Pipeline(stages=[assembler, scaler, rf])

    # ---------- Etape 8 : CrossValidator sur la PR-AUC ----------
    evaluator = BinaryClassificationEvaluator(labelCol=LABEL_COL, metricName="areaUnderPR")
    if args.quick:
        gbt_grid = (ParamGridBuilder().addGrid(gbt.maxDepth, [5])
                    .addGrid(gbt.maxIter, [20]).addGrid(gbt.stepSize, [0.1]).build())
        rf_grid = (ParamGridBuilder().addGrid(rf.numTrees, [50])
                   .addGrid(rf.maxDepth, [5]).build())
        folds = 2
    else:
        gbt_grid = (ParamGridBuilder().addGrid(gbt.maxDepth, [3, 5, 7])
                    .addGrid(gbt.maxIter, [20, 50]).addGrid(gbt.stepSize, [0.05, 0.1]).build())
        rf_grid = (ParamGridBuilder().addGrid(rf.numTrees, [50, 100])
                   .addGrid(rf.maxDepth, [5, 10]).build())
        folds = args.folds

    def run_cv(name, pipe, grid, chunk):
        """CrossValidator par blocs de combinaisons, avec reprise : les PR-AUC deja
        calculees sont stockees dans cv_progress_<name>.json (un crash ne perd pas tout)."""
        log(f"CrossValidator {name} : {len(grid)} combinaisons x {folds} folds (blocs de {chunk})")
        t0 = time.time()
        pfile = f"{args.outdir}/cv_progress_{name.lower()}.json"
        done = json.load(open(pfile)) if os.path.exists(pfile) else {}
        key = lambda pm: json.dumps({p.name: v for p, v in pm.items()}, sort_keys=True)
        for s in range(0, len(grid), chunk):
            sub = grid[s:s + chunk]
            if all(key(pm) in done for pm in sub):
                log(f"{name} bloc {s // chunk + 1} deja calcule, saute")
                continue
            cv = CrossValidator(estimator=pipe, estimatorParamMaps=sub, evaluator=evaluator,
                                numFolds=folds, parallelism=args.parallelism, seed=SEED).fit(train_w)
            for pm, m in zip(sub, cv.avgMetrics):
                done[key(pm)] = float(m)
            json.dump(done, open(pfile, "w"))
            log(f"{name} bloc {s // chunk + 1}/{-(-len(grid) // chunk)} termine")
        rows = []
        for pm in grid:
            row = {p.name: v for p, v in pm.items()}
            row["PR_AUC_cv"] = round(done[key(pm)], 4)
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("PR_AUC_cv", ascending=False)
        df.to_csv(f"{args.outdir}/cv_avgmetrics_{name.lower()}.csv", index=False)
        print(df.to_string(index=False))
        best_pm = max(grid, key=lambda pm: done[key(pm)])
        best_model = pipe.fit(train_w, best_pm)   # refit sur tout le train (comme CrossValidator)
        log(f"{name} termine en {(time.time() - t0) / 60:.1f} min")
        return best_model

    best_gbt = run_cv("GBT", gbt_pipe, gbt_grid, chunk=4)
    best_rf = run_cv("RF", rf_pipe, rf_grid, chunk=2)
    bg = best_gbt.stages[-1]
    best_params = dict(maxDepth=bg.getMaxDepth(), maxIter=bg.getMaxIter(), stepSize=bg.getStepSize())
    log(f"meilleurs hyperparametres GBT : {best_params}")
    br = best_rf.stages[-1]
    log(f"meilleurs hyperparametres RF : numTrees={br.getNumTrees}, maxDepth={br.getMaxDepth()}")

    # ---------- Seuil choisi sur AOUT (modele entraine avril-juillet) ----------
    train_jul = train_w.filter(F.col("ts") < VAL_START)
    val_aug = train.filter(F.col("ts") >= VAL_START)
    gbt_val = GBTClassifier(labelCol=LABEL_COL, featuresCol="features", weightCol="weight",
                            seed=SEED, **best_params)
    val_model = Pipeline(stages=[assembler, scaler, gbt_val]).fit(train_jul)
    thr_gbt, thr_table = best_threshold(score(val_model, val_aug))
    thr_table.to_csv(f"{args.outdir}/threshold_selection_val_aout.csv", index=False)
    log(f"seuil retenu (cout {COST_FN:g}*FN + FP sur aout) : {thr_gbt}")
    # meme seuil pour le RF (modele entraine avril-juillet avec ses meilleurs parametres)
    rf_val = RandomForestClassifier(labelCol=LABEL_COL, featuresCol="features", weightCol="weight",
                                    seed=SEED, numTrees=br.getNumTrees, maxDepth=br.getMaxDepth())
    rf_val_model = Pipeline(stages=[assembler, scaler, rf_val]).fit(train_jul)
    thr_rf, _ = best_threshold(score(rf_val_model, val_aug))

    # ---------- Etape 9 : evaluation sur le test ----------
    roc_eval = BinaryClassificationEvaluator(labelCol=LABEL_COL, metricName="areaUnderROC")
    results, pdfs, cms, thr_tables = [], {}, {}, []

    def evaluate(name, model, thr):
        pred = score(model, test)
        pr = evaluator.evaluate(pred)
        roc = roc_eval.evaluate(pred)
        for t in sorted(set(REPORT_THRESHOLDS) | {thr}):
            c = confusion(pred, t); c["model"] = name; thr_tables.append(c)
        c = confusion(pred, thr)
        results.append(dict(model=name, threshold=thr, PR_AUC=round(pr, 4), ROC_AUC=round(roc, 4),
                            precision=c["precision"], recall=c["recall"], f1=c["f1"],
                            tp=c["tp"], fp=c["fp"], fn=c["fn"], tn=c["tn"]))
        pdfs[name] = pred.select(LABEL_COL, "p1").toPandas()
        cms[name] = c
        log(f"{name} @ {thr}: PR-AUC={pr:.3f} ROC-AUC={roc:.3f} P={c['precision']} R={c['recall']} F1={c['f1']}")

    evaluate("GBT", best_gbt, thr_gbt)
    evaluate("RF", best_rf, thr_rf)

    # Comparaison ponderation / sur-echantillonnage (GBT, memes hyperparametres)
    over_pos = train.filter(F.col(LABEL_COL) == 1).sample(withReplacement=True, fraction=w, seed=SEED)
    train_over = train.filter(F.col(LABEL_COL) == 0).union(over_pos).cache()
    gbt_over = GBTClassifier(labelCol=LABEL_COL, featuresCol="features", seed=SEED, **best_params)
    over_model = Pipeline(stages=[assembler, scaler, gbt_over]).fit(train_over)
    evaluate("GBT (sur-echantillonnage)", over_model, thr_gbt)

    comp = pd.DataFrame(results)
    comp.to_csv(f"{args.outdir}/model_comparison.csv", index=False)
    pd.DataFrame(thr_tables).to_csv(f"{args.outdir}/threshold_table_test.csv", index=False)
    print("\n=== Comparaison sur le test (septembre) ===")
    print(comp.to_string(index=False))

    plot_curves({k: pdfs[k] for k in ("GBT", "RF")}, figdir)
    plot_confusions({k: cms[k] for k in ("GBT", "RF")},
                    f"GBT {thr_gbt} / RF {thr_rf}", figdir)

    imp = pd.DataFrame({"feature": FEATURE_COLS,
                        "importance": best_gbt.stages[-1].featureImportances.toArray()})
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(f"{args.outdir}/feature_importance.csv", index=False)
    plot_importance(imp, figdir)
    print("\n=== Top 10 importances (GBT) ===")
    print(imp.head(10).to_string(index=False))

    # ---------- Etape 10 : sauvegarde du pipeline complet ----------
    final = PipelineModel(stages=prep_model.stages + best_gbt.stages)
    final.write().overwrite().save(model_path)
    log(f"PipelineModel sauvegarde : {model_path}")
    reloaded = PipelineModel.load(model_path)
    (reloaded.transform(raw.filter(F.col("ts") >= TEST_START))
     .select("device_id", "timestamp", "probability", "prediction").show(5, truncate=False))

    with open("threshold.txt", "w") as f:
        f.write(f"{thr_gbt}\n")
    with open(f"{args.outdir}/summary.json", "w") as f:
        json.dump(dict(threshold_gbt=thr_gbt, threshold_rf=thr_rf, cost_fn=COST_FN,
                       class_weight=w, best_gbt=best_params,
                       best_rf=dict(numTrees=br.getNumTrees, maxDepth=br.getMaxDepth())), f, indent=2)
    log("termine. Seuil GBT ecrit dans threshold.txt (a transmettre a la Personne A).")
    spark.stop()


if __name__ == "__main__":
    main()
