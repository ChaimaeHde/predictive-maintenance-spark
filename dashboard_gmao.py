"""
dashboard_gmao.py - Tableau de bord de maintenance predictive (Streamlit)

Ce que fait l'interface :
  1. On choisit une journee (septembre 2026 = mois de test, jamais vu a l'entrainement).
  2. Le bouton « Lancer la detection » execute inference_gmao.py : le modele GBT
     sauvegarde sur HDFS predit la probabilite de panne de chaque machine, heure par heure.
  3. L'interface relit sur HDFS les deux fichiers produits et affiche :
     les ordres de travail, les machines en surveillance, une carte de chaleur
     machines x heures et la courbe de probabilite d'une machine.

Lancement (HDFS demarre, dans le dossier du projet) :
    pip install streamlit
    streamlit run dashboard_gmao.py
"""
import datetime as dt
import glob
import io
import subprocess
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

BASE = "hdfs://localhost:9000"      # ou file:///C:/... pour travailler sans HDFS
PRED_DIR = "/maintenance/predictions"
HERE = Path(__file__).resolve().parent
RED, ORANGE, GREY = "#A6192E", "#E07B39", "#8A8478"

st.set_page_config(page_title="Maintenance prédictive", page_icon="🛠️", layout="wide")


# ---------------------------------------------------------------------------
# Lecture des fichiers produits par inference_gmao.py
# ---------------------------------------------------------------------------
def read_csv_from_store(path):
    """Lit un dossier CSV ecrit par Spark, sur HDFS (hdfs dfs -cat) ou en local."""
    if BASE.startswith("file://"):
        files = glob.glob(BASE[len("file://"):] + path + "/*.csv")
        return pd.concat([pd.read_csv(f, sep=";") for f in files]) if files else None
    # Guillemets obligatoires sous Windows : le '=' de batch=... couperait l'argument
    r = subprocess.run(f'hdfs dfs -cat "{path}/*.csv"', shell=True, capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return pd.read_csv(io.StringIO(r.stdout), sep=";")


@st.cache_data(show_spinner=False)
def load_batch(batch):
    folder = f"{PRED_DIR}/batch={batch}"
    return read_csv_from_store(f"{folder}/predictions"), read_csv_from_store(f"{folder}/alertes_machines")


def run_inference(day):
    start = f"{day:%Y-%m-%d}T00:00:00Z"
    end = f"{day + dt.timedelta(days=1):%Y-%m-%d}T00:00:00Z"
    cmd = [sys.executable, str(HERE / "inference_gmao.py"), "--base", BASE, "--start", start, "--end", end]
    return subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Barre laterale
# ---------------------------------------------------------------------------
st.sidebar.title("🛠️ Maintenance prédictive")
st.sidebar.caption("Mini-projet 04 · Big Data · ENSAM Rabat")
day = st.sidebar.date_input("Journée à analyser", value=dt.date(2026, 9, 8),
                            min_value=dt.date(2026, 9, 1), max_value=dt.date(2026, 9, 27))
batch = f"{day:%Y-%m-%d}T23"
st.sidebar.caption("Septembre = mois de test : le modèle ne l'a jamais vu pendant l'entraînement.")

if st.sidebar.button("Lancer la détection", type="primary", use_container_width=True):
    with st.spinner("Spark charge le modèle depuis HDFS et prédit les pannes… (environ 30 s)"):
        res = run_inference(day)
    if res.returncode != 0:
        st.sidebar.error("L'inférence a échoué. HDFS est-il démarré (jps) ?")
        st.sidebar.code(res.stderr[-1500:])
    else:
        st.sidebar.success("Prédictions écrites sur HDFS.")
        load_batch.clear()

preds, alerts = load_batch(batch)

# ---------------------------------------------------------------------------
# Page principale
# ---------------------------------------------------------------------------
st.title(f"Détection des pannes du {day:%d/%m/%Y}")

if preds is None:
    st.info("Aucune prédiction pour cette journée. Cliquez sur **Lancer la détection** dans le menu de gauche.")
    st.stop()

threshold = float(preds["seuil"].iloc[0])
alerts = alerts if alerts is not None else pd.DataFrame(columns=["device_id", "priorite"])
orders = alerts[alerts["priorite"].isin(["HAUTE", "MOYENNE"])]
watch = alerts[alerts["priorite"] == "SURVEILLANCE"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Machines surveillées", preds["device_id"].nunique())
c2.metric("Ordres de travail", len(orders))
c3.metric("En surveillance", len(watch))
c4.metric("Seuil d'alerte", f"{threshold:.2f}")

# --- Ordres de travail ---
st.subheader("Ordres de travail à transmettre à la GMAO")
if orders.empty:
    st.success("Aucune machine ne demande d'intervention à la fin de cette journée.")
else:
    st.dataframe(
        orders.rename(columns={"device_id": "Machine", "proba_derniere_mesure": "Probabilité actuelle",
                               "proba_max": "Probabilité max", "heures_en_alerte": "Heures en alerte",
                               "premiere_alerte": "Première alerte", "priorite": "Priorité", "action": "Action"})
              .drop(columns=["derniere_mesure"]),
        hide_index=True, use_container_width=True)

with st.expander(f"Machines en surveillance ({len(watch)}) : alerte retombée, pas d'intervention"):
    st.dataframe(watch[["device_id", "proba_derniere_mesure", "proba_max", "heures_en_alerte", "premiere_alerte"]]
                 .rename(columns={"device_id": "Machine", "proba_derniere_mesure": "Probabilité actuelle",
                                  "proba_max": "Probabilité max", "heures_en_alerte": "Heures en alerte",
                                  "premiere_alerte": "Première alerte"}),
                 hide_index=True, use_container_width=True)

# --- Carte de chaleur machines x heures ---
st.subheader("Probabilité de panne, machine par machine et heure par heure")
heat = preds.assign(heure=pd.to_datetime(preds["timestamp"]).dt.strftime("%Hh"))
flagged = list(alerts["device_id"])
heat = heat[heat["device_id"].isin(flagged)] if flagged else heat.head(0)
if heat.empty:
    st.caption("Aucune machine n'a dépassé le seuil pendant la journée.")
else:
    order = list(alerts.sort_values("proba_derniere_mesure", ascending=False)["device_id"])
    chart = (alt.Chart(heat).mark_rect(stroke="white", strokeWidth=1)
             .encode(x=alt.X("heure:O", title="Heure (UTC)"),
                     y=alt.Y("device_id:N", title=None, sort=order),
                     color=alt.Color("proba_panne:Q", title="Probabilité",
                                     scale=alt.Scale(domain=[0, 1], range=["#F7F5F2", RED])),
                     tooltip=["device_id", "timestamp", alt.Tooltip("proba_panne:Q", format=".3f")])
             .properties(height=34 * len(order) + 40))
    st.altair_chart(chart, use_container_width=True)
    st.caption("Seules les machines qui ont dépassé le seuil au moins une fois sont affichées. "
               "Une ligne qui reste rouge jusqu'à 23h signale une dégradation durable.")

# --- Courbe d'une machine ---
st.subheader("Évolution d'une machine")
default = order[0] if flagged else sorted(preds["device_id"].unique())[0]
machines = sorted(preds["device_id"].unique())
dev = st.selectbox("Machine", machines, index=machines.index(default))
# heures en texte (UTC) : un axe temporel serait converti dans le fuseau du navigateur
one = preds[preds["device_id"] == dev].assign(heure=lambda d: pd.to_datetime(d["timestamp"]).dt.strftime("%Hh"))
line = (alt.Chart(one).mark_line(point=alt.OverlayMarkDef(color=RED, size=50), color=RED, strokeWidth=2)
        .encode(x=alt.X("heure:O", title="Heure (UTC)", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("proba_panne:Q", title="Probabilité de panne dans les 24 h", scale=alt.Scale(domain=[0, 1])),
                tooltip=["timestamp", alt.Tooltip("proba_panne:Q", format=".3f")]))
rule = (alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(strokeDash=[6, 4], color=GREY)
        .encode(y="y:Q"))
label = (alt.Chart(pd.DataFrame({"y": [threshold], "t": [f"seuil {threshold:.2f}"]}))
         .mark_text(align="left", dx=4, dy=-8, color=GREY).encode(y="y:Q", text="t:N"))
st.altair_chart((line + rule + label).properties(height=320), use_container_width=True)

st.caption(f"Source : {BASE}{PRED_DIR}/batch={batch} · modèle GBT (gbt_maintenance.model) · "
           f"décision sur la dernière mesure de la journée.")
