"""
generate_telemetry.py
Génère des données de télémétrie industrielle synthétiques (JSON Lines),
au format exact du Mini-Projet 04 :
{"device_id": "DEV-8890", "temperature": 92.1, "vibration_level": 18.5,
 "pressure": 4.2, "operating_hours": 3450, "failure_in_24h": 1,
 "timestamp": "2026-10-06T08:00:00Z"}

Logique de simulation :
- 1 mesure par heure et par machine.
- Comportement normal : valeurs de base propres à chaque machine + bruit
  + cycle journalier sur la température.
- Avant chaque panne : dégradation progressive (24 à 72 h) -> température et
  vibration montent, pression baisse.
- failure_in_24h = 1 sur les 24 heures qui précèdent une panne.
- Fausses alertes : pics courts sans panne (pour que le problème ne soit pas trivial).
- Valeurs manquantes injectées aléatoirement (null) pour tester l'imputation.

Usage :
    python generate_telemetry.py
    python generate_telemetry.py --devices 50 --days 180 --out telemetry.json
"""
import argparse

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Générateur de télémétrie synthétique")
    p.add_argument("--devices", type=int, default=50, help="nombre de machines")
    p.add_argument("--days", type=int, default=180, help="durée simulée en jours")
    p.add_argument("--start", default="2026-04-01", help="date de début (UTC)")
    p.add_argument("--missing", type=float, default=0.02, help="taux de valeurs manquantes par capteur")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="telemetry.json")
    return p.parse_args()


def simulate_device(dev_idx, timestamps, rng, missing_rate):
    n = len(timestamps)
    hours = np.arange(n)

    # Valeurs de base propres à la machine
    base_temp = rng.normal(70, 4)
    base_vib = rng.normal(10, 1.5)
    base_press = rng.normal(4.0, 0.3)
    start_hours = int(rng.integers(500, 8000))

    # Fonctionnement normal
    temp = base_temp + 3 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 1.5, n)
    vib = base_vib + rng.normal(0, 1.0, n)
    press = base_press + rng.normal(0, 0.15, n)
    label = np.zeros(n, dtype=int)

    # Pannes : une toutes les 15 à 45 jours environ
    t = int(rng.integers(72, 24 * 30))
    while t < n:
        ramp_len = int(rng.integers(24, 72))
        lo = max(0, t - ramp_len)
        ramp = np.linspace(0, 1, t - lo) ** 2  # dégradation qui s'accélère
        temp[lo:t] += ramp * rng.uniform(12, 25)
        vib[lo:t] += ramp * rng.uniform(5, 10)
        press[lo:t] -= ramp * rng.uniform(0.5, 1.2)
        label[max(0, t - 24):t] = 1
        t += int(rng.integers(24 * 15, 24 * 45))

    # Fausses alertes (pics courts sans panne)
    for _ in range(int(rng.integers(2, 6))):
        s = int(rng.integers(0, n - 12))
        length = int(rng.integers(3, 12))
        temp[s:s + length] += rng.uniform(5, 12)
        vib[s:s + length] += rng.uniform(2, 5)

    df = pd.DataFrame({
        "device_id": f"DEV-{8800 + dev_idx}",
        "temperature": np.round(temp, 1),
        "vibration_level": np.round(vib, 1),
        "pressure": np.round(press, 2),
        "operating_hours": start_hours + hours,
        "failure_in_24h": label,
        "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
    })

    # Valeurs manquantes
    for col in ["temperature", "vibration_level", "pressure"]:
        mask = rng.random(n) < missing_rate
        df.loc[mask, col] = np.nan

    return df


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    timestamps = pd.date_range(args.start, periods=args.days * 24, freq="h", tz="UTC")

    df = pd.concat(
        [simulate_device(i, timestamps, rng, args.missing) for i in range(args.devices)],
        ignore_index=True,
    )
    df = df.sort_values(["timestamp", "device_id"]).reset_index(drop=True)

    # JSON Lines : une ligne = un enregistrement (NaN -> null)
    df.to_json(args.out, orient="records", lines=True)

    print(f"Fichier écrit      : {args.out}")
    print(f"Lignes             : {len(df):,}")
    print(f"Machines           : {df['device_id'].nunique()}")
    print(f"Période            : {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Taux de panne (=1) : {df['failure_in_24h'].mean():.2%}")
    print("Valeurs manquantes :")
    print(df[["temperature", "vibration_level", "pressure"]].isna().mean().map("{:.2%}".format).to_string())


if __name__ == "__main__":
    main()
