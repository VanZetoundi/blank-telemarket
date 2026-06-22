# =============================================================
#  PARTIE 1b – Prétraitement des données
#  TP Deep Learning – ENSY DIPES2 2025-2026
# =============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

DATA_PATH  = "data/bank-additional-full.csv"
MODEL_DIR  = "outputs/models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, sep=";")

print("=" * 65)
print("  PARTIE 1b – PRÉTRAITEMENT DES DONNÉES")
print("=" * 65)
print(f"\n  Dimensions initiales : {df.shape}")

# ── 1. Suppression de 'duration' (fuite de données) ──────────
# 'duration' n'est connue qu'après l'appel → biais si incluse
df.drop(columns=["duration"], inplace=True)
print("\n  ✔ Colonne 'duration' supprimée (biais post-appel)")

# ── 2. Gestion des 'unknown' ──────────────────────────────────
# On remplace 'unknown' par le mode de la colonne concernée
cat_cols = df.select_dtypes(include="object").columns.drop("y").tolist()

print("\n  ▶ Remplacement des valeurs 'unknown' par le mode :")
for col in cat_cols:
    n_unknown = (df[col] == "unknown").sum()
    if n_unknown > 0:
        mode_val = df[col][df[col] != "unknown"].mode()[0]
        df[col] = df[col].replace("unknown", mode_val)
        print(f"     {col:<20} : {n_unknown} 'unknown' → '{mode_val}'")

# ── 3. Vérification valeurs manquantes ────────────────────────
miss = df.isnull().sum().sum()
print(f"\n  ▶ Valeurs NaN restantes : {miss}")

# ── 4. Encodage de la cible ───────────────────────────────────
df["y"] = df["y"].map({"no": 0, "yes": 1})
print(f"\n  ✔ Encodage cible : 'no' → 0  |  'yes' → 1")

# ── 5. Variables dummy pour les catégorielles ─────────────────
print(f"\n  ▶ Colonnes catégorielles à encoder ({len(cat_cols)}) :")
print(f"     {cat_cols}")

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print(f"\n  ✔ After get_dummies : {df.shape[1]} colonnes")

# ── 6. Séparation features / target ───────────────────────────
X = df.drop(columns=["y"])
y = df["y"]

print(f"\n  ▶ Features (X) : {X.shape}")
print(f"  ▶ Cible    (y) : {y.shape}  |  {y.value_counts().to_dict()}")

# ── 7. Séparation train / test  (80% / 20%) ───────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y          # conserver la proportion des classes
)

print(f"\n  ▶ Train : {X_train.shape[0]} instances")
print(f"  ▶ Test  : {X_test.shape[0]} instances")
print(f"  ▶ Proportion 'yes' train : {y_train.mean()*100:.1f}%")
print(f"  ▶ Proportion 'yes' test  : {y_test.mean()*100:.1f}%")

# ── 8. Standardisation des features numériques ────────────────
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

scaler = StandardScaler()
X_train_sc = X_train.copy()
X_test_sc  = X_test.copy()

X_train_sc[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_sc[num_cols]  = scaler.transform(X_test[num_cols])

print(f"\n  ✔ StandardScaler appliqué sur {len(num_cols)} colonnes numériques")

# ── 9. Sauvegarde des données prétraitées ─────────────────────
joblib.dump((X_train, X_test, y_train, y_test),
            f"{MODEL_DIR}/data_raw_split.pkl")
joblib.dump((X_train_sc, X_test_sc, y_train, y_test),
            f"{MODEL_DIR}/data_scaled_split.pkl")
joblib.dump(scaler,
            f"{MODEL_DIR}/scaler.pkl")
joblib.dump(list(X.columns),
            f"{MODEL_DIR}/feature_names.pkl")

print(f"\n  ✔ Données sauvegardées dans {MODEL_DIR}/")
print(f"     - data_raw_split.pkl    (X_train, X_test, y_train, y_test)")
print(f"     - data_scaled_split.pkl (version standardisée)")
print(f"     - scaler.pkl")
print(f"     - feature_names.pkl")

# ── 10. Aperçu final ──────────────────────────────────────────
print(f"\n  ▶ Aperçu X_train (3 lignes) :")
print(X_train.head(3).T.to_string())

print("\n✅  Script 1c terminé. Passez à 1d_modeles_simples.py")
