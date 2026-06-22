# =============================================================
#  PARTIE 1f – Importance des variables (Feature Importances)
#  TP Deep Learning – ENSY DIPES2 2025-2026
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

MODEL_DIR = "outputs/models"
FIG_DIR   = "outputs/figures"
RPT_DIR   = "outputs/reports"

X_train, X_test, y_train, y_test = joblib.load(f"{MODEL_DIR}/data_raw_split.pkl")
feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")

best_cart = joblib.load(f"{MODEL_DIR}/best_cart.pkl")
best_rf   = joblib.load(f"{MODEL_DIR}/best_rf.pkl")
best_gb   = joblib.load(f"{MODEL_DIR}/best_gb.pkl")

print("=" * 65)
print("  PARTIE 1f – IMPORTANCE DES VARIABLES")
print("=" * 65)

print("""
▶ Comment sont calculés les scores d'importance ?
  → Pour les arbres (CART, RF, GB) : chaque variable reçoit un score
    proportionnel à la réduction totale d'impureté (Gini ou entropie)
    qu'elle génère sur tous les nœuds de tous les arbres.
  → Pour RF et GB : on moyenne sur les B arbres de la forêt.
  → Ces scores sont normalisés : leur somme vaut 1.
  → Limitation : les variables à haute cardinalité tendent à être
    sur-estimées. Préférer la permutation importance pour plus de rigueur.
""")

# ── Calcul et tri des importances ─────────────────────────────
models = {
    "CART Optimisé":             best_cart,
    "Random Forest Optimisé":    best_rf,
    "Gradient Boosting Optimisé":best_gb,
}

TOP_N = 20  # Nombre de features à afficher

fig, axes = plt.subplots(1, 3, figsize=(20, 10))

all_importances = {}
for ax, (name, model) in zip(axes, models.items()):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances_sorted = importances.sort_values(ascending=False)
    all_importances[name] = importances_sorted

    top = importances_sorted.head(TOP_N)
    colors = plt.cm.YlOrRd(np.linspace(0.4, 0.9, TOP_N))[::-1]

    ax.barh(top.index[::-1], top.values[::-1], color=colors, edgecolor="black")
    ax.set_title(f"{name}\n(Top {TOP_N})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.tick_params(axis="y", labelsize=8)

plt.suptitle("Importance des variables – Comparaison des 3 modèles",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/16_feature_importances.png", dpi=150)
plt.show()
print(f"  ✔ Figure sauvegardée : {FIG_DIR}/16_feature_importances.png")

# ── Tableau récapitulatif ─────────────────────────────────────
df_imp = pd.DataFrame(all_importances)
df_imp["Moyenne"] = df_imp.mean(axis=1)
df_imp = df_imp.sort_values("Moyenne", ascending=False)

print(f"\n▶ Top 15 variables par importance moyenne :")
print(df_imp.head(15).round(4).to_string())
df_imp.to_csv(f"{RPT_DIR}/feature_importances.csv")
print(f"\n  ✔ Tableau sauvegardé : {RPT_DIR}/feature_importances.csv")

# ── Top 10 communes aux 3 modèles ─────────────────────────────
top10_cart = set(all_importances["CART Optimisé"].head(10).index)
top10_rf   = set(all_importances["Random Forest Optimisé"].head(10).index)
top10_gb   = set(all_importances["Gradient Boosting Optimisé"].head(10).index)

common = top10_cart & top10_rf & top10_gb
print(f"\n▶ Variables importantes communes aux 3 modèles (Top 10) :")
for feat in common:
    print(f"   • {feat}")

# Sauvegarde des 10 variables les plus importantes (pour Partie 2)
top10_features = df_imp.head(10).index.tolist()
joblib.dump(top10_features, f"{MODEL_DIR}/top10_features.pkl")
print(f"\n  ✔ Top 10 features sauvegardées : {MODEL_DIR}/top10_features.pkl")
print(f"  → {top10_features}")

print("""
  Commentaire :
  - Les variables macroéconomiques (euribor3m, nr.employed,
    emp.var.rate) dominent → le contexte économique est le
    facteur principal de souscription.
  - 'pdays' (jours depuis dernier contact) et 'poutcome'
    (résultat campagne précédente) figurent aussi en top.
  - CART sélectionne moins de variables que RF et GB (arbres
    peu profonds → sélection plus parcimonieuse).
""")

print("✅  Script 1h terminé. Passez à 1i_roc_curve.py")
