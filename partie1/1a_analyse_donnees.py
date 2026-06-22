# =============================================================
#  PARTIE 1a – Analyse exploratoire des données (EDA)
#  TP Deep Learning – ENSY DIPES2 2025-2026
#  Dataset : Bank Telemarketing (UCI)
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Chemins ──────────────────────────────────────────────────
DATA_PATH = "data/bank-additional-full.csv"
FIG_DIR   = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── 1. Chargement ─────────────────────────────────────────────
# Le séparateur est ";" dans ce dataset UCI
df = pd.read_csv(DATA_PATH, sep=";")

print("=" * 65)
print("  PARTIE 1a – ANALYSE EXPLORATOIRE DES DONNÉES")
print("=" * 65)

# ── 2. Dimensions ─────────────────────────────────────────────
print(f"\n▶ [shape]  {df.shape[0]} instances  ×  {df.shape[1]} colonnes")

# ── 3. Résumé rapide ──────────────────────────────────────────
print("\n▶ [info]")
df.info()

# ── 4. Statistiques descriptives ──────────────────────────────
print("\n▶ [describe – variables numériques]")
print(df.describe().T.to_string())

# ── 5. Premières lignes ───────────────────────────────────────
print("\n▶ [head(5)]")
print(df.head(5).to_string())

# ── 6. Variable cible ─────────────────────────────────────────
print("\n" + "=" * 65)
print("  VARIABLE CIBLE  →  y  (souscription dépôt à terme)")
print("=" * 65)

vc = df["y"].value_counts()
print(f"\n[value_counts]\n{vc}")
pct_yes = vc["yes"] / len(df) * 100
pct_no  = vc["no"]  / len(df) * 100
print(f"\n  Classe 'no'  (non-souscription) : {vc['no']}  ({pct_no:.1f}%)")
print(f"  Classe 'yes' (souscription)     : {vc['yes']}  ({pct_yes:.1f}%)")

print("""
  ─ Type d'apprentissage : SUPERVISÉ – Classification binaire.
  ─ Le dataset est DÉSÉQUILIBRÉ (≈89% 'no' vs ≈11% 'yes').
  ─ Métriques recommandées :
      • AUC-ROC         → robuste au déséquilibre
      • F1-score (yes)  → compromis précision/rappel sur la classe minoritaire
      • Matrice de confusion → visualiser les FP/FN
      • Éviter l'accuracy seule : un classifieur naïf prédit
        toujours 'no' et obtient déjà ~89% d'accuracy !
""")

# ── 7. Types des variables ────────────────────────────────────
cat_cols = df.select_dtypes(include="object").columns.drop("y").tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"▶ Variables catégorielles ({len(cat_cols)}) :")
print(f"   {cat_cols}")
print(f"\n▶ Variables numériques ({len(num_cols)}) :")
print(f"   {num_cols}")

# ── 8. Valeurs manquantes ─────────────────────────────────────
print("\n▶ Valeurs manquantes :")
miss = df.isnull().sum()
if miss.sum() == 0:
    print("  → Aucune valeur NaN détectée.")
    print("  NB : Des valeurs 'unknown' existent dans les colonnes catégorielles.")
else:
    print(miss[miss > 0])

# Compter les 'unknown' par colonne catégorielle
print("\n▶ Valeurs 'unknown' par colonne catégorielle :")
for col in cat_cols:
    n = (df[col] == "unknown").sum()
    if n > 0:
        print(f"   {col:<20} : {n} ({n/len(df)*100:.1f}%)")

# ── 9. Distribution de la cible ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Bar chart
vc.plot(kind="bar", ax=axes[0], color=["#e74c3c", "#2ecc71"], edgecolor="black")
axes[0].set_title("Distribution de la variable cible y")
axes[0].set_xlabel("Classe")
axes[0].set_ylabel("Nombre d'instances")
axes[0].set_xticklabels(["no", "yes"], rotation=0)
for p in axes[0].patches:
    axes[0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2, p.get_height()),
                     ha="center", va="bottom", fontsize=10)

# Pie chart
axes[1].pie(vc.values, labels=["no", "yes"], autopct="%1.1f%%",
            colors=["#e74c3c", "#2ecc71"], startangle=90)
axes[1].set_title("Proportion des classes")

plt.suptitle("Équilibre des classes – Bank Telemarketing", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_distribution_target.png", dpi=150)
plt.show()
print(f"\n  ✔ Figure sauvegardée : {FIG_DIR}/01_distribution_target.png")

# ── 10. Distribution des variables numériques ─────────────────
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    axes[i].hist(df[col], bins=40, color="#3498db", edgecolor="white", alpha=0.85)
    axes[i].set_title(col, fontsize=9)
    axes[i].set_xlabel("")

plt.suptitle("Distribution des variables numériques", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_distributions_num.png", dpi=150)
plt.show()
print(f"  ✔ Figure sauvegardée : {FIG_DIR}/02_distributions_num.png")

# ── 11. Matrice de corrélations ───────────────────────────────
# Encoder la cible temporairement pour la corrélation
df_corr = df.copy()
df_corr["y_bin"] = (df_corr["y"] == "yes").astype(int)
corr = df_corr[num_cols + ["y_bin"]].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))  # triangle supérieur masqué
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, mask=mask, ax=ax,
            annot_kws={"size": 8})
ax.set_title("Matrice de corrélations (variables numériques + cible)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_matrice_correlations.png", dpi=150)
plt.show()
print(f"  ✔ Figure sauvegardée : {FIG_DIR}/03_matrice_correlations.png")

# ── 12. Corrélation avec la cible ─────────────────────────────
print("\n▶ Corrélation des features numériques avec y (binaire) :")
corr_target = corr["y_bin"].drop("y_bin").sort_values(ascending=False)
print(corr_target.to_string())

print("""
  Commentaire :
  - 'duration' (durée du dernier appel) est la variable la plus
    corrélée avec y, mais elle n'est connue qu'après l'appel →
    à écarter pour un modèle prédictif réaliste.
  - 'nr.employed' (nbre employés économie) et 'euribor3m'
    (taux interbancaire) ont une corrélation négative avec y.
  - 'pdays' et 'previous' sont peu corrélées avec les autres.
  → Sélection de variables : pertinente, notamment pour exclure
    'duration' et traiter les redondances.
""")

# ── 13. Distribution des catégorielles vs cible ───────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for i, col in enumerate(cat_cols[:8]):
    ct = pd.crosstab(df[col], df["y"], normalize="index") * 100
    ct["yes"].sort_values(ascending=False).plot(
        kind="bar", ax=axes[i], color="#2ecc71", edgecolor="black", alpha=0.85
    )
    axes[i].set_title(f"% souscription par '{col}'", fontsize=9)
    axes[i].set_ylabel("% yes")
    axes[i].tick_params(axis="x", labelsize=7, rotation=45)

plt.suptitle("Taux de souscription par variable catégorielle", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_cat_vs_target.png", dpi=150)
plt.show()
print(f"  ✔ Figure sauvegardée : {FIG_DIR}/04_cat_vs_target.png")

print("\n✅  Script 1a terminé. Passez à 1b_scatter_correlation.py")
