# =============================================================
#  PARTIE 1a (suite) – Nuages de points & droites de régression
#  TP Deep Learning – ENSY DIPES2 2025-2026
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

DATA_PATH = "data/bank-additional-full.csv"
FIG_DIR   = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, sep=";")
df["y_bin"] = (df["y"] == "yes").astype(int)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# On retire duration (biais) et y_bin pour les scatter
plot_cols = [c for c in num_cols if c not in ["duration", "y_bin"]]

print("=" * 65)
print("  PARTIE 1a – NUAGES DE POINTS & DROITES DE RÉGRESSION")
print("=" * 65)

# ── 1. Scatter matrix (pairplot) ──────────────────────────────
print("\n▶ Génération du scatter matrix (peut prendre quelques secondes)...")

# Sélection d'un sous-ensemble de colonnes pour la lisibilité
scatter_cols = ["age", "campaign", "pdays", "previous",
                "emp.var.rate", "cons.price.idx", "cons.conf.idx",
                "euribor3m", "nr.employed", "y_bin"]

df_scatter = df[scatter_cols].copy()

fig = sns.pairplot(
    df_scatter,
    hue="y_bin",
    palette={0: "#e74c3c", 1: "#2ecc71"},
    diag_kind="kde",
    plot_kws={"alpha": 0.3, "s": 10},
    height=1.8
)
fig.fig.suptitle("Scatter matrix – variables numériques (rouge=no, vert=yes)",
                 y=1.01, fontsize=12, fontweight="bold")
fig.savefig(f"{FIG_DIR}/05_scatter_matrix.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"  ✔ Figure sauvegardée : {FIG_DIR}/05_scatter_matrix.png")

# ── 2. Scatter plots deux à deux avec droite de régression ────
# On sélectionne les paires les plus intéressantes
pairs = [
    ("age",            "euribor3m"),
    ("nr.employed",    "euribor3m"),
    ("cons.price.idx", "euribor3m"),
    ("cons.conf.idx",  "nr.employed"),
    ("emp.var.rate",   "nr.employed"),
    ("previous",       "pdays"),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, (x_col, y_col) in enumerate(pairs):
    ax = axes[i]

    # Nettoyer les NaN pour la paire
    sub = df[[x_col, y_col, "y_bin"]].dropna()
    x = sub[x_col].values
    y = sub[y_col].values

    # Scatter coloré par classe
    for cls, color, label in [(0, "#e74c3c", "no"), (1, "#2ecc71", "yes")]:
        mask = sub["y_bin"] == cls
        ax.scatter(sub.loc[mask, x_col], sub.loc[mask, y_col],
                   c=color, alpha=0.25, s=8, label=label)

    # Droite de régression (scipy linregress)
    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="navy", linewidth=2,
            label=f"Régression\ny={slope:.3f}x + {intercept:.3f}\nr={r:.3f}, p={p:.2e}")

    ax.set_xlabel(x_col, fontsize=9)
    ax.set_ylabel(y_col, fontsize=9)
    ax.set_title(f"{x_col}  vs  {y_col}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="best")

plt.suptitle("Nuages de points + droites de régression – Bank Telemarketing",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_scatter_regression.png", dpi=150)
plt.show()
print(f"  ✔ Figure sauvegardée : {FIG_DIR}/06_scatter_regression.png")

# ── 3. Paramètres des régressions – tableau récapitulatif ─────
print("\n▶ Paramètres des droites de régression :")
print(f"  {'Paire':<40} {'Pente':>8} {'Intercept':>12} {'r':>8} {'p-value':>14}")
print("  " + "-" * 85)

for x_col, y_col in pairs:
    sub = df[[x_col, y_col]].dropna()
    slope, intercept, r, p, se = stats.linregress(sub[x_col].values, sub[y_col].values)
    pair_str = f"{x_col}  →  {y_col}"
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  {pair_str:<40} {slope:>8.4f} {intercept:>12.4f} {r:>8.4f} {p:>14.2e} {sig}")

print("""
  Commentaire :
  - euribor3m et nr.employed sont très fortement corrélés (r proche
    de 1) → multicolinéarité à considérer lors de la modélisation.
  - age vs euribor3m : faible corrélation, relation non linéaire.
  - La droite de régression linéaire ne capture pas toujours bien
    la structure des données → des modèles non linéaires seront
    plus adaptés (arbres, réseaux de neurones).
""")

print("✅  Script 1b terminé. Passez à 1c_preprocessing.py")
