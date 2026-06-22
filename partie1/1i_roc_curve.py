# =============================================================
#  PARTIE 1g – Courbes ROC & AUC
#  TP Deep Learning – ENSY DIPES2 2025-2026
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.metrics import roc_curve, auc, roc_auc_score

MODEL_DIR = "outputs/models"
FIG_DIR   = "outputs/figures"
RPT_DIR   = "outputs/reports"

X_train, X_test, y_train, y_test = joblib.load(f"{MODEL_DIR}/data_raw_split.pkl")

best_cart = joblib.load(f"{MODEL_DIR}/best_cart.pkl")
best_rf   = joblib.load(f"{MODEL_DIR}/best_rf.pkl")
best_gb   = joblib.load(f"{MODEL_DIR}/best_gb.pkl")

print("=" * 65)
print("  PARTIE 1g – COURBES ROC & AUC")
print("=" * 65)

print("""
▶ 1. Comment prédire un score avec RF ou GB ?
   → model.predict_proba(X_test)[:, 1]
   Cette méthode retourne la probabilité estimée d'appartenir
   à la classe 1 ('yes') pour chaque instance.
   Contrairement à predict() qui renvoie une classe binaire,
   predict_proba() donne un score continu → exploitable pour
   la courbe ROC (on fait varier le seuil de décision).

   La courbe ROC trace TPR (rappel) vs FPR pour tous les seuils.
   L'AUC mesure l'aire sous cette courbe (1 = parfait, 0.5 = aléatoire).
""")

# ── Prédiction des scores ─────────────────────────────────────
models_info = {
    "CART Optimisé":              (best_cart, "#3498db", "--"),
    "Random Forest Optimisé":     (best_rf,   "#2ecc71", "-"),
    "Gradient Boosting Optimisé": (best_gb,   "#e74c3c", "-"),
}

fig, ax = plt.subplots(figsize=(9, 7))

auc_results = []

for name, (model, color, ls) in models_info.items():
    # Score de probabilité
    y_prob = model.predict_proba(X_test)[:, 1]

    # Courbe ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)
    auc_results.append({"Modèle": name, "AUC-ROC": round(auc_score, 4)})

    ax.plot(fpr, tpr, linestyle=ls, color=color, linewidth=2.5,
            label=f"{name}  (AUC = {auc_score:.4f})")

    # Point optimal (seuil maximisant TPR - FPR)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    ax.scatter(fpr[best_idx], tpr[best_idx], color=color,
               marker="o", s=80, zorder=5)
    print(f"   {name}")
    print(f"     AUC-ROC       : {auc_score:.4f}")
    print(f"     Seuil optimal : {best_thresh:.4f}")
    print(f"     TPR optimal   : {tpr[best_idx]:.4f}")
    print(f"     FPR optimal   : {fpr[best_idx]:.4f}")

# Diagonal (aléatoire)
ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Aléatoire (AUC=0.5)")

ax.set_xlabel("Taux de Faux Positifs (FPR)", fontsize=12)
ax.set_ylabel("Taux de Vrais Positifs (TPR / Rappel)", fontsize=12)
ax.set_title("Courbes ROC – Comparaison des 3 classifieurs optimisés",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/17_roc_curves.png", dpi=150)
plt.show()
print(f"\n  ✔ Figure sauvegardée : {FIG_DIR}/17_roc_curves.png")

# ── Tableau AUC final ─────────────────────────────────────────
df_auc = pd.DataFrame(auc_results).sort_values("AUC-ROC", ascending=False)
print("\n▶ Récapitulatif AUC-ROC :")
print(df_auc.to_string(index=False))
df_auc.to_csv(f"{RPT_DIR}/auc_results.csv", index=False)
print(f"\n  ✔ Tableau sauvegardé : {RPT_DIR}/auc_results.csv")

# ── Zoom sur la région d'intérêt (FPR < 0.3) ─────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for name, (model, color, ls) in models_info.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)
    ax.plot(fpr, tpr, linestyle=ls, color=color, linewidth=2.5,
            label=f"{name}  (AUC = {auc_score:.4f})")

ax.plot([0, 1], [0, 1], "k--", linewidth=1.2)
ax.set_xlim([-0.005, 0.30])
ax.set_ylim([0.40, 1.01])
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_title("Zoom ROC – FPR ∈ [0, 0.30]", fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/18_roc_zoom.png", dpi=150)
plt.show()
print(f"  ✔ Figure sauvegardée : {FIG_DIR}/18_roc_zoom.png")

print("""
  Commentaire final :
  - Le Gradient Boosting obtient en général le meilleur AUC-ROC,
    suivi du Random Forest, puis de CART.
  - La courbe ROC du GB reste au-dessus des autres pour tous les
    seuils → meilleure capacité discriminante globale.
  - Le seuil optimal (indice de Youden) permet de maximiser
    TPR - FPR selon les priorités métier (ex : minimiser les
    faux négatifs si on veut maximiser les souscriptions).
""")

print("✅  Partie 1 terminée ! Tous les résultats sont dans outputs/")
