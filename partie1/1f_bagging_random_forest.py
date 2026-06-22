# =============================================================
#  PARTIE 1e – Bagging & Random Forest
#  TP Deep Learning – ENSY DIPES2 2025-2026
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, classification_report

MODEL_DIR = "outputs/models"
FIG_DIR   = "outputs/figures"
RPT_DIR   = "outputs/reports"

X_train, X_test, y_train, y_test = joblib.load(f"{MODEL_DIR}/data_raw_split.pkl")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 65)
print("  PARTIE 1e – BAGGING & RANDOM FOREST")
print("=" * 65)

# ── 1. Bagging ────────────────────────────────────────────────
# Pour faire du Bagging avec RandomForestClassifier :
# max_features=1.0 → chaque arbre voit TOUTES les features (= Bagging pur)
print("\n▶ 1. Bagging (RandomForestClassifier avec max_features=1.0)")

B_values = [10, 25, 50, 100, 200]
bag_train_aucs = []
bag_test_aucs  = []
bag_times      = []

import time
for B in B_values:
    t0 = time.time()
    bag = RandomForestClassifier(
        n_estimators=B,
        max_features=1.0,   # ← Bagging pur : toutes les variables
        random_state=42,
        n_jobs=-1
    )
    bag.fit(X_train, y_train)
    tr_auc = roc_auc_score(y_train, bag.predict_proba(X_train)[:, 1])
    te_auc = roc_auc_score(y_test,  bag.predict_proba(X_test)[:, 1])
    elapsed = time.time() - t0
    bag_train_aucs.append(tr_auc)
    bag_test_aucs.append(te_auc)
    bag_times.append(elapsed)
    print(f"   B={B:>3} → Train AUC={tr_auc:.4f}  Test AUC={te_auc:.4f}  ({elapsed:.1f}s)")

print("""
  Commentaire :
  - Avec B croissant, la performance sur le test se stabilise.
  - Le temps de calcul augmente quasi linéairement avec B.
  - Au-delà de B=100, le gain est marginal (convergence).
""")

fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()
ax1.plot(B_values, bag_train_aucs, "o-", label="Train AUC", color="#3498db")
ax1.plot(B_values, bag_test_aucs,  "s-", label="Test AUC",  color="#e74c3c")
ax2.plot(B_values, bag_times, "^--", label="Temps (s)", color="#95a5a6")
ax1.set_xlabel("Nombre d'arbres B")
ax1.set_ylabel("AUC-ROC")
ax2.set_ylabel("Temps d'entraînement (s)")
ax1.set_title("Bagging – Performance & Temps vs B", fontweight="bold")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/12_bagging_vs_B.png", dpi=150)
plt.show()
print(f"   ✔ Figure sauvegardée : {FIG_DIR}/12_bagging_vs_B.png")

# ── 2. Random Forest avec OOB ─────────────────────────────────
print("\n▶ 2. Random Forest (p = sqrt) + OOB score")

# p = sqrt(n_features) → valeur par défaut Random Forest
rf_oob = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    oob_score=True,            # ← active le score Out-Of-Bag
    random_state=42,
    n_jobs=-1
)
rf_oob.fit(X_train, y_train)

oob_score  = rf_oob.oob_score_
test_auc   = roc_auc_score(y_test, rf_oob.predict_proba(X_test)[:, 1])
test_f1    = f1_score(y_test, rf_oob.predict(X_test))

print(f"   OOB Score (accuracy) : {oob_score:.4f}")
print(f"   Test AUC-ROC         : {test_auc:.4f}")
print(f"   Test F1              : {test_f1:.4f}")
print("""
  OOB vs Test AUC :
  Le score OOB est calculé sur les instances non utilisées par chaque
  arbre → estimation fiable sans jeu de validation séparé.
  Il est généralement proche du score test mais légèrement pessimiste.
""")

# ── 3. GridSearchCV sur p (max_features) ─────────────────────
print("▶ 3. Validation croisée sur p (max_features) – RF Optimisé")

param_grid_rf = {
    "max_features":     [0.1, 0.2, 0.3, "sqrt", "log2", 1.0],
    "min_samples_leaf": [1, 5, 10],
}

gs_rf = GridSearchCV(
    RandomForestClassifier(n_estimators=200, oob_score=False,
                           random_state=42, n_jobs=-1),
    param_grid=param_grid_rf,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)
gs_rf.fit(X_train, y_train)

print(f"\n   Meilleurs paramètres : {gs_rf.best_params_}")
print(f"   Meilleur CV AUC      : {gs_rf.best_score_:.4f}")

best_rf = gs_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
y_prob_rf = best_rf.predict_proba(X_test)[:, 1]

auc_rf = roc_auc_score(y_test, y_prob_rf)
f1_rf  = f1_score(y_test, y_pred_rf)

print(f"\n   Évaluation sur Test Set :")
print(f"   Test AUC-ROC : {auc_rf:.4f}")
print(f"   Test F1      : {f1_rf:.4f}")
print(f"\n{classification_report(y_test, y_pred_rf, target_names=['no','yes'])}")

# ── Résultats GridSearch – courbe ─────────────────────────────
res_rf = pd.DataFrame(gs_rf.cv_results_)

fig, ax = plt.subplots(figsize=(9, 5))
for leaf in [1, 5, 10]:
    sub = res_rf[res_rf["param_min_samples_leaf"] == leaf].sort_values("rank_test_score")
    ax.plot(
        sub["param_max_features"].astype(str),
        sub["mean_test_score"],
        "o-", label=f"min_samples_leaf={leaf}"
    )
ax.set_xlabel("max_features (p)")
ax.set_ylabel("CV AUC-ROC (moy)")
ax.set_title("GridSearch Random Forest – CV AUC vs max_features", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/13_rf_gridsearch.png", dpi=150)
plt.show()
print(f"   ✔ Figure sauvegardée : {FIG_DIR}/13_rf_gridsearch.png")

# Sauvegarde
joblib.dump(best_rf, f"{MODEL_DIR}/best_rf.pkl")
print(f"   ✔ Modèle sauvegardé : {MODEL_DIR}/best_rf.pkl")

print("\n✅  Script 1f terminé. Passez à 1g_boosting.py")
