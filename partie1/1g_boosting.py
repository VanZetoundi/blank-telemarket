# =============================================================
#  PARTIE 1e (suite) – Gradient Boosting & AdaBoost
#  TP Deep Learning – ENSY DIPES2 2025-2026
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.ensemble import (GradientBoostingClassifier,
                               AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, classification_report

MODEL_DIR = "outputs/models"
FIG_DIR   = "outputs/figures"
RPT_DIR   = "outputs/reports"

X_train, X_test, y_train, y_test = joblib.load(f"{MODEL_DIR}/data_raw_split.pkl")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 65)
print("  PARTIE 1e – GRADIENT BOOSTING")
print("=" * 65)

# ── a. Paramètres cruciaux du Gradient Boosting ───────────────
print("""
▶ a. Paramètres cruciaux de GradientBoostingClassifier :
   • n_estimators (B)       : nombre total d'arbres agrégés
   • learning_rate (λ)      : taux d'apprentissage (shrinkage)
   • max_depth (p)          : profondeur max de chaque arbre
   • subsample              : fraction d'instances par arbre
   • max_features           : fraction de features par split

   Correspondance avec AdaBoost :
   → learning_rate=1.0, max_depth=1, loss='exponential'
     reproduit le comportement d'AdaBoost.
""")

# ── b. Early Stopping – choix de B optimal ────────────────────
print("▶ b. Early Stopping pour choisir B")

gb_es = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    validation_fraction=0.1,    # 10% du train pour la validation interne
    n_iter_no_change=20,        # arrêt si pas d'amélioration sur 20 rounds
    tol=1e-4,
    random_state=42
)
gb_es.fit(X_train, y_train)

B_optimal = gb_es.n_estimators_
print(f"   B optimal (early stopping) : {B_optimal}")
print(f"   Test AUC : {roc_auc_score(y_test, gb_es.predict_proba(X_test)[:,1]):.4f}")

# Courbe de convergence
train_scores = []
test_scores  = []
for pred in gb_es.staged_predict_proba(X_train):
    train_scores.append(roc_auc_score(y_train, pred[:, 1]))
for pred in gb_es.staged_predict_proba(X_test):
    test_scores.append(roc_auc_score(y_test, pred[:, 1]))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train_scores, label="Train AUC", color="#3498db", alpha=0.8)
ax.plot(test_scores,  label="Test AUC",  color="#e74c3c", alpha=0.8)
ax.axvline(B_optimal - 1, color="purple", linestyle="--",
           label=f"Early stop : B={B_optimal}")
ax.set_xlabel("Nombre d'arbres (B)")
ax.set_ylabel("AUC-ROC")
ax.set_title("Gradient Boosting – Convergence (Early Stopping)", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/14_gb_early_stopping.png", dpi=150)
plt.show()
print(f"   ✔ Figure sauvegardée : {FIG_DIR}/14_gb_early_stopping.png")

# ── c. GridSearchCV – GB Optimisé ────────────────────────────
print("\n▶ c. GridSearchCV – Gradient Boosting Optimisé")

param_grid_gb = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth":     [3, 4, 5],
    "subsample":     [0.7, 0.9, 1.0],
}

gs_gb = GridSearchCV(
    GradientBoostingClassifier(n_estimators=B_optimal, random_state=42),
    param_grid=param_grid_gb,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)
gs_gb.fit(X_train, y_train)

print(f"\n   Meilleurs paramètres : {gs_gb.best_params_}")
print(f"   Meilleur CV AUC      : {gs_gb.best_score_:.4f}")

best_gb = gs_gb.best_estimator_
y_pred_gb = best_gb.predict(X_test)
y_prob_gb = best_gb.predict_proba(X_test)[:, 1]

auc_gb = roc_auc_score(y_test, y_prob_gb)
f1_gb  = f1_score(y_test, y_pred_gb)

print(f"\n   Évaluation sur Test Set :")
print(f"   Test AUC-ROC : {auc_gb:.4f}")
print(f"   Test F1      : {f1_gb:.4f}")
print(f"\n{classification_report(y_test, y_pred_gb, target_names=['no','yes'])}")

joblib.dump(best_gb, f"{MODEL_DIR}/best_gb.pkl")
print(f"   ✔ Modèle sauvegardé : {MODEL_DIR}/best_gb.pkl")

# ── Comparaison finale des 3 classifieurs optimisés ──────────
best_cart = joblib.load(f"{MODEL_DIR}/best_cart.pkl")
best_rf   = joblib.load(f"{MODEL_DIR}/best_rf.pkl")

y_prob_cart = best_cart.predict_proba(X_test)[:, 1]
y_prob_rf   = best_rf.predict_proba(X_test)[:, 1]

comparison = pd.DataFrame({
    "Modèle":    ["CART Optimisé", "Random Forest Optimisé", "Gradient Boosting Optimisé"],
    "Test AUC":  [
        round(roc_auc_score(y_test, y_prob_cart), 4),
        round(roc_auc_score(y_test, y_prob_rf),   4),
        round(auc_gb, 4),
    ],
    "Test F1":   [
        round(f1_score(y_test, best_cart.predict(X_test)), 4),
        round(f1_score(y_test, best_rf.predict(X_test)),   4),
        round(f1_gb, 4),
    ],
})

print("\n" + "=" * 65)
print("  COMPARAISON DES 3 CLASSIFIEURS OPTIMISÉS")
print("=" * 65)
print(comparison.to_string(index=False))
comparison.to_csv(f"{RPT_DIR}/comparaison_top3.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
colors = ["#3498db", "#2ecc71", "#e74c3c"]

for ax, metric in zip(axes, ["Test AUC", "Test F1"]):
    bars = ax.bar(comparison["Modèle"], comparison[metric],
                  color=colors, edgecolor="black")
    ax.set_title(f"Comparaison – {metric}", fontweight="bold")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, comparison[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

plt.suptitle("CART vs Random Forest vs Gradient Boosting (optimisés)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/15_comparaison_top3.png", dpi=150)
plt.show()
print(f"   ✔ Figure sauvegardée : {FIG_DIR}/15_comparaison_top3.png")

print("\n✅  Script 1g terminé. Passez à 1h_feature_importance.py")
