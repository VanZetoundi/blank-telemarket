# =============================================================
#  PARTIE 1d – Validation croisée & Arbre de décision (CART)
#  TP Deep Learning – ENSY DIPES2 2025-2026
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score)
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             classification_report)
from sklearn.dummy import DummyClassifier

MODEL_DIR = "outputs/models"
FIG_DIR   = "outputs/figures"
RPT_DIR   = "outputs/reports"

X_train, X_test, y_train, y_test = joblib.load(f"{MODEL_DIR}/data_raw_split.pkl")
feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 65)
print("  PARTIE 1d – VALIDATION CROISÉE & CART")
print("=" * 65)

# ── 1. Meilleur classifieur constant (baseline) ───────────────
print("\n▶ 1. Classifieur constant (baseline)")

for strategy in ["most_frequent", "stratified", "prior"]:
    dummy = DummyClassifier(strategy=strategy, random_state=42)
    dummy.fit(X_train, y_train)
    y_pred_d = dummy.predict(X_test)
    acc = accuracy_score(y_test, y_pred_d)
    f1  = f1_score(y_test, y_pred_d, zero_division=0)
    print(f"   strategy='{strategy}' → Accuracy={acc:.4f}  F1={f1:.4f}")

print("""
  → Le meilleur classifieur constant ('most_frequent') prédit
    toujours 'no' et atteint ~89% d'accuracy MAIS F1=0.
  → Ce sera notre référentiel minimum à battre.
""")

# ── 2. Arbre de décision simple ───────────────────────────────
print("▶ 2. Arbre de décision (max_depth=4)")

dt_simple = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_simple.fit(X_train, y_train)
y_pred_s = dt_simple.predict(X_test)
auc_s = roc_auc_score(y_test, dt_simple.predict_proba(X_test)[:, 1])
f1_s  = f1_score(y_test, y_pred_s)
print(f"   Test AUC-ROC : {auc_s:.4f}  |  Test F1 : {f1_s:.4f}")
print(f"   Nombre de nœuds : {dt_simple.tree_.node_count}")

# Visualisation de l'arbre
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(dt_simple, feature_names=feature_names,
          class_names=["no", "yes"], filled=True,
          fontsize=7, rounded=True, ax=ax)
ax.set_title("Arbre de décision (max_depth=4)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/09_arbre_decision_simple.png", dpi=130)
plt.show()
print(f"   ✔ Figure sauvegardée : {FIG_DIR}/09_arbre_decision_simple.png")

# ── 3. Influence de max_depth ─────────────────────────────────
print("\n▶ 3. Influence de max_depth sur l'erreur")

depths     = list(range(1, 21))
train_aucs = []
test_aucs  = []
cv_aucs    = []

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_aucs.append(roc_auc_score(y_train, dt.predict_proba(X_train)[:, 1]))
    test_aucs.append(roc_auc_score(y_test,  dt.predict_proba(X_test)[:, 1]))
    cv_score = cross_val_score(dt, X_train, y_train,
                               cv=cv, scoring="roc_auc", n_jobs=-1).mean()
    cv_aucs.append(cv_score)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(depths, train_aucs, "o-", label="Train AUC", color="#3498db")
ax.plot(depths, test_aucs,  "s-", label="Test AUC",  color="#e74c3c")
ax.plot(depths, cv_aucs,    "^--", label="CV AUC (5-fold)", color="#2ecc71", linewidth=2)
best_depth = depths[np.argmax(cv_aucs)]
ax.axvline(best_depth, color="purple", linestyle="--",
           label=f"Best depth (CV) = {best_depth}")
ax.set_xlabel("max_depth")
ax.set_ylabel("AUC-ROC")
ax.set_title("AUC-ROC vs max_depth – Arbre CART", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/10_depth_vs_auc.png", dpi=150)
plt.show()
print(f"   Meilleure profondeur (CV) : {best_depth}")
print(f"   ✔ Figure sauvegardée : {FIG_DIR}/10_depth_vs_auc.png")

# ── 4. GridSearchCV – CART ────────────────────────────────────
print("\n▶ 4. GridSearchCV – CART (CART optimisé)")

param_grid = {
    "max_depth":        [3, 5, 7, 10, 15, None],
    "criterion":        ["gini", "entropy"],
    "min_samples_leaf": [1, 5, 10, 20],
    "max_features":     [None, "sqrt", "log2"],
}

gs_cart = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)
gs_cart.fit(X_train, y_train)

print(f"\n   Meilleurs paramètres : {gs_cart.best_params_}")
print(f"   Meilleur CV AUC      : {gs_cart.best_score_:.4f}")

best_cart = gs_cart.best_estimator_
y_pred_best = best_cart.predict(X_test)
y_prob_best = best_cart.predict_proba(X_test)[:, 1]

auc_best = roc_auc_score(y_test, y_prob_best)
f1_best  = f1_score(y_test, y_pred_best)

print(f"\n   Évaluation sur Test Set :")
print(f"   Test AUC-ROC : {auc_best:.4f}")
print(f"   Test F1      : {f1_best:.4f}")
print(f"\n{classification_report(y_test, y_pred_best, target_names=['no','yes'])}")

# Sauvegarde
joblib.dump(best_cart, f"{MODEL_DIR}/best_cart.pkl")
print(f"   ✔ Modèle sauvegardé : {MODEL_DIR}/best_cart.pkl")

# Visualisation arbre optimisé
fig, ax = plt.subplots(figsize=(22, 9))
depth_viz = best_cart.get_depth() if best_cart.get_depth() <= 5 else 5
dt_viz = DecisionTreeClassifier(**{**gs_cart.best_params_,
                                    "max_depth": depth_viz,
                                    "random_state": 42})
dt_viz.fit(X_train, y_train)
plot_tree(dt_viz, feature_names=feature_names,
          class_names=["no", "yes"], filled=True,
          fontsize=6, rounded=True, ax=ax)
ax.set_title(f"CART Optimisé (visualisé à depth={depth_viz})",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/11_arbre_optimise.png", dpi=130)
plt.show()
print(f"   ✔ Figure sauvegardée : {FIG_DIR}/11_arbre_optimise.png")

print("\n✅  Script 1e terminé. Passez à 1f_bagging_random_forest.py")
