# =============================================================
#  PARTIE 1c – Modèles simples & méthodes d'ensemble
#  TP Deep Learning – ENSY DIPES2 2025-2026
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import time

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier, BaggingClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier)

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score, ConfusionMatrixDisplay)

MODEL_DIR = "outputs/models"
FIG_DIR   = "outputs/figures"
RPT_DIR   = "outputs/reports"
os.makedirs(RPT_DIR, exist_ok=True)

# ── Chargement des données prétraitées ────────────────────────
print("=" * 65)
print("  PARTIE 1c – ENTRAÎNEMENT DES MODÈLES")
print("=" * 65)

# Données scalées pour SVM, KNN, LR, ANN
X_train_sc, X_test_sc, y_train, y_test = joblib.load(
    f"{MODEL_DIR}/data_scaled_split.pkl")
# Données brutes pour DT, RF, Bagging, GB (non nécessaire de scaler)
X_train_raw, X_test_raw, _, _ = joblib.load(
    f"{MODEL_DIR}/data_raw_split.pkl")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Définition des modèles ────────────────────────────────────
models_scaled = {
    "SVM (RBF)":          SVC(probability=True, random_state=42),
    "KNN (k=5)":          KNeighborsClassifier(n_neighbors=5),
    "Logistic Reg.":      LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes":        GaussianNB(),
    "MLP (ANN)":          MLPClassifier(hidden_layer_sizes=(64, 32),
                                        max_iter=300, random_state=42),
}

models_raw = {
    "Decision Tree":      DecisionTreeClassifier(random_state=42),
    "Random Forest":      RandomForestClassifier(n_estimators=100, random_state=42),
    "Bagging (DT)":       BaggingClassifier(estimator=DecisionTreeClassifier(),
                                            n_estimators=50, random_state=42),
    "Gradient Boosting":  GradientBoostingClassifier(n_estimators=100, random_state=42),
    "AdaBoost":           AdaBoostClassifier(n_estimators=100, random_state=42),
}

results = []

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    """Entraîne, évalue en CV et sur le test set."""
    print(f"\n  ▶ {name} ...")
    t0 = time.time()

    # Validation croisée – AUC-ROC
    scores_auc = cross_val_score(model, X_tr, y_tr, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
    scores_f1  = cross_val_score(model, X_tr, y_tr, cv=cv,
                                 scoring="f1", n_jobs=-1)

    # Entraînement final sur tout le train
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    # AUC test
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)[:, 1]
    else:
        y_prob = model.decision_function(X_te)
    auc_test = roc_auc_score(y_te, y_prob)
    f1_test  = f1_score(y_te, y_pred)

    elapsed = time.time() - t0
    print(f"     CV AUC-ROC : {scores_auc.mean():.4f} ± {scores_auc.std():.4f}")
    print(f"     CV F1      : {scores_f1.mean():.4f} ± {scores_f1.std():.4f}")
    print(f"     Test AUC   : {auc_test:.4f}  |  Test F1 : {f1_test:.4f}")
    print(f"     Temps      : {elapsed:.1f}s")

    return {
        "Modèle":        name,
        "CV AUC (moy)":  round(scores_auc.mean(), 4),
        "CV AUC (std)":  round(scores_auc.std(), 4),
        "CV F1 (moy)":   round(scores_f1.mean(), 4),
        "CV F1 (std)":   round(scores_f1.std(), 4),
        "Test AUC":      round(auc_test, 4),
        "Test F1":       round(f1_test, 4),
        "Temps (s)":     round(elapsed, 1),
    }

# ── Évaluation modèles sur données scalées ────────────────────
print("\n── Modèles sur données standardisées ──")
for name, model in models_scaled.items():
    res = evaluate_model(name, model, X_train_sc, X_test_sc, y_train, y_test)
    results.append(res)
    joblib.dump(model, f"{MODEL_DIR}/{name.replace(' ','_').replace('(','').replace(')','').replace('.','')}.pkl")

# ── Évaluation modèles sur données brutes ─────────────────────
print("\n── Modèles sur données brutes (arbres) ──")
for name, model in models_raw.items():
    res = evaluate_model(name, model, X_train_raw, X_test_raw, y_train, y_test)
    results.append(res)
    joblib.dump(model, f"{MODEL_DIR}/{name.replace(' ','_').replace('(','').replace(')','')}.pkl")

# ── Tableau récapitulatif ─────────────────────────────────────
df_res = pd.DataFrame(results).sort_values("Test AUC", ascending=False)
print("\n" + "=" * 65)
print("  TABLEAU RÉCAPITULATIF DES PERFORMANCES")
print("=" * 65)
print(df_res.to_string(index=False))
df_res.to_csv(f"{RPT_DIR}/comparaison_modeles.csv", index=False)
print(f"\n  ✔ Tableau sauvegardé : {RPT_DIR}/comparaison_modeles.csv")

# ── Graphique comparatif ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

df_sorted = df_res.sort_values("Test AUC")
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(df_sorted)))

# AUC
bars = axes[0].barh(df_sorted["Modèle"], df_sorted["Test AUC"],
                    color=colors, edgecolor="black")
axes[0].set_xlabel("AUC-ROC (Test)")
axes[0].set_title("Comparaison – AUC-ROC sur Test Set", fontweight="bold")
axes[0].axvline(0.5, color="red", linestyle="--", label="Aléatoire")
axes[0].set_xlim(0.45, 1.0)
for bar, val in zip(bars, df_sorted["Test AUC"]):
    axes[0].text(val + 0.003, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=8)

# F1
bars2 = axes[1].barh(df_sorted["Modèle"], df_sorted["Test F1"],
                     color=colors, edgecolor="black")
axes[1].set_xlabel("F1-score (Test)")
axes[1].set_title("Comparaison – F1-score sur Test Set", fontweight="bold")
axes[1].set_xlim(0.0, 0.85)
for bar, val in zip(bars2, df_sorted["Test F1"]):
    axes[1].text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=8)

plt.suptitle("Comparaison des modèles – Bank Telemarketing",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_comparaison_modeles.png", dpi=150)
plt.show()
print(f"  ✔ Figure sauvegardée : {FIG_DIR}/07_comparaison_modeles.png")

# ── Meilleur modèle – matrice de confusion ────────────────────
best_name = df_res.iloc[0]["Modèle"]
print(f"\n  ▶ Meilleur modèle : {best_name}")

# Recharger et prédire
best_model_key = best_name.replace(' ','_').replace('(','').replace(')','').replace('.','')
best_model = joblib.load(f"{MODEL_DIR}/{best_model_key}.pkl")

X_te = X_test_sc if best_name in models_scaled else X_test_raw
y_pred_best = best_model.predict(X_te)

print(f"\n  Classification Report – {best_name} :")
print(classification_report(y_test, y_pred_best, target_names=["no", "yes"]))

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_best,
                                        display_labels=["no", "yes"],
                                        colorbar=False, ax=ax,
                                        cmap="Blues")
ax.set_title(f"Matrice de confusion – {best_name}", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/08_confusion_best_model.png", dpi=150)
plt.show()
print(f"  ✔ Figure sauvegardée : {FIG_DIR}/08_confusion_best_model.png")

print("\n✅  Script 1d terminé. Passez à 1e_validation_croisee.py")
