# =============================================================
#  PARTIE 2 – Réseau de Neurones Artificiel (ANN / Keras)
#  TP Deep Learning – ENSY DIPES2 2025-2026
#  Dataset : Bank Telemarketing (UCI)
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (classification_report, roc_auc_score,
                             f1_score, ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_class_weight

# ── Chemins ───────────────────────────────────────────────────
# Ce script est dans partie2/ mais les données viennent de partie1/
PART1_MODELS = "../partie1/outputs/models"
FIG_DIR      = "outputs/figures"
MODEL_DIR    = "outputs/models"
RPT_DIR      = "outputs/reports"
for d in [FIG_DIR, MODEL_DIR, RPT_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 65)
print("  PARTIE 2 – RÉSEAU DE NEURONES ARTIFICIEL (ANN/Keras)")
print(f"  TensorFlow : {tf.__version__}")
print("=" * 65)

# ── Chargement des données (prétraitées en Partie 1) ──────────
X_train, X_test, y_train, y_test = joblib.load(
    f"{PART1_MODELS}/data_scaled_split.pkl")
top10_features = joblib.load(f"{PART1_MODELS}/top10_features.pkl")

print(f"\n  Train : {X_train.shape}  |  Test : {X_test.shape}")
print(f"  Top 10 features sélectionnées : {top10_features}")

# ── Équilibrage des classes ───────────────────────────────────
weights = compute_class_weight("balanced",
                               classes=np.unique(y_train),
                               y=y_train)
class_weight = dict(zip(np.unique(y_train), weights))
print(f"\n  Poids des classes : {class_weight}")

# ── Conversion numpy ──────────────────────────────────────────
X_tr = X_train.values.astype("float32")
X_te = X_test.values.astype("float32")
y_tr = y_train.values.astype("float32")
y_te = y_test.values.astype("float32")
n_features = X_tr.shape[1]

# =============================================================
#  MODÈLE 1 – Réseau simple (1 couche cachée) avec Softmax
# =============================================================
print("\n" + "─" * 65)
print("  MODÈLE 1 – Réseau simple (Dense 64 → Softmax/Sigmoid)")
print("─" * 65)

# Pour classification binaire on utilise sigmoid en sortie.
# Pour voir Softmax : on peut faire une couche Dense(2, softmax)
# puis prendre la proba de la classe 1.
model1 = Sequential([
    Dense(64, activation="relu", input_shape=(n_features,), name="couche_1"),
    Dense(1,  activation="sigmoid", name="sortie")
], name="ANN_Simple")

model1.compile(optimizer=Adam(0.001),
               loss="binary_crossentropy",
               metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

print("\n▶ Architecture – model1.summary() :")
model1.summary()

h1 = model1.fit(X_tr, y_tr,
                validation_split=0.15,
                epochs=80,
                batch_size=256,
                class_weight=class_weight,
                callbacks=[EarlyStopping(patience=10,
                                         restore_best_weights=True,
                                         monitor="val_auc", mode="max")],
                verbose=1)

# =============================================================
#  MODÈLE 2 – Réseau profond (3 couches + Dropout + BN)
# =============================================================
print("\n" + "─" * 65)
print("  MODÈLE 2 – Réseau profond (3 couches cachées)")
print("─" * 65)

model2 = Sequential([
    Dense(128, activation="relu", input_shape=(n_features,), name="couche_1"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation="relu", name="couche_2"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation="relu", name="couche_3"),
    Dropout(0.2),

    Dense(1, activation="sigmoid", name="sortie")
], name="ANN_Profond")

model2.compile(optimizer=Adam(0.001),
               loss="binary_crossentropy",
               metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

print("\n▶ Architecture – model2.summary() :")
model2.summary()

h2 = model2.fit(X_tr, y_tr,
                validation_split=0.15,
                epochs=150,
                batch_size=256,
                class_weight=class_weight,
                callbacks=[
                    EarlyStopping(patience=15, restore_best_weights=True,
                                  monitor="val_auc", mode="max"),
                    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                     patience=5, min_lr=1e-6, verbose=1),
                    ModelCheckpoint(f"{MODEL_DIR}/best_ann.keras",
                                    monitor="val_auc", mode="max",
                                    save_best_only=True, verbose=0)
                ],
                verbose=1)

# =============================================================
#  MODÈLE 3 – Réseau sur Top 10 variables importantes
# =============================================================
print("\n" + "─" * 65)
print("  MODÈLE 3 – Réseau sur les 10 variables les plus importantes")
print("─" * 65)

top10_idx = [i for i, f in enumerate(X_train.columns)
             if f in top10_features]
X_tr_10 = X_tr[:, top10_idx]
X_te_10 = X_te[:, top10_idx]
print(f"  Features après sélection : {X_tr_10.shape[1]}")

model3 = Sequential([
    Dense(64, activation="relu", input_shape=(len(top10_idx),), name="couche_1"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation="relu", name="couche_2"),
    Dropout(0.2),
    Dense(16, activation="relu", name="couche_3"),
    Dense(1,  activation="sigmoid", name="sortie")
], name="ANN_Top10")

model3.compile(optimizer=Adam(0.001),
               loss="binary_crossentropy",
               metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

print("\n▶ Architecture – model3.summary() :")
model3.summary()

h3 = model3.fit(X_tr_10, y_tr,
                validation_split=0.15,
                epochs=150,
                batch_size=256,
                class_weight=class_weight,
                callbacks=[EarlyStopping(patience=15,
                                         restore_best_weights=True,
                                         monitor="val_auc", mode="max")],
                verbose=1)

# =============================================================
#  ÉVALUATION
# =============================================================
def evaluer(nom, model, X_eval, y_eval):
    y_prob = model.predict(X_eval, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_eval, y_prob)
    f1  = f1_score(y_eval, y_pred)
    print(f"\n  ── {nom} ──")
    print(f"  AUC-ROC : {auc:.4f}  |  F1 : {f1:.4f}")
    print(classification_report(y_eval, y_pred, target_names=["no","yes"]))
    return auc, f1, y_prob, y_pred

print("\n" + "=" * 65)
print("  ÉVALUATION SUR LE TEST SET")
print("=" * 65)

auc1, f1_1, prob1, pred1 = evaluer("Modèle 1 – Simple",  model1, X_te,    y_te)
auc2, f1_2, prob2, pred2 = evaluer("Modèle 2 – Profond", model2, X_te,    y_te)
auc3, f1_3, prob3, pred3 = evaluer("Modèle 3 – Top10",   model3, X_te_10, y_te)

# =============================================================
#  GRAPHIQUES
# =============================================================

# ── Courbes d'apprentissage ───────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for col, (hist, nom) in enumerate([
    (h1, "Modèle 1 Simple"),
    (h2, "Modèle 2 Profond"),
    (h3, "Modèle 3 Top10"),
]):
    axes[0, col].plot(hist.history["loss"],     label="Train", color="#3498db")
    axes[0, col].plot(hist.history["val_loss"], label="Val",   color="#e74c3c")
    axes[0, col].set_title(f"{nom}\nLoss (Binary Crossentropy)", fontweight="bold")
    axes[0, col].set_xlabel("Épochs"); axes[0, col].legend(); axes[0, col].grid(alpha=0.3)

    axes[1, col].plot(hist.history["auc"],     label="Train", color="#2ecc71")
    axes[1, col].plot(hist.history["val_auc"], label="Val",   color="#e67e22")
    axes[1, col].set_title(f"{nom}\nAUC-ROC", fontweight="bold")
    axes[1, col].set_xlabel("Épochs"); axes[1, col].legend(); axes[1, col].grid(alpha=0.3)

plt.suptitle("Courbes d'apprentissage – ANN Telemarketing",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/19_ann_learning_curves.png", dpi=150)
plt.show()
print(f"  ✔ {FIG_DIR}/19_ann_learning_curves.png")

# ── Matrice de confusion meilleur modèle ─────────────────────
aucs = [auc1, auc2, auc3]
best_idx = int(np.argmax(aucs))
best_info = [(model1, X_te, pred1, "ANN Simple"),
             (model2, X_te, pred2, "ANN Profond"),
             (model3, X_te_10, pred3, "ANN Top10")][best_idx]
_, _, best_pred, best_nom = best_info

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_te, best_pred, display_labels=["no", "yes"],
    colorbar=False, ax=ax, cmap="Blues")
ax.set_title(f"Matrice de confusion – {best_nom}", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/20_ann_confusion_best.png", dpi=150)
plt.show()
print(f"  ✔ {FIG_DIR}/20_ann_confusion_best.png")

# ── Comparaison ───────────────────────────────────────────────
df_res = pd.DataFrame({
    "Modèle":   ["ANN Simple", "ANN Profond", "ANN Top10"],
    "Test AUC": [round(auc1,4), round(auc2,4), round(auc3,4)],
    "Test F1":  [round(f1_1,4), round(f1_2,4), round(f1_3,4)],
})
print("\n▶ Comparaison des 3 modèles ANN :")
print(df_res.to_string(index=False))
df_res.to_csv(f"{RPT_DIR}/comparaison_ann.csv", index=False)

# =============================================================
#  SAUVEGARDE → telemarketing.pkl
# =============================================================
best_models = [model1, model2, model3]
best_model  = best_models[best_idx]
best_model.save(f"{MODEL_DIR}/telemarketing.keras")

joblib.dump({
    "model_name":     best_nom,
    "model_path":     f"{MODEL_DIR}/telemarketing.keras",
    "top10_features": top10_features,
    "top10_idx":      top10_idx,
    "auc":            max(aucs),
}, f"{MODEL_DIR}/telemarketing.pkl")

print(f"\n  ✔ Meilleur modèle : {best_nom}  (AUC={max(aucs):.4f})")
print(f"  ✔ Sauvegardé      : {MODEL_DIR}/telemarketing.pkl")
print(f"  ✔ Keras format    : {MODEL_DIR}/telemarketing.keras")

print("\n✅  Partie 2 terminée ! Passez à 2b_deep_learning_fashion.py")
