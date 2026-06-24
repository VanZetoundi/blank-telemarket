# =============================================================
#  PARTIE 3 – Apprentissage Profond : Fashion MNIST
#  TP Deep Learning – ENSY DIPES2 2025-2026
# =============================================================

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, Flatten, Conv2D,
                                     MaxPooling2D, AveragePooling2D,
                                     BatchNormalization, GlobalAveragePooling2D,
                                     Input, Add, Activation, ZeroPadding2D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical

FIG_DIR   = "outputs/figures"
MODEL_DIR = "outputs/models"
RPT_DIR   = "outputs/reports"
for d in [FIG_DIR, MODEL_DIR, RPT_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 65)
print("  PARTIE 3 – DEEP LEARNING : Fashion MNIST")
print(f"  TensorFlow : {tf.__version__}")
print("=" * 65)

# =============================================================
#  CHARGEMENT & PRÉPARATION DES DONNÉES
# =============================================================
print("\n▶ Chargement Fashion MNIST...")
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

CLASS_NAMES = ["T-shirt", "Pantalon", "Pull", "Robe", "Manteau",
               "Sandale", "Chemise", "Sneaker", "Sac", "Bottine"]

print(f"  Train : {X_train.shape}  |  Test : {X_test.shape}")
print(f"  Classes : {CLASS_NAMES}")

# Normalisation [0, 255] → [0, 1]
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32")  / 255.0

# Reshape pour CNN : (N, 28, 28) → (N, 28, 28, 1)
X_train_cnn = X_train[..., np.newaxis]
X_test_cnn  = X_test[..., np.newaxis]

# Flatten pour MLP : (N, 28, 28) → (N, 784)
X_train_flat = X_train.reshape(-1, 784)
X_test_flat  = X_test.reshape(-1, 784)

# One-hot pour categorical_crossentropy
y_train_oh = to_categorical(y_train, 10)
y_test_oh  = to_categorical(y_test,  10)

# ── Visualisation d'exemples ──────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(X_train[i], cmap="gray")
    ax.set_title(CLASS_NAMES[y_train[i]], fontsize=9)
    ax.axis("off")
plt.suptitle("Exemples – Fashion MNIST", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/21_fashion_exemples.png", dpi=150)
plt.show()
print(f"  ✔ {FIG_DIR}/21_fashion_exemples.png")

# Callbacks communs
def get_callbacks(name):
    return [
        EarlyStopping(patience=10, restore_best_weights=True,
                      monitor="val_accuracy", mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1)
    ]

results = []

def evaluer_modele(nom, model, X_te, y_te, history, t_train):
    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    results.append({"Modèle": nom, "Accuracy": round(acc,4),
                    "Loss": round(loss,4), "Temps(s)": round(t_train,1)})
    print(f"  {nom:<25} → Accuracy={acc:.4f}  Loss={loss:.4f}  ({t_train:.0f}s)")
    return acc, history

# =============================================================
#  MODÈLE 0 – MLP de base (référence)
# =============================================================
print("\n" + "─"*65)
print("  MODÈLE 0 – MLP de base (référence)")
print("─"*65)

mlp = Sequential([
    Dense(512, activation="relu", input_shape=(784,)),
    Dropout(0.3),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(10, activation="softmax")
], name="MLP_Base")

mlp.compile(optimizer=Adam(0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"])
mlp.summary()

t0 = time.time()
h_mlp = mlp.fit(X_train_flat, y_train_oh,
                validation_split=0.1,
                epochs=30, batch_size=256,
                callbacks=get_callbacks("mlp"), verbose=1)
t_mlp = time.time() - t0
evaluer_modele("MLP Base", mlp, X_test_flat, y_test_oh, h_mlp, t_mlp)

# =============================================================
#  MODÈLE 1 – LeNet-5
# =============================================================
print("\n" + "─"*65)
print("  MODÈLE 1 – LeNet-5")
print("─"*65)

lenet = Sequential([
    Conv2D(6, (5,5), activation="tanh", padding="same",
           input_shape=(28,28,1)),
    AveragePooling2D((2,2)),
    Conv2D(16, (5,5), activation="tanh"),
    AveragePooling2D((2,2)),
    Flatten(),
    Dense(120, activation="tanh"),
    Dense(84,  activation="tanh"),
    Dense(10,  activation="softmax")
], name="LeNet5")

lenet.compile(optimizer=Adam(0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
lenet.summary()

t0 = time.time()
h_lenet = lenet.fit(X_train_cnn, y_train_oh,
                    validation_split=0.1,
                    epochs=30, batch_size=256,
                    callbacks=get_callbacks("lenet"), verbose=1)
t_lenet = time.time() - t0
evaluer_modele("LeNet-5", lenet, X_test_cnn, y_test_oh, h_lenet, t_lenet)

# =============================================================
#  MODÈLE 2 – VGGNet simplifié (VGG-like)
# =============================================================
print("\n" + "─"*65)
print("  MODÈLE 2 – VGGNet simplifié")
print("─"*65)

vgg = Sequential([
    # Bloc 1
    Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(28,28,1)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    # Bloc 2
    Conv2D(64, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(64, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    # Classificateur
    Flatten(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation="softmax")
], name="VGGNet_Simplifie")

vgg.compile(optimizer=Adam(0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"])
vgg.summary()

t0 = time.time()
h_vgg = vgg.fit(X_train_cnn, y_train_oh,
                validation_split=0.1,
                epochs=30, batch_size=256,
                callbacks=get_callbacks("vgg"), verbose=1)
t_vgg = time.time() - t0
evaluer_modele("VGGNet Simplifié", vgg, X_test_cnn, y_test_oh, h_vgg, t_vgg)

# =============================================================
#  MODÈLE 3 – ResNet simplifié (avec Skip Connections)
# =============================================================
print("\n" + "─"*65)
print("  MODÈLE 3 – ResNet simplifié (Skip Connections)")
print("─"*65)

def residual_block(x, filters):
    shortcut = x
    x = Conv2D(filters, (3,3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    # Adapter le shortcut si nécessaire
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1,1), padding="same")(shortcut)
    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x

inputs = Input(shape=(28, 28, 1))
x = Conv2D(32, (3,3), padding="same", activation="relu")(inputs)
x = BatchNormalization()(x)
x = residual_block(x, 32)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.25)(x)
x = residual_block(x, 64)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.25)(x)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(10, activation="softmax")(x)

resnet = Model(inputs, outputs, name="ResNet_Simplifie")
resnet.compile(optimizer=Adam(0.001),
               loss="categorical_crossentropy",
               metrics=["accuracy"])
resnet.summary()

t0 = time.time()
h_resnet = resnet.fit(X_train_cnn, y_train_oh,
                      validation_split=0.1,
                      epochs=30, batch_size=256,
                      callbacks=get_callbacks("resnet"), verbose=1)
t_resnet = time.time() - t0
evaluer_modele("ResNet Simplifié", resnet, X_test_cnn, y_test_oh, h_resnet, t_resnet)

# =============================================================
#  GRAPHIQUES COMPARATIFS
# =============================================================

# ── Courbes d'apprentissage ───────────────────────────────────
all_histories = [
    (h_mlp,    "MLP Base",         X_test_flat),
    (h_lenet,  "LeNet-5",          X_test_cnn),
    (h_vgg,    "VGGNet Simplifié", X_test_cnn),
    (h_resnet, "ResNet Simplifié", X_test_cnn),
]

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for col, (hist, nom, _) in enumerate(all_histories):
    axes[0, col].plot(hist.history["loss"],     label="Train", color="#3498db")
    axes[0, col].plot(hist.history["val_loss"], label="Val",   color="#e74c3c")
    axes[0, col].set_title(f"{nom}\nLoss", fontweight="bold", fontsize=9)
    axes[0, col].legend(fontsize=8); axes[0, col].grid(alpha=0.3)

    axes[1, col].plot(hist.history["accuracy"],     label="Train", color="#2ecc71")
    axes[1, col].plot(hist.history["val_accuracy"], label="Val",   color="#e67e22")
    axes[1, col].set_title(f"{nom}\nAccuracy", fontweight="bold", fontsize=9)
    axes[1, col].legend(fontsize=8); axes[1, col].grid(alpha=0.3)

plt.suptitle("Courbes d'apprentissage – Fashion MNIST",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/22_fashion_learning_curves.png", dpi=130)
plt.show()
print(f"  ✔ {FIG_DIR}/22_fashion_learning_curves.png")

# ── Tableau comparatif ────────────────────────────────────────
df_results = pd.DataFrame(results)
print("\n" + "=" * 65)
print("  TABLEAU COMPARATIF – Fashion MNIST")
print("=" * 65)
print(df_results.sort_values("Accuracy", ascending=False).to_string(index=False))
df_results.to_csv(f"{RPT_DIR}/comparaison_fashion_mnist.csv", index=False)

# ── Bar chart comparatif ──────────────────────────────────────
df_sorted = df_results.sort_values("Accuracy")
fig, ax = plt.subplots(figsize=(9, 5))
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_sorted)))
bars = ax.barh(df_sorted["Modèle"], df_sorted["Accuracy"],
               color=colors, edgecolor="black")
for bar, val in zip(bars, df_sorted["Accuracy"]):
    ax.text(val - 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", ha="right",
            color="white", fontweight="bold", fontsize=10)
ax.set_xlabel("Accuracy (Test)")
ax.set_title("Comparaison des architectures – Fashion MNIST", fontweight="bold")
ax.set_xlim(0.5, 1.0)
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/23_fashion_comparaison.png", dpi=150)
plt.show()
print(f"  ✔ {FIG_DIR}/23_fashion_comparaison.png")

# ── Prédictions visuelles – meilleur modèle ───────────────────
best_row = df_results.loc[df_results["Accuracy"].idxmax()]
best_nom = best_row["Modèle"]
model_map = {
    "MLP Base":         (mlp,    X_test_flat),
    "LeNet-5":          (lenet,  X_test_cnn),
    "VGGNet Simplifié": (vgg,    X_test_cnn),
    "ResNet Simplifié": (resnet, X_test_cnn),
}
best_model_fashion, X_best = model_map[best_nom]
y_prob_best = best_model_fashion.predict(X_best[:16], verbose=0)
y_pred_best = np.argmax(y_prob_best, axis=1)

fig, axes = plt.subplots(2, 8, figsize=(16, 5))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(X_test[i], cmap="gray")
    color = "green" if y_pred_best[i] == y_test[i] else "red"
    ax.set_title(f"Pred: {CLASS_NAMES[y_pred_best[i]]}\nVrai: {CLASS_NAMES[y_test[i]]}",
                 fontsize=7, color=color)
    ax.axis("off")
plt.suptitle(f"Prédictions – {best_nom} (vert=correct, rouge=erreur)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/24_fashion_predictions.png", dpi=150)
plt.show()
print(f"  ✔ {FIG_DIR}/24_fashion_predictions.png")

# =============================================================
#  SAUVEGARDE DU MEILLEUR MODÈLE → bank-tel.pkl
# =============================================================
best_model_fashion.save(f"{MODEL_DIR}/best_fashion.keras")

import joblib
joblib.dump({
    "model_name": best_nom,
    "model_path": f"{MODEL_DIR}/best_fashion.keras",
    "accuracy":   float(best_row["Accuracy"]),
    "classes":    CLASS_NAMES,
}, f"{MODEL_DIR}/bank-tel.pkl")

print(f"\n  ✔ Meilleur modèle : {best_nom}  (Accuracy={best_row['Accuracy']:.4f})")
print(f"  ✔ Sauvegardé      : {MODEL_DIR}/bank-tel.pkl")
print(f"  ✔ Keras format    : {MODEL_DIR}/best_fashion.keras")

print("\n✅  Partie 3 terminée ! Tout le TP est complet.")
print(f"    Figures générées : {FIG_DIR}/")
print(f"    Modèles sauvés   : {MODEL_DIR}/")
print(f"    Rapports CSV     : {RPT_DIR}/")
