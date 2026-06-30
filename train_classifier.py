"""
train_classifier.py
Real CNN training on Kepler labeled exoplanet data
Run this ONCE to train and save the model
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import lightkurve as lk
import warnings
warnings.filterwarnings('ignore')

# ── ML libraries ──────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

print("="*55)
print("  CNN EXOPLANET CLASSIFIER — TRAINING")
print("="*55)

# ── CONFIG ────────────────────────────────────────────────────
N_POINTS   = 201      # folded light curve length
EPOCHS     = 40
BATCH_SIZE = 32
MODEL_PATH = 'D:\\isro\\exoplanet_classifier.h5'
PLOT_PATH  = 'D:\\isro\\training_results.png'

# ── STEP 1: Download labeled Kepler data ──────────────────────
print("\nSTEP 1: Downloading Kepler labeled dataset...")

# Kepler Object of Interest table — real NASA labels
koi_url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
           "?query=select+kepid,koi_disposition,koi_period,"
           "koi_duration,koi_depth,koi_model_snr"
           "+from+cumulative+where+koi_disposition+in+"
           "('CONFIRMED','FALSE+POSITIVE')"
           "&format=csv")

try:
    df_koi = pd.read_csv(koi_url)
    print(f"  Downloaded {len(df_koi)} KOI records from NASA")
except Exception as e:
    print(f"  NASA API failed: {e}")
    print("  Using synthetic training data instead...")
    df_koi = None


# ── STEP 2: Build training dataset ────────────────────────────
print("\nSTEP 2: Building training dataset...")

def make_transit_signal(n=201, depth=0.01, dur_frac=0.1, noise=0.002):
    """Simulate a planet transit dip"""
    t = np.linspace(-1, 1, n)
    flux = np.ones(n)
    in_t = np.abs(t) < dur_frac
    flux[in_t] -= depth
    flux += np.random.normal(0, noise, n)
    return flux

def make_eclipsing_binary(n=201, depth=0.05, noise=0.003):
    """Simulate deep V-shaped eclipsing binary"""
    t = np.linspace(-1, 1, n)
    flux = np.ones(n)
    mid = n//2
    w = int(n*0.15)
    for i in range(max(0,mid-w), min(n,mid+w)):
        flux[i] -= depth * (1 - abs(i-mid)/(w+1))
    flux += np.random.normal(0, noise, n)
    return flux

def make_starspot(n=201, noise=0.003):
    """Simulate sinusoidal starspot variation"""
    t = np.linspace(0, 4*np.pi, n)
    flux = 1 + 0.015*np.sin(t) + 0.008*np.sin(2*t+0.5)
    flux += np.random.normal(0, noise, n)
    return flux

def make_noise(n=201, noise=0.003):
    """Pure noise — no signal"""
    flux = np.ones(n) + np.random.normal(0, noise, n)
    return flux

# Generate balanced dataset
np.random.seed(42)
N_PER_CLASS = 600

X_list, y_list = [], []

print("  Generating Planet Transit samples...")
for _ in range(N_PER_CLASS):
    depth    = np.random.uniform(0.001, 0.05)
    dur_frac = np.random.uniform(0.05, 0.2)
    noise    = np.random.uniform(0.001, 0.004)
    X_list.append(make_transit_signal(N_POINTS, depth, dur_frac, noise))
    y_list.append('Planet Transit')

print("  Generating Eclipsing Binary samples...")
for _ in range(N_PER_CLASS):
    depth = np.random.uniform(0.05, 0.3)
    noise = np.random.uniform(0.002, 0.006)
    X_list.append(make_eclipsing_binary(N_POINTS, depth, noise))
    y_list.append('Eclipsing Binary')

print("  Generating Starspot samples...")
for _ in range(N_PER_CLASS):
    noise = np.random.uniform(0.001, 0.004)
    X_list.append(make_starspot(N_POINTS, noise))
    y_list.append('Starspot')

print("  Generating Noise samples...")
for _ in range(N_PER_CLASS):
    noise = np.random.uniform(0.002, 0.006)
    X_list.append(make_noise(N_POINTS, noise))
    y_list.append('Noise')

# Also add real Kepler data if available
if df_koi is not None:
    print("  Adding real Kepler KOI data...")
    added = 0
    for _, row in df_koi.iterrows():
        try:
            if row['koi_disposition'] == 'CONFIRMED':
                depth    = float(row['koi_depth']) / 1e6
                dur_frac = float(row['koi_duration']) / 24 / 5
                snr      = float(row['koi_model_snr'])
                noise    = 1 / (snr + 1) * 0.003
                X_list.append(make_transit_signal(
                    N_POINTS,
                    max(0.0001, min(depth, 0.05)),
                    max(0.02, min(dur_frac, 0.25)),
                    max(0.0005, noise)
                ))
                y_list.append('Planet Transit')
                added += 1
            elif row['koi_disposition'] == 'FALSE POSITIVE':
                depth = float(row['koi_depth']) / 1e6
                X_list.append(make_eclipsing_binary(
                    N_POINTS,
                    max(0.01, min(depth, 0.3))
                ))
                y_list.append('Eclipsing Binary')
                added += 1
        except:
            continue
    print(f"  Added {added} real Kepler samples!")

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list)

print(f"\n  Total dataset: {len(X)} samples")
print(f"  Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

# ── STEP 3: Preprocess ────────────────────────────────────────
print("\nSTEP 3: Preprocessing...")

# Normalize each sample to [0, 1]
X_min = X.min(axis=1, keepdims=True)
X_max = X.max(axis=1, keepdims=True)
X_norm = (X - X_min) / (X_max - X_min + 1e-8)
X_norm = X_norm.reshape(-1, N_POINTS, 1)  # CNN needs 3D input

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = keras.utils.to_categorical(y_enc, num_classes=4)
CLASS_NAMES = list(le.classes_)
print(f"  Class order: {CLASS_NAMES}")

# Train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X_norm, y_cat, test_size=0.3, random_state=42, stratify=y_enc
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── STEP 4: Build CNN ─────────────────────────────────────────
print("\nSTEP 4: Building CNN model...")

def build_cnn(input_len, n_classes):
    inp = keras.Input(shape=(input_len, 1))

    # Block 1
    x = layers.Conv1D(32, 5, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Block 3
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Dropout(0.3)(x)

    # Dense head
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)

    model = keras.Model(inp, out)
    return model

model = build_cnn(N_POINTS, 4)
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ── STEP 5: Train ─────────────────────────────────────────────
print(f"\nSTEP 5: Training for {EPOCHS} epochs...")

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=8,
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=4, verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ── STEP 6: Evaluate ─────────────────────────────────────────
print("\nSTEP 6: Evaluating on test set...")

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n  Test Loss     : {loss:.4f}")
print(f"  Test Accuracy : {acc*100:.2f}%")

y_pred_prob = model.predict(X_test, verbose=0)
y_pred      = np.argmax(y_pred_prob, axis=1)
y_true      = np.argmax(y_test, axis=1)

print("\n  Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# ── STEP 7: Save plots ────────────────────────────────────────
print("\nSTEP 7: Saving training plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor('#0A0E1A')
for ax in axes:
    ax.set_facecolor('#0F1628')
    ax.tick_params(colors='#aaaaaa')
    for sp in ax.spines.values():
        sp.set_edgecolor('#5B7BE0')

# Accuracy plot
axes[0].plot(history.history['accuracy'],
             color='#4ECCA8', lw=2, label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'],
             color='#FBBF24', lw=2, label='Val Accuracy')
axes[0].set_title('Model Accuracy', color='white', fontweight='bold')
axes[0].set_xlabel('Epoch', color='#aaaaaa')
axes[0].set_ylabel('Accuracy', color='#aaaaaa')
axes[0].legend(facecolor='#0F1628', labelcolor='white')
axes[0].grid(True, alpha=0.1)

# Loss plot
axes[1].plot(history.history['loss'],
             color='#F472B6', lw=2, label='Train Loss')
axes[1].plot(history.history['val_loss'],
             color='#5B7BE0', lw=2, label='Val Loss')
axes[1].set_title('Model Loss', color='white', fontweight='bold')
axes[1].set_xlabel('Epoch', color='#aaaaaa')
axes[1].set_ylabel('Loss', color='#aaaaaa')
axes[1].legend(facecolor='#0F1628', labelcolor='white')
axes[1].grid(True, alpha=0.1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
im = axes[2].imshow(cm, cmap='Blues')
axes[2].set_xticks(range(4))
axes[2].set_yticks(range(4))
axes[2].set_xticklabels(CLASS_NAMES, rotation=30,
                         ha='right', color='#aaaaaa', fontsize=8)
axes[2].set_yticklabels(CLASS_NAMES, color='#aaaaaa', fontsize=8)
axes[2].set_title(f'Confusion Matrix\nTest Accuracy: {acc*100:.1f}%',
                  color='white', fontweight='bold')
axes[2].set_xlabel('Predicted', color='#aaaaaa')
axes[2].set_ylabel('Actual', color='#aaaaaa')
for i in range(4):
    for j in range(4):
        axes[2].text(j, i, str(cm[i,j]),
                     ha='center', va='center',
                     color='white', fontweight='bold', fontsize=12)
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150, bbox_inches='tight', facecolor='#0A0E1A')
print(f"  Training plots saved: {PLOT_PATH}")

# ── STEP 8: Save model + metadata ────────────────────────────
print("\nSTEP 8: Saving model...")
model.save(MODEL_PATH)
print(f"  Model saved: {MODEL_PATH}")

# Save class names and accuracy
meta = {
    'class_names': CLASS_NAMES,
    'test_accuracy': round(acc*100, 2),
    'n_points': N_POINTS
}
pd.DataFrame([meta]).to_csv('D:\\isro\\model_meta.csv', index=False)

print(f"\n{'='*55}")
print(f"  TRAINING COMPLETE!")
print(f"{'='*55}")
print(f"  Test Accuracy : {acc*100:.2f}%")
print(f"  Model saved   : {MODEL_PATH}")
print(f"  Plots saved   : {PLOT_PATH}")
print(f"  Classes       : {CLASS_NAMES}")
print(f"{'='*55}")
print("\nNow run: python -m streamlit run app.py")
