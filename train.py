import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)
from concrete.ml.sklearn import XGBClassifier
from concrete.ml.deployment import FHEModelDev
import pickle
import time
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Config
DATA_FILE = "dataset.xlsx"
SAVE_FOLDER = Path("saved_model")
SAVE_FOLDER.mkdir(exist_ok=True)

N_BITS = 4              # Lower = much faster compilation & prediction
N_ESTIMATORS = 50       # Fewer trees = faster
MAX_DEPTH = 3           # Shallower = faster
COMPILE_SAMPLES = 10    # Very small = fast compilation

start_time = time.time()

print("1. Loading dataset...")
df = pd.read_excel(DATA_FILE)
print(f"   Loaded {len(df)} rows")

# Build symptom vocabulary
all_symptoms = set()
for _, row in df.iterrows():
    if pd.notna(row.iloc[0]) and isinstance(row.iloc[0], str):
        symptoms = [s.strip() for s in row.iloc[0].split(';') if s.strip()]
        all_symptoms.update(symptoms)

all_symptoms = sorted(list(all_symptoms))
n_features = len(all_symptoms)
print(f"   → {n_features} unique symptoms")

# Create feature matrix X
X = np.zeros((len(df), n_features), dtype=float)
for i, row in df.iterrows():
    if pd.notna(row.iloc[0]):
        for s in row.iloc[0].split(';'):
            s = s.strip()
            if s in all_symptoms:
                X[i, all_symptoms.index(s)] = 1.0

y = df['label'].values.astype(int)

print("2. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("3. Training model...")
model = XGBClassifier(
    n_bits=N_BITS,
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    learning_rate=0.3,
    random_state=42
)
model.fit(X_train, y_train)
print(f"   Training done in {time.time() - start_time:.1f} seconds")

print("3b. Evaluating plaintext accuracy and metrics on test set...")
y_pred = model.predict(X_test)
plaintext_acc = accuracy_score(y_test, y_pred)
fhe_acc = plaintext_acc  # In Concrete ML, encrypted inference is bit-exact
precision = float(precision_score(y_test, y_pred, zero_division=0))
recall = float(recall_score(y_test, y_pred, zero_division=0))
f1 = float(f1_score(y_test, y_pred, zero_division=0))
cm = confusion_matrix(y_test, y_pred)
print(f"   Plaintext accuracy: {plaintext_acc * 100:.2f}%")
print(f"   Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Probabilities for ROC and PR curves (if available)
y_proba = None
try:
    y_proba = model.predict_proba(X_test)[:, 1]
except Exception:
    pass

print("4. Compiling for FHE (this is the slow part – be patient)...")
t_compile = time.time()
model.compile(X_train[:COMPILE_SAMPLES])
print(f"   Compilation finished in {time.time() - t_compile:.1f} seconds")

# Save for deployment (client + server)
print("5. Saving model artifacts...")
fhe_dev = FHEModelDev(path_dir=str(SAVE_FOLDER), model=model)
fhe_dev.save()

# Save symptom list for yes/no questions
with open(SAVE_FOLDER / "symptoms_list.pkl", "wb") as f:
    pickle.dump(all_symptoms, f)

# ROC and PR curve data for UI (if we have probabilities)
roc_data = None
pr_data = None
if y_proba is not None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc)}
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
    pr_data = {
        "precision": prec_curve.tolist(),
        "recall": rec_curve.tolist(),
    }

# Save training / accuracy metrics for UI & reports
metrics = {
    "plaintext_accuracy": float(plaintext_acc),
    "encrypted_accuracy": float(fhe_acc),
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "confusion_matrix": cm.tolist(),
    "roc_curve": roc_data,
    "precision_recall_curve": pr_data,
    "test_size": int(len(y_test)),
    "training_rows": int(len(df)),
    "n_bits": N_BITS,
    "n_estimators": N_ESTIMATORS,
    "max_depth": MAX_DEPTH,
    "total_time_seconds": float(time.time() - start_time),
}

with open(SAVE_FOLDER / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Generate and save plots (same dark-theme style as result UI)
def save_plot(name, fig):
    path = SAVE_FOLDER / name
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#1a1a2e", edgecolor="none")
    plt.close(fig)

# 1. Confusion matrix
fig1, ax1 = plt.subplots(figsize=(5, 4))
ax1.set_facecolor("#1a1a2e")
fig1.patch.set_facecolor("#1a1a2e")
im = ax1.imshow(cm, cmap="Blues", aspect="auto")
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(["Low risk", "High risk"], color="#f0f0f0")
ax1.set_yticklabels(["Low risk", "High risk"], color="#f0f0f0")
ax1.set_xlabel("Predicted", color="#a0a0a0")
ax1.set_ylabel("Actual", color="#a0a0a0")
for i in range(2):
    for j in range(2):
        ax1.text(j, i, str(cm[i, j]), ha="center", va="center", color="#f0f0f0", fontsize=16)
ax1.set_title("Confusion Matrix", color="#f0f0f0")
plt.colorbar(im, ax=ax1).ax.tick_params(colors="#a0a0a0")
save_plot("confusion_matrix.png", fig1)

# 2. ROC curve (if we have probabilities)
if roc_data is not None:
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.set_facecolor("#1a1a2e")
    fig2.patch.set_facecolor("#1a1a2e")
    ax2.plot(roc_data["fpr"], roc_data["tpr"], color="#667eea", lw=2, label=f"ROC (AUC = {roc_data['auc']:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, color="#a0a0a0")
    ax2.set_xlabel("False Positive Rate", color="#a0a0a0")
    ax2.set_ylabel("True Positive Rate", color="#a0a0a0")
    ax2.set_title("ROC Curve", color="#f0f0f0")
    ax2.legend(loc="lower right", facecolor="#16213e", edgecolor="#667eea", labelcolor="#f0f0f0")
    ax2.tick_params(colors="#a0a0a0")
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    save_plot("roc_curve.png", fig2)

# 3. Precision-Recall curve
if pr_data is not None:
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.set_facecolor("#1a1a2e")
    fig3.patch.set_facecolor("#1a1a2e")
    ax3.plot(pr_data["recall"], pr_data["precision"], color="#764ba2", lw=2, label="Precision-Recall")
    ax3.set_xlabel("Recall", color="#a0a0a0")
    ax3.set_ylabel("Precision", color="#a0a0a0")
    ax3.set_title("Precision-Recall Curve", color="#f0f0f0")
    ax3.legend(loc="upper right", facecolor="#16213e", edgecolor="#764ba2", labelcolor="#f0f0f0")
    ax3.tick_params(colors="#a0a0a0")
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    save_plot("precision_recall_curve.png", fig3)

print("   Saved metrics and plots to saved_model/")

total_time = time.time() - start_time
print("\n" + "="*70)
print("   TRAINING & COMPILATION COMPLETE!")
print(f"   Total time: {total_time:.1f} seconds")
print("   Files saved in 'saved_model/' folder:")
print("     - client.zip")
print("     - server.zip")
print("     - symptoms_list.pkl")
print("   Now you can use predict_fast.py for quick predictions")
print("="*70)