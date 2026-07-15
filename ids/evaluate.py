"""
Model Evaluation & Comparison Report
Generates detailed performance metrics, ROC curves, confusion matrix,
and a side-by-side comparison of RF vs DNN vs Ensemble.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    accuracy_score, f1_score
)
from sklearn.preprocessing import label_binarize
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
from ids.models.ensemble import EnsembleIDS, CLASS_NAMES

PROC_DIR    = Path(__file__).parent / "data" / "processed"
MODELS_DIR  = Path(__file__).parent / "models" / "saved"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: Path):
    """Plot and save a normalised confusion matrix."""
    cm = confusion_matrix(y_true, y_pred,
                          labels=list(range(len(CLASS_NAMES))),
                          normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_curves(models_proba: dict, y_test, save_path: Path):
    """Macro-average one-vs-rest ROC curve for each model."""
    y_bin = label_binarize(y_test, classes=list(range(len(CLASS_NAMES))))
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4f8ef7", "#f74f8e", "#4ff7a0"]
    for (name, proba), color in zip(models_proba.items(), colors):
        fpr_list, tpr_list = [], []
        for i in range(len(CLASS_NAMES)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            fpr_list.append(fpr); tpr_list.append(tpr)
        all_fpr = np.unique(np.concatenate(fpr_list))
        mean_tpr = np.zeros_like(all_fpr)
        for fpr, tpr in zip(fpr_list, tpr_list):
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= len(CLASS_NAMES)
        roc_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, color=color, lw=2,
                label=f"{name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – Macro Average (One-vs-Rest)", fontsize=13)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[EVAL] Loading test data ...")
    X_test = np.load(PROC_DIR / "X_test.npy")
    y_test = np.load(PROC_DIR / "y_test.npy")

    print("[EVAL] Loading models ...")
    import tensorflow as tf
    rf  = joblib.load(MODELS_DIR / "random_forest.pkl")
    dnn = tf.keras.models.load_model(str(MODELS_DIR / "dnn_model.keras"))
    ens = EnsembleIDS(rf_model=rf, dnn_model=dnn)

    # ── Probabilities ─────────────────────────────────────────────────────────
    rf_proba  = rf.predict_proba(X_test)
    dnn_proba = dnn.predict(X_test, batch_size=512, verbose=0)
    ens_proba = ens.predict_proba(X_test)

    rf_pred  = np.argmax(rf_proba,  axis=1)
    dnn_pred = np.argmax(dnn_proba, axis=1)
    ens_pred = np.argmax(ens_proba, axis=1)

    # ── Summary Table ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("═" * 60)
    header = f"{'Model':<16} {'Accuracy':>10} {'F1 (weighted)':>14}"
    print(header)
    print("-" * 42)
    for name, pred in [("Random Forest", rf_pred),
                        ("DNN (CPU)",     dnn_pred),
                        ("Ensemble",      ens_pred)]:
        acc = accuracy_score(y_test, pred)
        f1  = f1_score(y_test, pred, average="weighted")
        print(f"{name:<16} {acc*100:>9.2f}%  {f1*100:>13.2f}%")
    print("═" * 60)

    # ── Confusion Matrices ────────────────────────────────────────────────────
    print("\n[EVAL] Generating confusion matrices ...")
    plot_confusion_matrix(y_test, rf_pred,  "Random Forest – Confusion Matrix",
                          REPORTS_DIR / "cm_rf.png")
    plot_confusion_matrix(y_test, dnn_pred, "DNN – Confusion Matrix",
                          REPORTS_DIR / "cm_dnn.png")
    plot_confusion_matrix(y_test, ens_pred, "Ensemble – Confusion Matrix",
                          REPORTS_DIR / "cm_ensemble.png")

    # ── ROC Curves ───────────────────────────────────────────────────────────
    print("\n[EVAL] Generating ROC curves ...")
    plot_roc_curves(
        {"Random Forest": rf_proba, "DNN": dnn_proba, "Ensemble": ens_proba},
        y_test,
        REPORTS_DIR / "roc_curves.png"
    )

    print(f"\n✓ Evaluation complete. Reports saved in: {REPORTS_DIR}\n")


if __name__ == "__main__":
    main()
