"""
Adversarial Attack Simulator
==============================
Generates adversarial examples that attempt to EVADE the IDS,
then shows how the ensemble detects or misses them.

Novel contribution:
  - Most IDS papers never test adversarial robustness
  - Demonstrates real-world evasion risk
  - Uses FGSM-style perturbation on continuous features
  - Generates a 'cat-and-mouse' report: attack vs defence

Technique:
  Feature-space FGSM (Fast Gradient Sign Method) adapted for tabular data:
    x_adv = x + epsilon * sign(gradient of loss w.r.t. x)
  Applied only to continuous features (not binary/categorical).
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import joblib
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "saved"
PROC_DIR   = Path(__file__).parent.parent.parent / "data"  / "processed"

CLASS_NAMES = ["normal", "dos", "probe", "r2l", "u2r"]

# Continuous feature indices (safe to perturb)
CONTINUOUS_FEATURES = [0, 4, 5, 7, 8, 9, 22, 23, 31, 32]

# Binary/categorical feature indices (keep unchanged)
CATEGORICAL_FEATURES = [1, 2, 3, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def fgsm_perturb(X: np.ndarray, dnn_model,
                 target_class: int = 0,
                 epsilon: float = 0.1,
                 n_steps: int = 5) -> np.ndarray:
    """
    Iterative FGSM perturbation to push a sample toward 'normal' class.
    Perturbs only continuous features to maintain realism.

    Args:
        X: input features (1, 41)
        dnn_model: trained Keras DNN
        target_class: class index to fool the model into predicting (0=normal)
        epsilon: perturbation step size per iteration
        n_steps: number of gradient steps

    Returns:
        x_adv: perturbed sample (1, 41)
    """
    import tensorflow as tf
    x_adv = tf.Variable(X.astype(np.float32), trainable=True)
    target = tf.constant([[1.0 if i == target_class else 0.0
                           for i in range(len(CLASS_NAMES))]])

    for _ in range(n_steps):
        with tf.GradientTape() as tape:
            preds = dnn_model(x_adv, training=False)
            loss  = tf.keras.losses.categorical_crossentropy(target, preds)
        grads = tape.gradient(loss, x_adv)

        # Apply perturbation only to continuous features
        mask = np.zeros(X.shape[1], dtype=np.float32)
        for i in CONTINUOUS_FEATURES:
            mask[i] = 1.0

        perturbation = epsilon * tf.sign(grads) * mask
        x_adv.assign_sub(perturbation)   # gradient descent toward target

    return x_adv.numpy()


def run_adversarial_simulation(X_sample: np.ndarray,
                                y_sample: np.ndarray,
                                dnn_model,
                                rf_model,
                                epsilon: float = 0.15,
                                n_samples: int = 100) -> dict:
    """
    Run adversarial evasion simulation on n_samples attack records.
    Tries to make the DNN classify attacks as 'normal'.

    Returns a report dict with evasion rates and model resilience scores.
    """
    # Pick only attack samples (not normal)
    attack_mask = (y_sample != 0)
    X_attacks = X_sample[attack_mask][:n_samples]
    y_attacks = y_sample[attack_mask][:n_samples]

    if len(X_attacks) == 0:
        return {"error": "No attack samples found in provided data"}

    print(f"[ADV] Running adversarial simulation on {len(X_attacks)} attack samples...")

    dnn_original = []
    dnn_adversarial = []
    rf_original = []
    rf_adversarial = []

    for i, (x, y_true) in enumerate(zip(X_attacks, y_attacks)):
        x_orig = x.reshape(1, -1)

        # Original predictions
        dnn_pred_orig = int(np.argmax(dnn_model.predict(x_orig, verbose=0)[0]))
        rf_pred_orig  = int(rf_model.predict(x_orig)[0])

        # Generate adversarial example
        x_adv = fgsm_perturb(x_orig, dnn_model,
                              target_class=0, epsilon=epsilon, n_steps=5)

        # Adversarial predictions
        dnn_pred_adv = int(np.argmax(dnn_model.predict(x_adv, verbose=0)[0]))
        rf_pred_adv  = int(rf_model.predict(x_adv)[0])

        dnn_original.append(dnn_pred_orig == int(y_true))
        dnn_adversarial.append(dnn_pred_adv == int(y_true))
        rf_original.append(rf_pred_orig == int(y_true))
        rf_adversarial.append(rf_pred_adv == int(y_true))

        if (i + 1) % 20 == 0:
            print(f"[ADV]   Processed {i+1}/{len(X_attacks)}")

    n = len(X_attacks)
    dnn_orig_acc  = sum(dnn_original)  / n * 100
    dnn_adv_acc   = sum(dnn_adversarial) / n * 100
    rf_orig_acc   = sum(rf_original)   / n * 100
    rf_adv_acc    = sum(rf_adversarial) / n * 100

    dnn_evasion_rate = max(0, dnn_orig_acc - dnn_adv_acc)
    rf_evasion_rate  = max(0, rf_orig_acc  - rf_adv_acc)

    # Ensemble robustness: attack evades both models
    both_fooled = sum(
        not da and not ra
        for da, ra in zip(dnn_adversarial, rf_adversarial)
    ) / n * 100

    report = {
        "n_samples":         n,
        "epsilon":           epsilon,
        "dnn": {
            "original_accuracy":    round(dnn_orig_acc, 2),
            "adversarial_accuracy": round(dnn_adv_acc, 2),
            "evasion_rate":         round(dnn_evasion_rate, 2),
            "resilience_score":     round(100 - dnn_evasion_rate, 2),
        },
        "rf": {
            "original_accuracy":    round(rf_orig_acc, 2),
            "adversarial_accuracy": round(rf_adv_acc, 2),
            "evasion_rate":         round(rf_evasion_rate, 2),
            "resilience_score":     round(100 - rf_evasion_rate, 2),
        },
        "ensemble_evasion_rate":  round(both_fooled, 2),
        "ensemble_resilience":    round(100 - both_fooled, 2),
        "verdict": (
            "HIGHLY ROBUST — adversarial attacks barely affect the ensemble."
            if both_fooled < 10 else
            "MODERATELY ROBUST — some evasion possible with careful crafting."
            if both_fooled < 30 else
            "VULNERABLE — adversarial retraining recommended."
        )
    }

    print(f"[ADV] DNN evasion rate: {dnn_evasion_rate:.1f}%")
    print(f"[ADV] RF  evasion rate: {rf_evasion_rate:.1f}%")
    print(f"[ADV] Ensemble evasion rate: {both_fooled:.1f}%")
    print(f"[ADV] Verdict: {report['verdict']}")

    # Save report
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(report, MODELS_DIR / "adversarial_report.pkl")
    return report


if __name__ == "__main__":
    import tensorflow as tf
    rf  = joblib.load(MODELS_DIR / "random_forest.pkl")
    dnn = tf.keras.models.load_model(str(MODELS_DIR / "dnn_model.keras"))
    X   = np.load(PROC_DIR / "processed" / "X_test.npy")
    y   = np.load(PROC_DIR / "processed" / "y_test.npy")
    report = run_adversarial_simulation(X, y, dnn, rf, epsilon=0.15, n_samples=100)
