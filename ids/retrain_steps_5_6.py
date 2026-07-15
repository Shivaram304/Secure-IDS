"""
Quick re-run of Steps 5 & 6 only (SHAP + Feature Pruning).
RF, DNN and Autoencoder are already trained and saved.
Also recalibrates the autoencoder threshold to 95th percentile.
"""
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PROC_DIR   = Path("data/processed")
MODELS_DIR = Path("models/saved")

# Load data
X_train = np.load(PROC_DIR / "X_train.npy")
X_test  = np.load(PROC_DIR / "X_test.npy")
y_train = np.load(PROC_DIR / "y_train.npy")
y_test  = np.load(PROC_DIR / "y_test.npy")

# Load models
import tensorflow as tf
rf  = joblib.load(MODELS_DIR / "random_forest.pkl")
dnn = tf.keras.models.load_model(str(MODELS_DIR / "dnn_model.keras"))
ae  = tf.keras.models.load_model(str(MODELS_DIR / "autoencoder.keras"))

# ── Recalibrate AE threshold to 95th percentile ────────────────────────────
print("\n[AE] Recalibrating threshold to 95th percentile...")
from ids.models.autoencoder_model import compute_reconstruction_errors, evaluate
normal_mask = (y_train == 0)
X_normal = X_train[normal_mask]
split    = int(0.9 * len(X_normal))
X_va_ae  = X_normal[split:]
val_errors = compute_reconstruction_errors(ae, X_va_ae)
threshold  = float(np.percentile(val_errors, 95))
print(f"[AE] New threshold (95th pct): {threshold:.6f}")
ae_stats   = evaluate(ae, X_test, y_test, threshold)
joblib.dump({"threshold": threshold, "stats": ae_stats}, MODELS_DIR / "ae_threshold.pkl")
print("[AE] Threshold saved.")

# ── Step 5: SHAP Global Feature Importance ─────────────────────────────────
print("\n" + "="*62)
print("  Step 5 / 6 -- Computing SHAP Global Feature Importance")
print("="*62)
sample_idx = np.random.choice(len(X_test), min(1000, len(X_test)), replace=False)
X_sample   = X_test[sample_idx]
bg_idx = np.random.choice(len(X_train), 100, replace=False)
bg     = X_train[bg_idx]

from ids.models.shap_explainer import SHAPExplainer
explainer  = SHAPExplainer(rf_model=rf, dnn_model=dnn, background_data=bg)
importance = explainer.global_importance(X_sample)
explainer.save_global_importance(importance)
print("\nTop 10 features by SHAP:")
for i, f in enumerate(importance[:10], 1):
    bar = "#" * int(f["importance"] * 200)
    print(f"  {i:2d}. {f['feature']:35s} {f['importance']:.4f}  {bar}")

# ── Step 6: SHAP-Guided Lite Model ─────────────────────────────────────────
print("\n" + "="*62)
print("  Step 6 / 6 -- Training SHAP-Pruned Lite Model (top 15 features)")
print("="*62)
from ids.models.feature_pruner import select_top_features, train_lite_model, compare_models
feat_indices, feat_names = select_top_features(importance, top_k=15)
lite_model = train_lite_model(X_train, y_train, feat_indices, feat_names, save=True)
comparison = compare_models(rf, lite_model, X_test, y_test, feat_indices)

print(f"\nDone! Speedup: {comparison['speedup']:.1f}x | Accuracy drop: {comparison['acc_drop']:.2f}%")
print("Now start the server:  python ids/api/server.py")
