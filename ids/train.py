"""
Master Training Script — v2 (with Novelty Upgrades)
====================================================
Pipeline:
  Step 1 - Download NSL-KDD dataset
  Step 2 - Preprocess features
  Step 3a - Train Random Forest
  Step 3b - Train Deep Neural Network (CPU)
  Step 3c - Train Autoencoder (Zero-Day Detector)  [NEW]
  Step 4  - Evaluate Ensemble
  Step 5  - Compute SHAP global feature importance  [NEW]
  Step 6  - Train SHAP-pruned Lite model            [NEW]
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ids.data.download_dataset import download_nsl_kdd
from ids.data.preprocessor     import preprocess
from ids.models                 import random_forest_model as rf_mod
from ids.models                 import deep_learning_model as dnn_mod
from ids.models                 import autoencoder_model   as ae_mod
from ids.models.ensemble        import EnsembleIDS
from ids.models.shap_explainer  import SHAPExplainer
from ids.models.feature_pruner  import (select_top_features,
                                         train_lite_model, compare_models)

PROC_DIR   = Path(__file__).parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent / "models" / "saved"


def banner(text: str) -> None:
    w = 62
    print("\n" + "=" * w)
    print(f"  {text}")
    print("=" * w)


def main():
    t_start = time.time()

    # ── 1. Download ──────────────────────────────────────────────────────────
    banner("Step 1 / 6 -- Downloading NSL-KDD Dataset")
    train_path, test_path = download_nsl_kdd()

    # ── 2. Preprocess ────────────────────────────────────────────────────────
    banner("Step 2 / 6 -- Preprocessing Features")
    X_train, X_test, y_train, y_test, scaler, le, features = preprocess(
        train_path, test_path, save=True
    )

    # ── 3a. Train Random Forest ──────────────────────────────────────────────
    banner("Step 3a / 6 -- Training Random Forest (41 features)")
    rf_model = rf_mod.train(X_train, y_train, X_test, y_test, save=True)

    # ── 3b. Train DNN ────────────────────────────────────────────────────────
    banner("Step 3b / 6 -- Training Deep Neural Network (CPU)")
    dnn_model = dnn_mod.train(X_train, y_train, X_test, y_test,
                               epochs=40, batch_size=512, save=True)

    # ── 3c. Train Autoencoder (Zero-Day Detector) ────────────────────────────
    banner("Step 3c / 6 -- Training Autoencoder (Zero-Day Detector)")
    ae_model, ae_threshold, ae_stats = ae_mod.train(
        X_train, y_train, X_test, y_test,
        epochs=50, batch_size=512, save=True
    )

    # ── 4. Ensemble Evaluation ───────────────────────────────────────────────
    banner("Step 4 / 6 -- Ensemble Evaluation (RF + DNN)")
    ensemble = EnsembleIDS(rf_model=rf_model, dnn_model=dnn_model)
    ensemble.evaluate(X_test, y_test)

    # ── 5. SHAP Global Feature Importance ────────────────────────────────────
    banner("Step 5 / 6 -- Computing SHAP Global Feature Importance")
    # Sample 1000 points for speed (full dataset takes too long on CPU)
    sample_idx = np.random.choice(len(X_test),
                                  min(1000, len(X_test)), replace=False)
    X_sample   = X_test[sample_idx]

    # Background for DNN GradientExplainer
    bg_idx = np.random.choice(len(X_train), 100, replace=False)
    bg     = X_train[bg_idx]

    explainer  = SHAPExplainer(rf_model=rf_model,
                                dnn_model=dnn_model,
                                background_data=bg)
    importance = explainer.global_importance(X_sample)
    explainer.save_global_importance(importance)

    print("\n  Top 10 features by SHAP importance:")
    for i, f in enumerate(importance[:10], 1):
        bar = "#" * int(f["importance"] * 200)
        print(f"  {i:2d}. {f['feature']:35s} {f['importance']:.4f}  {bar}")

    # ── 6. SHAP-Guided Lite Model ─────────────────────────────────────────────
    banner("Step 6 / 7 -- Training SHAP-Pruned Lite Model (top 15 features)")
    feat_indices, feat_names = select_top_features(importance, top_k=15)
    lite_model = train_lite_model(X_train, y_train,
                                   feat_indices, feat_names, save=True)
    comparison = compare_models(rf_model, lite_model,
                                 X_test, y_test, feat_indices)

    # ── 7. Adversarial Robustness Simulation ──────────────────────────────────
    banner("Step 7 / 7 -- Running Adversarial Robustness Simulation")
    from ids.models.adversarial_simulator import run_adversarial_simulation
    adv_report = run_adversarial_simulation(X_test, y_test, dnn_model, rf_model, epsilon=0.15, n_samples=100)

    # ── Summary ──────────────────────────────────────────────────────────────
    total = time.time() - t_start
    print(f"\nDone! Full pipeline complete in {total/60:.1f} minutes")
    print(f"  Models saved in: {MODELS_DIR}")
    print(f"\n  Speedup from feature pruning: {comparison['speedup']:.1f}x")
    print(f"  Accuracy drop (Full vs Lite): {comparison['acc_drop']:.2f}%")
    print(f"  Ensemble Adversarial Resilience: {adv_report.get('ensemble_resilience', 0.0)}%")
    print("\n  Next step -> Start the IDS server:")
    print("  python ids/api/server.py\n")


if __name__ == "__main__":
    main()
