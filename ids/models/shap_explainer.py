"""
SHAP Explainability Engine
==========================
Provides local (per-prediction) and global (dataset-level) explanations
for both the Random Forest and DNN models using SHAP values.

Key capabilities:
  - TreeExplainer for RF   (fast, exact Shapley values)
  - GradientExplainer for DNN (fast approximation on CPU)
  - Global feature importance ranking
  - Top-N feature explanation for individual predictions
  - SHAP-guided feature selection (for Feature Pruner)

References:
  Lundberg & Lee (2017) - "A Unified Approach to Interpreting Model Predictions"
  Applied to IDS: addresses the "black-box" problem flagged in 2024 research.
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
PROC_DIR   = Path(__file__).parent.parent.parent / "data" / "processed"

FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

CLASS_NAMES = ["normal", "dos", "probe", "r2l", "u2r"]


class SHAPExplainer:
    """
    Unified SHAP explainer for the IDS ensemble.

    Uses TreeExplainer for RF (exact, fast) and
    GradientExplainer for DNN (approximate, still fast on CPU).
    """

    def __init__(self, rf_model=None, dnn_model=None,
                 background_data: Optional[np.ndarray] = None):
        import shap
        self.shap = shap
        self.rf_model  = rf_model
        self.dnn_model = dnn_model
        self._rf_explainer  = None
        self._dnn_explainer = None
        self.background     = background_data   # small sample for DNN background

        if rf_model is not None:
            print("[SHAP] Initialising TreeExplainer for Random Forest...")
            self._rf_explainer = shap.TreeExplainer(
                rf_model,
                feature_perturbation="interventional",
            )
            print("[SHAP] TreeExplainer ready.")

        if dnn_model is not None and background_data is not None:
            print("[SHAP] Initialising GradientExplainer for DNN...")
            import tensorflow as tf
            # Use a small background sample (100 rows) for speed
            bg = background_data[:100]
            self._dnn_explainer = shap.GradientExplainer(
                dnn_model, bg
            )
            print("[SHAP] GradientExplainer ready.")

    # ── Local Explanation (single prediction) ─────────────────────────────────

    def explain_rf(self, X: np.ndarray, top_n: int = 10) -> list[dict]:
        """
        Explain a single prediction using RF SHAP values.
        Handles both list-of-arrays (old SHAP) and 3D array (new SHAP) output.
        """
        if self._rf_explainer is None:
            return []
        shap_vals  = self._rf_explainer.shap_values(X)
        proba      = self.rf_model.predict_proba(X)[0]
        pred_class = int(np.argmax(proba))
        # New SHAP: 3D array (n_samples, n_features, n_classes)
        if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            vals = shap_vals[0, :, pred_class]      # shape (41,)
        else:
            # Old SHAP: list of (n_samples, n_features), one per class
            vals = np.array(shap_vals[pred_class])[0]  # shape (41,)
        return _top_features(vals, top_n)

    def explain_dnn(self, X: np.ndarray, top_n: int = 10) -> list[dict]:
        """Explain a single prediction using DNN SHAP values."""
        if self._dnn_explainer is None:
            return []
        shap_vals  = self._dnn_explainer.shap_values(X)   # list per class
        dnn_proba  = self.dnn_model.predict(X, verbose=0)
        pred_class = int(np.argmax(dnn_proba[0]))
        vals       = np.array(shap_vals[pred_class])[0]
        return _top_features(vals, top_n)

    def explain_ensemble(self, X: np.ndarray, top_n: int = 10,
                         rf_weight: float = 0.55) -> list[dict]:
        """
        Weighted combination of RF + DNN SHAP values.
        Mirrors the ensemble's soft-voting weights.
        """
        rf_vals, dnn_vals = None, None

        if self._rf_explainer is not None:
            shap_vals  = self._rf_explainer.shap_values(X)
            proba      = self.rf_model.predict_proba(X)[0]
            pred_class = int(np.argmax(proba))
            rf_vals    = shap_vals[pred_class][0]

        if self._dnn_explainer is not None:
            shap_vals  = self._dnn_explainer.shap_values(X)
            dnn_proba  = self.dnn_model.predict(X, verbose=0)
            pred_class = int(np.argmax(dnn_proba[0]))
            dnn_vals   = np.array(shap_vals[pred_class])[0]

        if rf_vals is not None and dnn_vals is not None:
            combined = rf_weight * rf_vals + (1 - rf_weight) * dnn_vals
        elif rf_vals is not None:
            combined = rf_vals
        else:
            combined = dnn_vals

        return _top_features(combined, top_n)

    # ── Global Feature Importance ─────────────────────────────────────────────

    def global_importance(self, X_sample: np.ndarray) -> list[dict]:
        """
        Compute mean |SHAP| across a sample to get global feature importance.
        Handles both 3D array (new SHAP) and list-of-arrays (old SHAP) formats.
        """
        if self._rf_explainer is None:
            return []

        print(f"[SHAP] Computing global importance on {len(X_sample)} samples...")
        shap_vals = self._rf_explainer.shap_values(X_sample)

        # New SHAP versions return a 3D ndarray: (n_samples, n_features, n_classes)
        if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            # mean |SHAP| over classes then over samples -> (n_features,)
            mean_abs = np.mean(np.abs(shap_vals), axis=(0, 2))  # (n_features,)
        else:
            # Old SHAP: list of (n_samples, n_features) arrays, one per class
            all_abs  = np.mean([np.abs(sv) for sv in shap_vals], axis=0)  # (N, 41)
            mean_abs = np.mean(all_abs, axis=0)                            # (41,)

        n_features = mean_abs.shape[0]
        ranked = sorted(
            [{"feature": FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feat_{i}",
              "index": i, "importance": float(mean_abs[i])}
             for i in range(n_features)],
            key=lambda x: x["importance"], reverse=True
        )
        return ranked

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_global_importance(self, importance: list[dict]) -> Path:
        out = MODELS_DIR / "shap_global_importance.pkl"
        joblib.dump(importance, out)
        print(f"[SHAP] Global importance saved -> {out}")
        return out

    @classmethod
    def load(cls, background_size: int = 100):
        """Load both sub-models and build explainer."""
        import tensorflow as tf
        rf  = joblib.load(MODELS_DIR / "random_forest.pkl")
        dnn = tf.keras.models.load_model(str(MODELS_DIR / "dnn_model.keras"))
        X_train = np.load(PROC_DIR / "X_train.npy")
        bg = X_train[np.random.choice(len(X_train), background_size, replace=False)]
        return cls(rf_model=rf, dnn_model=dnn, background_data=bg)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _top_features(shap_values: np.ndarray, top_n: int) -> list[dict]:
    """Sort features by |SHAP| and return top_n as list of dicts."""
    indices = np.argsort(np.abs(shap_values))[::-1][:top_n]
    return [
        {
            "feature":    FEATURE_NAMES[i],
            "shap_value": float(shap_values[i]),
            "abs_impact": float(abs(shap_values[i])),
            "direction":  "increases_risk" if shap_values[i] > 0 else "reduces_risk",
        }
        for i in indices
    ]


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[SHAP] Loading models and computing explanations...")
    explainer = SHAPExplainer.load()

    X_test = np.load(PROC_DIR / "X_test.npy")

    # Local explanation for first test record
    explanation = explainer.explain_ensemble(X_test[:1], top_n=10)
    print("\nTop 10 features for first test record:")
    for e in explanation:
        bar = "+" * int(e["abs_impact"] * 100)
        direction = "^" if e["direction"] == "increases_risk" else "v"
        print(f"  [{direction}] {e['feature']:35s} SHAP={e['shap_value']:+.4f}")

    # Global importance
    sample_idx = np.random.choice(len(X_test), 500, replace=False)
    importance = explainer.global_importance(X_test[sample_idx])
    explainer.save_global_importance(importance)
    print("\nTop 15 globally important features:")
    for f in importance[:15]:
        bar = "#" * int(f["importance"] * 200)
        print(f"  {f['feature']:35s} {f['importance']:.4f}  {bar}")
