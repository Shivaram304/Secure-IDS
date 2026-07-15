"""
Ensemble Model: Random Forest + DNN Weighted Voting
Combines probability outputs from both models to give a final prediction.
RF is typically slightly more accurate on NSL-KDD, so it gets more weight.
"""

import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report

CLASS_NAMES = ["normal", "dos", "probe", "r2l", "u2r"]

# Weights: RF gets 0.55, DNN gets 0.45 (tunable)
DEFAULT_WEIGHTS = (0.55, 0.45)


class EnsembleIDS:
    """
    Weighted soft-voting ensemble of RandomForest + DNN.
    Both models must produce class probabilities of shape (N, 5).
    """

    def __init__(self, rf_model=None, dnn_model=None,
                 weights: tuple = DEFAULT_WEIGHTS):
        self.rf_model  = rf_model
        self.dnn_model = dnn_model
        self.rf_w, self.dnn_w = weights

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return weighted average probability matrix (N × 5)."""
        rf_proba  = self.rf_model.predict_proba(X)          # sklearn API
        dnn_proba = self.dnn_model.predict(X, batch_size=256, verbose=0)  # keras API
        return self.rf_w * rf_proba + self.dnn_w * dnn_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class indices."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_label(self, X: np.ndarray) -> list[str]:
        """Return human-readable class names."""
        indices = self.predict(X)
        return [CLASS_NAMES[i] for i in indices]

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        print(f"\n[ENSEMBLE] Accuracy : {acc*100:.2f}%")
        print(f"[ENSEMBLE] F1 Score : {f1*100:.2f}%")
        present = sorted(set(y_test))
        print(classification_report(y_test, y_pred,
                                     labels=present,
                                     target_names=[CLASS_NAMES[i] for i in present],
                                     zero_division=0))
        return {"accuracy": acc, "f1": f1, "y_pred": y_pred}

    @classmethod
    def load(cls, weights: tuple = DEFAULT_WEIGHTS):
        """Load both sub-models from disk and return a ready ensemble."""
        import joblib
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        import tensorflow as tf

        saved = Path(__file__).parent.parent.parent / "models" / "saved"
        rf  = joblib.load(saved / "random_forest.pkl")
        dnn = tf.keras.models.load_model(str(saved / "dnn_model.keras"))
        print("[ENSEMBLE] Both models loaded.")
        return cls(rf_model=rf, dnn_model=dnn, weights=weights)


def predict_single(ensemble: EnsembleIDS, feature_vector: np.ndarray) -> dict:
    """
    Predict a single connection record.
    feature_vector: 1-D array of 41 scaled features.
    Returns dict with label, confidence, and all class probabilities.
    """
    X = feature_vector.reshape(1, -1)
    proba = ensemble.predict_proba(X)[0]
    idx   = int(np.argmax(proba))
    return {
        "label":       CLASS_NAMES[idx],
        "confidence":  float(proba[idx]),
        "probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, proba)},
    }
