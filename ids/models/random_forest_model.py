"""
Random Forest Intrusion Detector
Trains a RandomForestClassifier on pre-processed NSL-KDD features.
Includes GridSearchCV-based hyperparameter tuning (fast subset).
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
import time

MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "saved"
PROC_DIR   = Path(__file__).parent.parent.parent / "data" / "processed"

CLASS_NAMES = ["normal", "dos", "probe", "r2l", "u2r"]


def build_rf(n_estimators: int = 150,
             max_depth: int = None,
             min_samples_split: int = 2,
             n_jobs: int = -1,
             random_state: int = 42) -> RandomForestClassifier:
    """Instantiate a RandomForestClassifier with sensible defaults."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features="sqrt",
        class_weight="balanced",       # handles class imbalance (R2L/U2R rare)
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=0,
    )


def train(X_train: np.ndarray, y_train: np.ndarray,
          X_test: np.ndarray = None, y_test: np.ndarray = None,
          save: bool = True) -> RandomForestClassifier:
    """
    Train the Random Forest model.
    Returns the fitted classifier.
    """
    print("\n[RF] Building Random Forest model ...")
    model = build_rf()

    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"[RF] Training complete in {elapsed:.1f}s")

    if X_test is not None and y_test is not None:
        evaluate(model, X_test, y_test)

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = MODELS_DIR / "random_forest.pkl"
        joblib.dump(model, out_path)
        print(f"[RF] Model saved → {out_path}")

    return model


def evaluate(model: RandomForestClassifier,
             X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Print and return evaluation metrics."""
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    print(f"\n[RF] Accuracy : {acc*100:.2f}%")
    print(f"[RF] F1 Score : {f1*100:.2f}%")
    print("\n[RF] Classification Report:")
    # Only show classes present in test set
    present = sorted(set(y_test))
    target_names = [CLASS_NAMES[i] for i in present]
    print(classification_report(y_test, y_pred,
                                 labels=present,
                                 target_names=target_names,
                                 zero_division=0))
    return {"accuracy": acc, "f1": f1, "y_pred": y_pred}


def load_model() -> RandomForestClassifier:
    """Load a previously saved RF model."""
    path = MODELS_DIR / "random_forest.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}. Run train.py first.")
    return joblib.load(path)


def predict_proba(model: RandomForestClassifier,
                  X: np.ndarray) -> np.ndarray:
    """Return class probabilities for input X."""
    return model.predict_proba(X)


if __name__ == "__main__":
    X_train = np.load(PROC_DIR / "X_train.npy")
    X_test  = np.load(PROC_DIR / "X_test.npy")
    y_train = np.load(PROC_DIR / "y_train.npy")
    y_test  = np.load(PROC_DIR / "y_test.npy")
    train(X_train, y_train, X_test, y_test)
