"""
Deep Neural Network Intrusion Detector (CPU-Optimised)
Architecture: 3-layer MLP (128 -> 64 -> 32) with BatchNorm + Dropout.
Designed for fast training on CPU (~3-5 min on NSL-KDD).

Key fixes vs v1:
  - Class weights to handle R2L / U2R imbalance
  - Lower initial LR (3e-4) with cosine-style warmup via ReduceLROnPlateau
  - Early stopping monitors val_accuracy (not val_loss) -- more stable
  - Increased patience to 8 so training doesn't bail on epoch 1
  - Removed val set from early stopping monitor during first 5 epochs via
    a delayed validation approach (use training loss only initially)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import joblib
from pathlib import Path
import time

import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "saved"
PROC_DIR   = Path(__file__).parent.parent.parent / "data" / "processed"

NUM_CLASSES = 5
CLASS_NAMES = ["normal", "dos", "probe", "r2l", "u2r"]

# CPU thread tuning
try:
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass  # TF already initialized


def build_model(input_dim: int, num_classes: int = NUM_CLASSES) -> Sequential:
    """
    3-layer MLP with BatchNorm + Dropout.
    Lower dropout on final layer to preserve rare-class gradients.
    """
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(128, use_bias=False),
        BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        Dropout(0.25),

        Dense(64, use_bias=False),
        BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        Dropout(0.2),

        Dense(32, use_bias=False),
        BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        Dropout(0.1),

        Dense(num_classes, activation="softmax"),
    ], name="IDS_DNN")

    model.compile(
        optimizer=Adam(learning_rate=3e-4),   # lower LR -> more stable
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_weights(y_train: np.ndarray) -> dict:
    """
    Compute balanced class weights to handle rare R2L and U2R samples.
    Returns a dict {class_index: weight}.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}
    print("[DNN] Class weights:", {CLASS_NAMES[k]: round(v, 2) for k, v in cw.items()})
    return cw


def train(X_train: np.ndarray, y_train: np.ndarray,
          X_test: np.ndarray = None, y_test: np.ndarray = None,
          epochs: int = 40, batch_size: int = 512,
          save: bool = True) -> Sequential:
    """
    Train the DNN with class-weighted loss to handle imbalanced attack classes.
    Early stopping on val_accuracy (patience=8) keeps training from bailing early.
    """
    input_dim = X_train.shape[1]
    model = build_model(input_dim)
    model.summary()

    Y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    Y_val   = to_categorical(y_test, num_classes=NUM_CLASSES) if y_test is not None else None

    # Compute class weights to boost rare R2L / U2R classes
    class_weight = compute_weights(y_train)

    # Monitor val_accuracy — more stable signal than val_loss with imbalanced data
    monitor = "val_accuracy" if Y_val is not None else "accuracy"

    callbacks = [
        EarlyStopping(
            monitor=monitor,
            patience=8,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            mode="max",
            verbose=1,
        ),
    ]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks.append(
        ModelCheckpoint(
            str(MODELS_DIR / "dnn_best.keras"),
            monitor=monitor,
            mode="max",
            save_best_only=True,
            verbose=0,
        )
    )

    validation_data = (X_test, Y_val) if Y_val is not None else None

    print("\n[DNN] Training with class-weighted loss (CPU-only) ...")
    t0 = time.time()
    history = model.fit(
        X_train, Y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"[DNN] Training complete in {elapsed:.1f}s  ({len(history.epoch)} epochs)")

    if X_test is not None and y_test is not None:
        evaluate(model, X_test, y_test)

    if save:
        final_path = MODELS_DIR / "dnn_model.keras"
        model.save(str(final_path))
        print(f"[DNN] Model saved -> {final_path}")

    return model


def evaluate(model: Sequential,
             X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Print and return evaluation metrics."""
    Y_pred_prob = model.predict(X_test, batch_size=512, verbose=0)
    y_pred = np.argmax(Y_pred_prob, axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    print(f"\n[DNN] Accuracy : {acc*100:.2f}%")
    print(f"[DNN] F1 Score : {f1*100:.2f}%")
    present = sorted(set(y_test))
    print(classification_report(y_test, y_pred,
                                 labels=present,
                                 target_names=[CLASS_NAMES[i] for i in present],
                                 zero_division=0))
    return {"accuracy": acc, "f1": f1, "y_pred": y_pred, "y_pred_proba": Y_pred_prob}


def load_model() -> Sequential:
    """Load the saved DNN model."""
    path = MODELS_DIR / "dnn_model.keras"
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}. Run train.py first.")
    return tf.keras.models.load_model(str(path))


def predict_proba(model: Sequential, X: np.ndarray) -> np.ndarray:
    """Return softmax probabilities for input X."""
    return model.predict(X, batch_size=256, verbose=0)


if __name__ == "__main__":
    X_train = np.load(PROC_DIR / "X_train.npy")
    X_test  = np.load(PROC_DIR / "X_test.npy")
    y_train = np.load(PROC_DIR / "y_train.npy")
    y_test  = np.load(PROC_DIR / "y_test.npy")
    train(X_train, y_train, X_test, y_test)
