import os
import sys
import numpy as np
import joblib
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, str(Path(__file__).parent))

import tensorflow as tf
from ids.models.adversarial_simulator import run_adversarial_simulation

PROC_DIR   = Path("data/processed")
MODELS_DIR = Path("models/saved")

print("[ADV] Loading dataset and models...")
X_test  = np.load(PROC_DIR / "X_test.npy")
y_test  = np.load(PROC_DIR / "y_test.npy")

rf  = joblib.load(MODELS_DIR / "random_forest.pkl")
dnn = tf.keras.models.load_model(str(MODELS_DIR / "dnn_model.keras"))

print("[ADV] Running adversarial simulation...")
report = run_adversarial_simulation(X_test, y_test, dnn, rf, epsilon=0.15, n_samples=100)
print("[ADV] Report successfully generated and saved to models/saved/adversarial_report.pkl")
