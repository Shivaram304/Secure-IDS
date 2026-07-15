"""
NSL-KDD Data Preprocessor
Loads raw NSL-KDD text files, engineers features, encodes categoricals,
normalises, and returns train/test numpy arrays ready for ML.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent.parent.parent / "data"
RAW_DIR     = DATA_DIR / "raw"
PROC_DIR    = DATA_DIR / "processed"
MODELS_DIR  = Path(__file__).parent.parent.parent / "models" / "saved"

# ── Feature Columns ───────────────────────────────────────────────────────────
COLUMNS = [
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
    "label", "difficulty_level"
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# Map NSL-KDD fine-grained labels → 5 main attack categories
ATTACK_MAP = {
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "mailbomb": "dos",
    # Probe
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "satan": "probe", "mscan": "probe", "saint": "probe",
    # R2L
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l",
    "multihop": "r2l", "phf": "r2l", "spy": "r2l", "warezclient": "r2l",
    "warezmaster": "r2l", "sendmail": "r2l", "named": "r2l",
    "snmpgetattack": "r2l", "snmpguess": "r2l", "xlock": "r2l",
    "xsnoop": "r2l", "httptunnel": "r2l",
    # U2R
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r", "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
}

CLASS_NAMES = ["normal", "dos", "probe", "r2l", "u2r"]


def load_raw(path: Path) -> pd.DataFrame:
    """Read a raw NSL-KDD text file into a DataFrame."""
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df.drop(columns=["difficulty_level"], inplace=True)
    return df


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map fine-grained labels to 5 attack categories."""
    df = df.copy()
    df["attack_cat"] = df["label"].str.lower().map(ATTACK_MAP)
    df["attack_cat"].fillna("unknown", inplace=True)
    df = df[df["attack_cat"] != "unknown"]
    return df


def encode_categoricals(train_df: pd.DataFrame,
                         test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """One-hot encode categorical columns using training distribution."""
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(train_df[col])
        # Handle unseen labels in test with a fallback
        def safe_transform(series, enc):
            known = set(enc.classes_)
            return series.apply(lambda x: enc.transform([x])[0]
                                if x in known else -1)
        train_df[col] = le.transform(train_df[col])
        test_df[col]  = safe_transform(test_df[col], le)
        encoders[col] = le
    return train_df, test_df, encoders


def preprocess(train_path: Path = None,
               test_path: Path = None,
               save: bool = True):
    """
    Full preprocessing pipeline.
    Returns: X_train, X_test, y_train, y_test, scaler, label_encoder, feature_names
    """
    if train_path is None:
        train_path = RAW_DIR / "KDDTrain+.txt"
    if test_path is None:
        test_path = RAW_DIR / "KDDTest+.txt"

    print("[PREPROCESS] Loading raw data ...")
    train_df = load_raw(train_path)
    test_df  = load_raw(test_path)

    print("[PREPROCESS] Mapping attack categories ...")
    train_df = map_labels(train_df)
    test_df  = map_labels(test_df)

    # Encode target
    le_label = LabelEncoder()
    le_label.fit(CLASS_NAMES)
    y_train = le_label.transform(train_df["attack_cat"])
    y_test  = le_label.transform(test_df["attack_cat"])

    # Drop label columns
    train_df.drop(columns=["label", "attack_cat"], inplace=True)
    test_df.drop(columns=["label", "attack_cat"], inplace=True)

    # Encode categoricals
    train_df, test_df, cat_encoders = encode_categoricals(train_df, test_df)

    feature_names = list(train_df.columns)

    X_train = train_df.values.astype(np.float32)
    X_test  = test_df.values.astype(np.float32)

    # Scale features
    print("[PREPROCESS] Scaling features ...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"[PREPROCESS] Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"[PREPROCESS] Classes: {le_label.classes_}")

    if save:
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(PROC_DIR / "X_train.npy", X_train)
        np.save(PROC_DIR / "X_test.npy",  X_test)
        np.save(PROC_DIR / "y_train.npy", y_train)
        np.save(PROC_DIR / "y_test.npy",  y_test)
        joblib.dump(scaler,      MODELS_DIR / "scaler.pkl")
        joblib.dump(le_label,    MODELS_DIR / "label_encoder.pkl")
        joblib.dump(cat_encoders,MODELS_DIR / "cat_encoders.pkl")
        joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
        print(f"[PREPROCESS] Saved processed data → {PROC_DIR}")

    return X_train, X_test, y_train, y_test, scaler, le_label, feature_names


if __name__ == "__main__":
    preprocess()
