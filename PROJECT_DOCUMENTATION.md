# Secure-IDS Project Documentation

## What this is

Secure-IDS is a Python-based, machine-learning network intrusion detection system for classifying network traffic into five categories: normal traffic, DoS, Probe, R2L, and U2R attacks. It combines Random Forest and TensorFlow neural-network models with live packet capture, PCAP analysis, a Flask REST/WebSocket API, and a browser dashboard for monitoring and investigation.

The project also includes explainable AI features using SHAP, an autoencoder for anomaly and potential zero-day detection, SHAP-guided feature pruning, attacker profiling, MITRE ATT&CK references, and adversarial robustness simulation.

## Stack

- **Languages:** Python, JavaScript, HTML, CSS
- **Runtime and frameworks:** Python 3.10+, Flask, Flask-SocketIO, TensorFlow/Keras
- **Machine learning:** scikit-learn, TensorFlow/Keras, NumPy, pandas, joblib
- **Network analysis:** Scapy
- **Visualization:** Matplotlib, Seaborn, Chart.js
- **Dataset:** NSL-KDD

## Repository structure

```text
Secure-IDS/
├── README.md
│   Project description, setup instructions, API overview, and feature list
│
├── requirements.txt
│   Python dependencies for ML, API, packet capture, and visualization
│
├── generate_adv_report.py
│   Loads trained models and creates an adversarial robustness report
│
├── ids/
│   Main Python application package
│
│   ├── train.py
│   │   Master training pipeline
│   │
│   ├── evaluate.py
│   │   Generates accuracy summaries, confusion matrices, and ROC curves
│   │
│   ├── retrain_steps_5_6.py
│   │   Auxiliary retraining script for later training stages
│   │
│   ├── data/
│   │   ├── download_dataset.py
│   │   │   Downloads the NSL-KDD train and test datasets
│   │   │
│   │   └── preprocessor.py
│   │       Maps labels, encodes categoricals, scales features, and saves arrays
│   │
│   ├── models/
│   │   ├── random_forest_model.py
│   │   │   Random Forest classifier
│   │   │
│   │   ├── deep_learning_model.py
│   │   │   TensorFlow/Keras multilayer perceptron
│   │   │
│   │   ├── ensemble.py
│   │   │   Weighted Random Forest + DNN probability ensemble
│   │   │
│   │   ├── autoencoder_model.py
│   │   │   Normal-traffic autoencoder for anomaly and zero-day detection
│   │   │
│   │   ├── shap_explainer.py
│   │   │   SHAP-based feature explanations and global importance
│   │   │
│   │   ├── feature_pruner.py
│   │   │   Selects important features and trains a smaller model
│   │   │
│   │   ├── narrative_engine.py
│   │   │   Generates natural-language incident narratives and attacker profiles
│   │   │
│   │   └── adversarial_simulator.py
│   │       Evaluates model behavior against adversarial perturbations
│   │
│   ├── capture/
│   │   ├── packet_sniffer.py
│   │   │   Captures live packets, analyzes PCAP files, or generates demo traffic
│   │   │
│   │   └── feature_extractor.py
│   │       Converts packets and connection records into 41 NSL-KDD features
│   │
│   └── api/
│       └── server.py
│           Flask application, REST endpoints, Socket.IO events, and capture control
│
├── dashboard/
│   ├── index.html
│   │   Dashboard page and monitoring widgets
│   │
│   ├── css/
│   │   └── styles.css
│   │       Dark-theme dashboard styling and responsive layout
│   │
│   └── js/
│       └── dashboard.js
│           Chart rendering, API calls, WebSocket handling, and UI state
│
├── data/
│   └── .gitkeep
│       Placeholder for generated raw and processed datasets
│
└── models/
    └── saved/
        Generated model files, preprocessing artifacts, SHAP results,
        lite-model data, and adversarial reports
```

## Architecture and data flow

The project begins by downloading NSL-KDD data through `ids/data/download_dataset.py`. `ids/data/preprocessor.py` converts the original records into 41 numerical features, maps detailed attack names to five classes, applies categorical encoding and standardization, and saves NumPy arrays and preprocessing objects.

`ids/train.py` trains the Random Forest, DNN, autoencoder, SHAP explainer, reduced-feature model, and adversarial simulation. The trained artifacts are stored under `models/saved/`. At runtime, `ids/api/server.py` loads the saved models, starts a Flask/Socket.IO server, and connects to `PacketSniffer`.

For real or simulated traffic, `ids/capture/feature_extractor.py` converts packets into the same 41-feature format used during training. `packet_sniffer.py` sends those features to the RF+DNN ensemble. Detection results are stored in server-side buffers, enriched with anomaly scores and narrative information, and emitted to the dashboard through Socket.IO.

The browser dashboard in `dashboard/index.html` displays packet counts, attack distribution, alert history, threat levels, SHAP importance, anomaly scores, attacker profiles, geographic attack animations, and adversarial robustness metrics.

## Machine-learning pipeline

### Dataset acquisition

`ids/data/download_dataset.py` downloads the NSL-KDD training and test files:

```text
data/raw/KDDTrain+.txt
data/raw/KDDTest+.txt
```

The downloader skips files that already exist and displays progress while downloading.

### Label transformation

The original NSL-KDD labels are converted into five categories:

| Category | Examples |
|---|---|
| `normal` | Normal traffic |
| `dos` | `neptune`, `smurf`, `teardrop`, `back`, `land` |
| `probe` | `nmap`, `ipsweep`, `portsweep`, `satan` |
| `r2l` | `ftp_write`, `guess_passwd`, `warezclient`, `imap` |
| `u2r` | `buffer_overflow`, `rootkit`, `loadmodule`, `xterm` |

The canonical class order is:

```text
normal, dos, probe, r2l, u2r
```

### Preprocessing

`ids/data/preprocessor.py` performs the following operations:

1. Loads raw NSL-KDD files.
2. Removes `difficulty_level`.
3. Maps detailed attack labels into five classes.
4. Encodes `protocol_type`, `service`, and `flag`.
5. Converts features to `float32`.
6. Applies `StandardScaler`.
7. Saves processed arrays and preprocessing objects.

Generated files include:

```text
data/processed/X_train.npy
data/processed/X_test.npy
data/processed/y_train.npy
data/processed/y_test.npy
models/saved/scaler.pkl
models/saved/label_encoder.pkl
models/saved/cat_encoders.pkl
models/saved/feature_names.pkl
```

### Random Forest

`ids/models/random_forest_model.py` creates a `RandomForestClassifier` with:

- 150 trees
- `class_weight="balanced"`
- Square-root feature selection
- A fixed random seed
- Parallel training through `n_jobs=-1`

The trained model is saved as:

```text
models/saved/random_forest.pkl
```

### Deep neural network

`ids/models/deep_learning_model.py` defines a CPU-optimized multilayer perceptron:

```text
41 inputs
  ↓
Dense(128) + BatchNormalization + ReLU + Dropout
  ↓
Dense(64) + BatchNormalization + ReLU + Dropout
  ↓
Dense(32) + BatchNormalization + ReLU + Dropout
  ↓
Dense(5) + Softmax
```

The DNN uses Adam, categorical cross-entropy, class-weighted loss, early stopping, learning-rate reduction, and model checkpointing.

Saved files include:

```text
models/saved/dnn_best.keras
models/saved/dnn_model.keras
```

### Weighted ensemble

`ids/models/ensemble.py` combines model probabilities using weighted soft voting:

```text
ensemble_probability =
    0.55 × Random Forest probability
  + 0.45 × DNN probability
```

The class with the highest combined probability becomes the final prediction.

The ensemble exposes:

- `predict_proba()`
- `predict()`
- `predict_label()`
- `evaluate()`
- `predict_single()`

### Autoencoder anomaly detector

`ids/models/autoencoder_model.py` trains only on normal traffic.

```text
Encoder: 41 → 32 → 16 → 8
Decoder: 8 → 16 → 32 → 41
```

The anomaly score is the mean squared reconstruction error. A threshold is calculated from the 95th percentile of normal validation errors.

Generated files:

```text
models/saved/autoencoder.keras
models/saved/ae_encoder.keras
models/saved/ae_threshold.pkl
```

The detector returns a reconstruction error, normalized anomaly score, anomaly flag, and low/medium/high zero-day risk.

### SHAP explainability

The training pipeline computes global feature importance using `SHAPExplainer`. These results are used to display the most important features, explain individual predictions, and select features for the lite model.

The generated global importance file is expected at:

```text
models/saved/shap_global_importance.pkl
```

### SHAP-guided lite model

`feature_pruner.py` selects the top 15 features based on SHAP importance and trains a smaller Random Forest model.

Expected artifacts include:

```text
models/saved/rf_lite.pkl
models/saved/lite_feature_indices.pkl
models/saved/lite_feature_names.pkl
models/saved/model_comparison.pkl
```

The dashboard compares full-model accuracy, lite-model accuracy, inference speedup, and accuracy loss.

### Adversarial simulation

`ids/models/adversarial_simulator.py` evaluates how model predictions change when continuous features are perturbed.

The standalone helper script runs:

```python
run_adversarial_simulation(
    X_test,
    y_test,
    dnn,
    rf,
    epsilon=0.15,
    n_samples=100,
)
```

The report is saved as:

```text
models/saved/adversarial_report.pkl
```

## Packet-capture pipeline

### Live capture

`ids/capture/packet_sniffer.py` uses Scapy to capture packets from a network interface:

```text
Scapy packet
    ↓
packet_to_features()
    ↓
ConnectionTracker statistics
    ↓
Feature scaling
    ↓
RF + DNN ensemble
    ↓
Detection record
```

Each detection includes a timestamp, source IP, destination IP, protocol, predicted label, severity, confidence, and per-class probabilities.

Severity mapping:

| Label | Severity |
|---|---|
| `normal` | Info |
| `dos` | Critical |
| `probe` | High |
| `r2l` | High |
| `u2r` | Critical |

### Connection tracking

`ConnectionTracker` maintains a sliding window of recent connections and computes NSL-KDD-style statistics such as connection counts, service counts, SYN/RST error rates, same-service rates, different-service rates, destination-host counts, and source-port relationships.

### PCAP analysis

Existing packet captures can be processed with:

```bash
python -m ids.capture.packet_sniffer --pcap capture.pcap
```

### Demo mode

Demo mode requires no administrative privileges or network interface:

```bash
python -m ids.capture.packet_sniffer --demo
```

It generates synthetic traffic with approximate class frequencies of 70% normal, 15% DoS, 8% Probe, 4% R2L, and 3% U2R.

## API server

The main backend is `ids/api/server.py`. It serves the dashboard from `dashboard/` and listens on:

```text
http://127.0.0.1:5000
```

### Core endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Serves the dashboard |
| `GET` | `/api/status` | Reports server and model status |
| `POST` | `/api/predict` | Classifies a connection record |
| `GET` | `/api/alerts` | Returns stored alerts |
| `GET` | `/api/stats` | Returns traffic statistics |
| `POST` | `/api/capture/start` | Starts live or demo capture |
| `POST` | `/api/capture/stop` | Stops capture |
| `POST` | `/api/pcap` | Uploads and analyzes a PCAP file |

### Explainability and analysis endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/explain` | Returns SHAP-style explanation for a prediction |
| `GET` | `/api/feature-importance` | Returns global SHAP feature ranking |
| `POST` | `/api/anomaly-score` | Calculates autoencoder anomaly score |
| `GET` | `/api/lite-model` | Returns full-versus-lite model statistics |
| `GET` | `/api/adversarial` | Returns adversarial robustness results |

### Threat-intelligence endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/narrative` | Generates an incident narrative |
| `GET` | `/api/attacker-profile` | Returns attacker behavior profiles |
| `GET` | `/api/threat-map` | Returns recent geolocated attack events |
| `GET` | `/api/mitre` | Returns MITRE ATT&CK technique counts |

### Example prediction request

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "duration": 0,
    "src_bytes": 491,
    "dst_bytes": 0,
    "protocol_type": 0
  }'
```

Missing fields are defaulted to zero by `dict_to_features()`.

### WebSocket events

The server emits:

| Event | Description |
|---|---|
| `detection` | Sent whenever a packet is classified |
| `stats_update` | Sent approximately every two seconds with current statistics |

## Dashboard

The dashboard is composed of:

- `dashboard/index.html`
- `dashboard/css/styles.css`
- `dashboard/js/dashboard.js`

It uses Chart.js for charts and Socket.IO for real-time communication.

### Dashboard panels

- Total packet counter
- Attack counter
- Attack-rate percentage
- Normal traffic counter
- Traffic timeline
- Attack distribution donut chart
- Live alert feed
- Threat-level gauge
- Attack-category progress bars
- ML model metrics
- Global cyber-threat map
- AI incident narrative
- Active adversary profiles
- SHAP feature-importance chart
- Zero-day anomaly gauge
- Full-versus-lite model comparison
- Adversarial simulation results

### Dashboard runtime flow

```text
Page load
  ↓
Initialize Chart.js components
  ↓
Fetch /api/stats and /api/alerts
  ↓
Fetch SHAP, lite-model, attacker, and adversarial data
  ↓
Connect to Socket.IO
  ↓
Receive detection and stats_update events
  ↓
Update charts, counters, alerts, and threat panels
```

The frontend expects the API at:

```javascript
const API_BASE = "http://127.0.0.1:5000";
```

## Installation and usage

### Requirements

- Python 3.10 or newer
- `pip`
- Internet access for downloading NSL-KDD
- Administrator/root privileges for live packet capture
- A compatible network interface for real-time capture

### Install

```bash
git clone https://github.com/Shivaram304/Secure-IDS.git
cd Secure-IDS
python -m venv .venv
```

Activate the environment.

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Train all models

```bash
python ids/train.py
```

This performs dataset download, preprocessing, Random Forest training, DNN training, autoencoder training, ensemble evaluation, SHAP analysis, lite-model training, and adversarial robustness simulation.

Generated artifacts are written under:

```text
data/raw/
data/processed/
models/saved/
```

### Evaluate models

```bash
python ids/evaluate.py
```

Generated reports include:

```text
reports/cm_rf.png
reports/cm_dnn.png
reports/cm_ensemble.png
reports/roc_curves.png
```

### Generate the adversarial report separately

```bash
python generate_adv_report.py
```

### Start the API and dashboard

```bash
python ids/api/server.py
```

Open the dashboard at:

```text
http://127.0.0.1:5000
```

### Run live capture

Windows example:

```bash
python -m ids.capture.packet_sniffer --interface "Ethernet"
```

Linux example:

```bash
sudo python -m ids.capture.packet_sniffer --interface eth0
```

### Analyze a PCAP file

```bash
python -m ids.capture.packet_sniffer --pcap capture.pcap
```

### Run standalone demo mode

```bash
python -m ids.capture.packet_sniffer --demo
```

## Important implementation considerations

- The classifier is trained on NSL-KDD records, while live packet capture approximates those historical features from raw packets.
- `feature_extractor.py` uses simplified protocol, service, and flag mappings for live traffic.
- Real packet capture requires Scapy and appropriate operating-system permissions.
- Demo mode generates synthetic detections and should not be treated as real network telemetry.
- Model files under `models/saved/` must exist before full model-backed inference works.
- The API includes fallback demo responses when models or analysis artifacts are unavailable.
- SHAP-related modules are present in the repository; verify that the SHAP package is installed if those modules require it in your environment.
- Some dashboard model metrics are presentation defaults unless refreshed from generated artifacts and API responses.
- The API currently enables permissive CORS and uses in-memory buffers, so additional hardening is needed for production deployment.
- Uploaded PCAP files are written to a temporary local directory and should be handled with filename validation and cleanup in production.

## Generated artifacts

The following directories are generated or populated during training and runtime:

```text
data/raw/              Downloaded NSL-KDD files
data/processed/         NumPy training and test arrays
models/saved/            Trained models and metadata
reports/                 Evaluation charts
tmp_pcap/                Uploaded PCAP files during API analysis
```

These files may be large and environment-specific. The repository currently includes placeholder files for empty generated directories rather than the complete trained dataset and model outputs.

## Troubleshooting

### Models are not found

Run the training pipeline first:

```bash
python ids/train.py
```

### Dashboard shows OFFLINE

Start the API server and verify that port 5000 is available:

```bash
python ids/api/server.py
```

Then open `http://127.0.0.1:5000`.

### Live capture fails

Use demo mode to verify the application first:

```bash
python -m ids.capture.packet_sniffer --demo
```

For real capture, confirm the interface name and run with the required administrator/root privileges.

### SHAP or novelty panels are empty

Run the full training pipeline so that SHAP, autoencoder, lite-model, and adversarial artifacts are generated:

```bash
python ids/train.py
```

### PCAP analysis fails

Confirm that Scapy is installed and that the uploaded file is a valid PCAP. Also verify that the API process can read and write the temporary PCAP directory.

## Recommended future improvements

- Add automated tests for preprocessing, feature extraction, API endpoints, and model loading.
- Add a pinned dependency lock file for reproducible environments.
- Add explicit SHAP dependency management if required by the explainability modules.
- Add authentication and authorization to the API.
- Replace permissive CORS with an allowlist.
- Add persistent storage for alerts and statistics.
- Add structured logging and health checks.
- Validate and safely clean up uploaded PCAP files.
- Add Docker and production deployment configuration.
- Add data and model versioning.
- Calibrate and continuously validate live-capture feature extraction against the training feature distribution.
- Replace static dashboard metrics with values loaded entirely from the trained model reports.
