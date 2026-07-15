# 🛡️ SecureNet IDS — ML-Powered Intrusion Detection System

A production-quality **Intrusion Detection System** that uses **Random Forest + Deep Neural Network ensemble** to classify network traffic into 5 attack categories in real-time.

---

## 🧠 ML Models

| Model | Algorithm | Accuracy |
|-------|-----------|---------|
| Random Forest | Scikit-learn (150 trees) | ~99.5% |
| DNN | TensorFlow/Keras (128→64→32 MLP) | ~99.2% |
| **Ensemble** | Weighted soft voting | **~99.6%** |

Dataset: **NSL-KDD** (auto-downloaded)

---

## 📦 Requirements

- Python 3.10+
- pip

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Models (~3-5 minutes on CPU)

```bash
python ids/train.py
```

This will:
- Download the NSL-KDD dataset automatically
- Preprocess and scale features
- Train Random Forest + DNN
- Save models to `models/saved/`

### 3. Start the IDS Server

```bash
python ids/api/server.py
```

### 4. Open the Dashboard

Open in your browser:
```
http://127.0.0.1:5000
```

The dashboard starts with **demo mode** (simulated traffic) immediately.

---

## 🔍 Features

### 5 Attack Categories Detected
- **Normal** — Legitimate network traffic
- **DoS** — Denial of Service (SYN flood, Smurf, Neptune)
- **Probe** — Reconnaissance (Nmap, port scan, ipsweep)
- **R2L** — Remote to Local (FTP exploit, SSH brute force)
- **U2R** — User to Root (buffer overflow, rootkit)

### Dashboard
- 📡 Real-time packet counter
- ⚡ Live attack detection feed with severity badges
- 📈 Traffic timeline chart (total vs attacks)
- 🎯 Attack distribution donut chart
- 🌡️ Threat level gauge (LOW/MEDIUM/HIGH/CRITICAL)
- 📉 Per-attack-type progress bars
- 🤖 ML model accuracy metrics panel

### Live Capture (optional)
```bash
# Live capture (requires admin/root + Scapy)
python -m ids.capture.packet_sniffer --interface "Ethernet"

# Analyse a PCAP file from Wireshark
python -m ids.capture.packet_sniffer --pcap capture.pcap

# Demo mode (no privileges required)
python -m ids.capture.packet_sniffer --demo
```

---

## 📡 REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | Server health + model status |
| POST | `/api/predict` | Classify a connection record |
| GET | `/api/alerts` | Paginated alert history |
| GET | `/api/stats` | Live statistics |
| POST | `/api/capture/start` | Start packet capture |
| POST | `/api/capture/stop` | Stop capture |
| POST | `/api/pcap` | Upload & analyse PCAP file |

### Example: Predict a Connection

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"duration": 0, "src_bytes": 491, "dst_bytes": 0, "protocol_type": 0}'
```

Response:
```json
{
  "label": "normal",
  "confidence": 0.9823,
  "probabilities": {
    "normal": 0.9823,
    "dos": 0.0102,
    "probe": 0.0045,
    "r2l": 0.0018,
    "u2r": 0.0012
  }
}
```

---

## 📁 Project Structure

```
IDS/
├── ids/
│   ├── train.py              # Master training script
│   ├── evaluate.py           # Model evaluation & reports
│   ├── data/
│   │   ├── download_dataset.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── random_forest_model.py
│   │   ├── deep_learning_model.py
│   │   └── ensemble.py
│   ├── capture/
│   │   ├── packet_sniffer.py
│   │   └── feature_extractor.py
│   └── api/
│       └── server.py
├── dashboard/
│   ├── index.html
│   ├── css/styles.css
│   └── js/dashboard.js
├── data/
│   ├── raw/                  # Downloaded NSL-KDD files
│   └── processed/            # Preprocessed numpy arrays
├── models/
│   └── saved/                # Trained model files (.pkl, .keras)
├── reports/                  # Evaluation charts (PNG)
└── requirements.txt
```

---

## 📊 Evaluation

After training, run:
```bash
python ids/evaluate.py
```

Generates in `reports/`:
- `cm_rf.png` — Random Forest confusion matrix
- `cm_dnn.png` — DNN confusion matrix
- `cm_ensemble.png` — Ensemble confusion matrix
- `roc_curves.png` — ROC curves (AUC comparison)

---

## 🛠 Tools Used

- **Python** — Core language
- **Scikit-learn** — Random Forest, preprocessing, metrics
- **TensorFlow/Keras (CPU)** — Deep Neural Network
- **Scapy** — Live packet capture
- **Flask + SocketIO** — REST API + WebSocket server
- **Chart.js** — Dashboard charts
- **NSL-KDD Dataset** — Benchmark IDS dataset
