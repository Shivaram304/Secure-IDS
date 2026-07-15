"""
Packet Sniffer – Live & PCAP Mode
Captures network packets via Scapy, extracts features, runs the ensemble
model, and emits detections via a callback (used by the API server).

Usage (standalone):
  python -m ids.capture.packet_sniffer --interface eth0 --count 0
  python -m ids.capture.packet_sniffer --pcap path/to/file.pcap
"""

import os
import sys
import time
import argparse
import threading
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import joblib

# Suppress TF noise
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ids.capture.feature_extractor import (
    packet_to_features, ConnectionTracker, FEATURE_NAMES
)

MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "saved"
CLASS_NAMES = ["normal", "dos", "probe", "r2l", "u2r"]

SEVERITY = {
    "normal": "info",
    "dos":    "critical",
    "probe":  "high",
    "r2l":    "high",
    "u2r":    "critical",
}


class PacketSniffer:
    """
    Thread-safe live packet sniffer + IDS classifier.

    Parameters
    ----------
    on_detection : callable(dict) -> None
        Called for every classified packet with a detection record.
    interface : str, optional
        Network interface (None = default interface).
    count : int
        Packets to capture (0 = infinite).
    threshold : float
        Minimum confidence to emit an alert (0-1).
    models_loaded : bool
        If False, runs in demo mode with random predictions.
    """

    def __init__(self,
                 on_detection: Callable[[dict], None] = None,
                 interface: str = None,
                 count: int = 0,
                 threshold: float = 0.0,
                 demo_mode: bool = False):
        self.on_detection = on_detection or (lambda x: print(x))
        self.interface    = interface
        self.count        = count
        self.threshold    = threshold
        self.demo_mode    = demo_mode
        self._stop_event  = threading.Event()
        self._tracker     = ConnectionTracker(window=100)
        self._start_time  = time.time()

        # Load models
        self.ensemble = None
        self.scaler   = None
        if not demo_mode:
            try:
                import tensorflow as tf
                rf  = joblib.load(MODELS_DIR / "random_forest.pkl")
                dnn = tf.keras.models.load_model(
                    str(MODELS_DIR / "dnn_model.keras"))
                self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
                from ids.models.ensemble import EnsembleIDS
                self.ensemble = EnsembleIDS(rf_model=rf, dnn_model=dnn)
                print("[SNIFFER] Models loaded successfully.")
            except Exception as e:
                print(f"[SNIFFER] Could not load models ({e}). "
                      "Switching to demo mode.")
                self.demo_mode = True

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start live capture (blocking). Call in a thread for async use."""
        if self.demo_mode:
            self._demo_loop()
            return
        try:
            from scapy.all import sniff
            print(f"[SNIFFER] Starting live capture on "
                  f"interface={self.interface or 'default'}, "
                  f"count={self.count or '∞'}")
            sniff(iface=self.interface,
                  prn=self._handle_packet,
                  count=self.count,
                  store=False,
                  stop_filter=lambda _: self._stop_event.is_set())
        except Exception as e:
            print(f"[SNIFFER] Scapy error: {e}. Falling back to demo mode.")
            self._demo_loop()

    def analyze_pcap(self, pcap_path: str) -> list[dict]:
        """Analyse an existing PCAP file and return all detections."""
        from scapy.all import rdpcap
        packets   = rdpcap(pcap_path)
        detections = []
        for pkt in packets:
            det = self._classify_packet(pkt)
            if det:
                detections.append(det)
        return detections

    def stop(self) -> None:
        self._stop_event.set()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _handle_packet(self, pkt) -> None:
        det = self._classify_packet(pkt)
        if det and det["confidence"] >= self.threshold:
            self.on_detection(det)

    def _classify_packet(self, pkt) -> Optional[dict]:
        """Extract features and classify; returns detection dict or None."""
        features = packet_to_features(
            pkt, self._tracker, self._start_time, self.scaler)
        if features is None:
            return None

        proba = self.ensemble.predict_proba(features)[0]
        idx   = int(np.argmax(proba))
        label = CLASS_NAMES[idx]

        try:
            from scapy.all import IP, TCP, UDP
            src_ip = pkt[IP].src if pkt.haslayer("IP") else "unknown"
            dst_ip = pkt[IP].dst if pkt.haslayer("IP") else "unknown"
            proto  = "TCP" if pkt.haslayer(TCP) else \
                     "UDP" if pkt.haslayer(UDP) else "OTHER"
        except Exception:
            src_ip = dst_ip = "unknown"; proto = "OTHER"

        return {
            "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
            "src_ip":        src_ip,
            "dst_ip":        dst_ip,
            "protocol":      proto,
            "label":         label,
            "severity":      SEVERITY[label],
            "confidence":    round(float(proba[idx]), 4),
            "probabilities": {c: round(float(p), 4)
                              for c, p in zip(CLASS_NAMES, proba)},
        }

    def _demo_loop(self) -> None:
        """Generate synthetic detections for demo / testing."""
        import random
        print("[SNIFFER] Running in DEMO mode (simulated traffic).")
        demo_ips = ["192.168.1.{i}".format(i=i) for i in range(1, 20)]
        weights  = [0.70, 0.15, 0.08, 0.04, 0.03]    # roughly realistic

        while not self._stop_event.is_set():
            label = random.choices(CLASS_NAMES, weights=weights)[0]
            conf  = round(random.uniform(0.70, 0.99), 4)
            proba = [round(random.uniform(0, 1 - conf) / 4, 4)
                     for _ in CLASS_NAMES]
            proba[CLASS_NAMES.index(label)] = conf

            detection = {
                "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
                "src_ip":        random.choice(demo_ips),
                "dst_ip":        random.choice(demo_ips),
                "protocol":      random.choice(["TCP", "UDP", "ICMP"]),
                "label":         label,
                "severity":      SEVERITY[label],
                "confidence":    conf,
                "probabilities": dict(zip(CLASS_NAMES, proba)),
            }
            self.on_detection(detection)
            time.sleep(random.uniform(0.5, 2.0))   # realistic cadence


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDS Packet Sniffer")
    parser.add_argument("--interface", "-i", default=None,
                        help="Network interface (default: system default)")
    parser.add_argument("--pcap", "-p", default=None,
                        help="Analyse a PCAP file instead of live capture")
    parser.add_argument("--count", "-c", type=int, default=0,
                        help="Number of packets (0=unlimited)")
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode with synthetic data")
    args = parser.parse_args()

    def print_detection(det):
        label = det["label"].upper()
        sev   = det["severity"].upper()
        conf  = det["confidence"]
        ts    = det["timestamp"]
        src   = det["src_ip"]
        print(f"[{ts}] [{sev:8s}] {label:6s} | {src:15s} | conf={conf:.2%}")

    sniffer = PacketSniffer(
        on_detection=print_detection,
        interface=args.interface,
        count=args.count,
        demo_mode=args.demo,
    )

    if args.pcap:
        results = sniffer.analyze_pcap(args.pcap)
        for r in results:
            print_detection(r)
    else:
        sniffer.start()
