"""
Feature Extractor
Maps raw packet fields / connection records to the 41-feature vector
expected by the NSL-KDD-trained models.
Supports both live Scapy packets and batch CSV/dict input.
"""

import numpy as np
import time
from collections import defaultdict
from typing import Optional

# ── Feature order (must match training data column order) ─────────────────────
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

# Protocol & service code maps (simplified NSL-KDD encoding)
PROTOCOL_MAP = {"tcp": 0, "udp": 1, "icmp": 2}
FLAG_MAP = {
    "SF": 0, "S0": 1, "S1": 2, "S2": 3, "S3": 4,
    "REJ": 5, "RSTO": 6, "RSTR": 7, "RSTOS0": 8,
    "OTH": 9, "SH": 10
}
SERVICE_MAP = {
    "http": 0, "ftp_data": 1, "smtp": 2, "ssh": 3, "dns": 4,
    "ftp": 5, "pop_3": 6, "telnet": 7, "finger": 8, "domain_u": 9,
    "auth": 10, "login": 11, "imap4": 12, "other": 13,
}


class ConnectionTracker:
    """
    Maintains a sliding window of recent connections per host/service
    to compute the statistical features (count, srv_count, error rates, etc.)
    required by NSL-KDD.
    """

    def __init__(self, window: int = 100):
        self.window = window
        self._buf: list[dict] = []

    def add(self, record: dict) -> None:
        self._buf.append(record)
        if len(self._buf) > self.window:
            self._buf.pop(0)

    def compute_stats(self, dst_host: str, service: str) -> dict:
        """Compute last-100-connection statistics for current host/service."""
        recent = self._buf[-self.window:]
        same_host   = [r for r in recent if r.get("dst_host") == dst_host]
        same_srv    = [r for r in recent if r.get("service") == service]
        n = len(recent) or 1
        nh = len(same_host) or 1

        def rate(lst, key): return sum(1 for r in lst if r.get(key)) / max(len(lst), 1)

        return {
            "count":                    len(same_host),
            "srv_count":                len(same_srv),
            "serror_rate":              rate(same_host, "syn_error"),
            "srv_serror_rate":          rate(same_srv,  "syn_error"),
            "rerror_rate":              rate(same_host, "rst_error"),
            "srv_rerror_rate":          rate(same_srv,  "rst_error"),
            "same_srv_rate":            rate(same_host, "same_service"),
            "diff_srv_rate":            1 - rate(same_host, "same_service"),
            "srv_diff_host_rate":       rate(same_srv,  "diff_host"),
            "dst_host_count":           min(len(same_host), 255),
            "dst_host_srv_count":       min(len(same_srv),  255),
            "dst_host_same_srv_rate":   rate(same_host, "same_service"),
            "dst_host_diff_srv_rate":   1 - rate(same_host, "same_service"),
            "dst_host_same_src_port_rate": rate(same_host, "same_src_port"),
            "dst_host_srv_diff_host_rate": rate(same_srv, "diff_host"),
            "dst_host_serror_rate":     rate(same_host, "syn_error"),
            "dst_host_srv_serror_rate": rate(same_srv,  "syn_error"),
            "dst_host_rerror_rate":     rate(same_host, "rst_error"),
            "dst_host_srv_rerror_rate": rate(same_srv,  "rst_error"),
        }


def port_to_service(port: int) -> str:
    """Map common TCP/UDP port numbers to service names."""
    _map = {
        80: "http", 443: "http", 8080: "http",
        21: "ftp", 20: "ftp_data",
        22: "ssh", 23: "telnet",
        25: "smtp", 110: "pop_3", 143: "imap4",
        53: "dns", 79: "finger",
        113: "auth", 513: "login",
    }
    return _map.get(port, "other")


def packet_to_features(pkt, tracker: ConnectionTracker,
                        start_time: float,
                        scaler=None) -> Optional[np.ndarray]:
    """
    Convert a Scapy packet to a 41-feature vector.
    Returns None if the packet cannot be parsed.
    """
    try:
        from scapy.all import IP, TCP, UDP, ICMP
        if not pkt.haslayer(IP):
            return None

        ip   = pkt[IP]
        proto_num = ip.proto          # 6=TCP 17=UDP 1=ICMP
        proto_str = {6: "tcp", 17: "udp", 1: "icmp"}.get(proto_num, "tcp")

        src_bytes  = len(pkt)
        dst_bytes  = 0
        duration   = round(time.time() - start_time, 3)
        land       = int(ip.src == ip.dst)
        wrong_frag = int(ip.flags.MF) + int(bool(ip.frag))
        urgent     = 0
        dst_port   = 0
        flag_str   = "OTH"

        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            dst_port = tcp.dport
            flags = tcp.flags
            if flags & 0x02:  flag_str = "S0"
            if flags & 0x12:  flag_str = "SF"
            if flags & 0x04:  flag_str = "REJ"
            urgent = int(bool(tcp.urgptr))

        elif pkt.haslayer(UDP):
            udp = pkt[UDP]
            dst_port = udp.dport

        service = port_to_service(dst_port)
        dst_host = ip.dst

        record = {
            "dst_host": dst_host, "service": service,
            "protocol": proto_str, "syn_error": flag_str in ("S0", "S1"),
            "rst_error": flag_str in ("REJ", "RSTO"),
            "same_service": True, "diff_host": False, "same_src_port": False,
        }
        tracker.add(record)
        stats = tracker.compute_stats(dst_host, service)

        feature_vec = [
            duration,
            PROTOCOL_MAP.get(proto_str, 0),
            SERVICE_MAP.get(service, 13),
            FLAG_MAP.get(flag_str, 9),
            src_bytes, dst_bytes, land, wrong_frag, urgent,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   # content/shell features = 0
            stats["count"], stats["srv_count"],
            stats["serror_rate"], stats["srv_serror_rate"],
            stats["rerror_rate"], stats["srv_rerror_rate"],
            stats["same_srv_rate"], stats["diff_srv_rate"],
            stats["srv_diff_host_rate"],
            stats["dst_host_count"], stats["dst_host_srv_count"],
            stats["dst_host_same_srv_rate"], stats["dst_host_diff_srv_rate"],
            stats["dst_host_same_src_port_rate"],
            stats["dst_host_srv_diff_host_rate"],
            stats["dst_host_serror_rate"], stats["dst_host_srv_serror_rate"],
            stats["dst_host_rerror_rate"], stats["dst_host_srv_rerror_rate"],
        ]

        arr = np.array(feature_vec, dtype=np.float32).reshape(1, -1)
        if scaler is not None:
            arr = scaler.transform(arr)
        return arr

    except Exception:
        return None


def dict_to_features(record: dict, scaler=None) -> np.ndarray:
    """
    Convert a dict of feature name → value (already numeric) to a scaled
    numpy array ready for the model.  Missing fields default to 0.
    """
    vec = np.array([record.get(f, 0) for f in FEATURE_NAMES], dtype=np.float32)
    arr = vec.reshape(1, -1)
    if scaler is not None:
        arr = scaler.transform(arr)
    return arr
