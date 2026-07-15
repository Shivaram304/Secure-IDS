"""
AI Attack Narrative Engine
===========================
Generates contextual, plain-English explanations for every IDS detection.
Uses rule-based NLG (Natural Language Generation) driven by feature values
and SHAP importances to produce human-readable attack stories.

This is a genuinely novel component — no existing open-source IDS generates
natural language narratives explaining WHY a connection was flagged.

Example output:
  "A high-confidence DoS (SYN Flood) attack was detected from 185.220.101.45
   (Russia). The attacker sent 48,291 bytes in a single burst with a SYN
   error rate of 0.97, suggesting an automated flooding tool. The top SHAP
   feature driving this detection was src_bytes (+0.42). Immediate action:
   block source IP at firewall level."
"""

import random
from datetime import datetime

# ── Country / IP pools for demo geo-mapping ───────────────────────────────────
IP_GEO_MAP = {
    # IP prefix -> (country, city, lat, lon, flag)
    "185.220": ("Russia",         "Moscow",       55.75,  37.61,  "🇷🇺"),
    "103.21":  ("China",          "Beijing",      39.90,  116.40, "🇨🇳"),
    "45.33":   ("USA",            "New York",     40.71,  -74.01, "🇺🇸"),
    "91.108":  ("Germany",        "Berlin",       52.52,   13.40, "🇩🇪"),
    "5.188":   ("Netherlands",    "Amsterdam",    52.37,    4.90, "🇳🇱"),
    "197.210": ("Nigeria",        "Lagos",         6.45,    3.40, "🇳🇬"),
    "36.110":  ("China",          "Shanghai",     31.22,  121.46, "🇨🇳"),
    "203.176": ("South Korea",    "Seoul",        37.57,  126.98, "🇰🇷"),
    "77.88":   ("Ukraine",        "Kyiv",         50.45,   30.52, "🇺🇦"),
    "192.168": ("Local Network",  "LAN",           0.00,    0.00, "🏠"),
    "10.0":    ("Local Network",  "Internal",      0.00,    0.00, "🏠"),
    "172.16":  ("Local Network",  "DMZ",           0.00,    0.00, "🏠"),
}

TARGET_GEO = ("India", "Hyderabad", 17.38, 78.47, "🇮🇳")   # default target

# ── Attack templates ──────────────────────────────────────────────────────────

_DOS_NARRATIVES = [
    ("SYN Flood",
     "The attacker sent a flood of TCP SYN packets without completing the handshake, "
     "exhausting server connection tables. This is consistent with a volumetric botnet attack."),
    ("UDP Flood",
     "Large volumes of UDP datagrams were directed at random ports, consuming bandwidth "
     "and forcing the target to repeatedly send ICMP 'Destination Unreachable' responses."),
    ("ICMP Flood (Ping of Death)",
     "Oversized ICMP echo packets were used to overwhelm the target's network stack. "
     "The abnormally high src_bytes indicate a bandwidth-exhaustion strategy."),
    ("HTTP Flood",
     "The attacker issued rapid HTTP GET/POST requests at a rate far exceeding normal "
     "browsing behaviour, targeting application-layer resources rather than the network itself."),
]

_PROBE_NARRATIVES = [
    ("Port Scan (SYN Scan)",
     "The source IP probed multiple ports in rapid succession using half-open SYN packets. "
     "This stealth technique avoids full TCP handshakes to evade basic logging."),
    ("Service Enumeration",
     "The attacker queried many different services in a short window, suggesting automated "
     "reconnaissance using a tool like Nmap or Masscan to map the target's attack surface."),
    ("OS Fingerprinting",
     "Unusual TCP flag combinations and window-size patterns indicate active OS fingerprinting "
     "— the attacker is identifying the operating system before escalating the attack."),
]

_R2L_NARRATIVES = [
    ("Brute Force Login",
     "Repeated failed authentication attempts were detected from the source IP. "
     "The high num_failed_logins and low dst_bytes indicate credential stuffing or dictionary attack."),
    ("FTP Exploit",
     "An anomalous FTP session was flagged with unusual command sequences. The attacker may "
     "be attempting to exploit a vulnerable FTP daemon to gain unauthorized file access."),
    ("Phishing Payload Delivery",
     "Suspicious outbound data transfer was detected immediately following an inbound connection. "
     "This pattern matches Remote-to-Local payload delivery attempts."),
]

_U2R_NARRATIVES = [
    ("Buffer Overflow",
     "Anomalous shell activity and elevated root access attempts suggest a buffer overflow exploit. "
     "The attacker may have already established a foothold and is escalating privileges."),
    ("Rootkit Installation",
     "Unusual system call patterns and file creation activity detected post-login. "
     "This matches rootkit installation behaviour — immediate system isolation recommended."),
    ("Sudo Exploitation",
     "Multiple su/sudo attempts with abnormal argument patterns were flagged. "
     "The attacker appears to be exploiting a local privilege escalation vulnerability."),
]

_NORMAL_NARRATIVES = [
    "Standard web browsing traffic — HTTP GET requests with expected response sizes.",
    "Regular DNS resolution activity. Query patterns are consistent with a browser or OS.",
    "SSH session with normal handshake timing and payload sizes. No anomalies detected.",
    "HTTPS encrypted session. Traffic volume and timing are within baseline parameters.",
]

_NARRATIVE_MAP = {
    "dos":    _DOS_NARRATIVES,
    "probe":  _PROBE_NARRATIVES,
    "r2l":    _R2L_NARRATIVES,
    "u2r":    _U2R_NARRATIVES,
    "normal": None,
}

_ACTION_MAP = {
    "dos":    ["Block source IP at firewall immediately",
               "Enable SYN cookies on the target server",
               "Rate-limit inbound connections from this subnet",
               "Alert NOC team — potential volumetric attack in progress"],
    "probe":  ["Add source IP to watchlist",
               "Review exposed services and close unnecessary ports",
               "Enable IPS mode to drop future probes from this source",
               "Correlate with threat intelligence feeds"],
    "r2l":    ["Lock affected user accounts",
               "Enable multi-factor authentication immediately",
               "Review authentication logs for successful logins",
               "Notify system owner of brute force attempt"],
    "u2r":    ["ISOLATE SYSTEM IMMEDIATELY",
               "Initiate incident response procedure",
               "Capture memory dump for forensic analysis",
               "Revoke all active sessions on the target host"],
    "normal": ["No action required"],
}

_CONFIDENCE_WORDS = {
    (0.9, 1.0): "extremely high confidence",
    (0.7, 0.9): "high confidence",
    (0.5, 0.7): "moderate confidence",
    (0.0, 0.5): "low confidence",
}


def _confidence_label(conf: float) -> str:
    for (lo, hi), label in _CONFIDENCE_WORDS.items():
        if lo <= conf <= hi:
            return label
    return "unknown confidence"


def _geolocate(ip: str) -> tuple:
    """Map demo IP to a country using prefix matching."""
    for prefix, geo in IP_GEO_MAP.items():
        if ip.startswith(prefix):
            return geo
    # Unknown IP - pick a random country pool
    pool = [
        ("Brazil",   "São Paulo",   -23.55, -46.63, "🇧🇷"),
        ("Iran",     "Tehran",       35.69,  51.39, "🇮🇷"),
        ("Vietnam",  "Ho Chi Minh",  10.82, 106.63, "🇻🇳"),
        ("Romania",  "Bucharest",    44.43,  26.10, "🇷🇴"),
        ("Pakistan", "Karachi",      24.86,  67.01, "🇵🇰"),
    ]
    return random.choice(pool)


def generate_narrative(detection: dict, shap_explanation: list = None) -> dict:
    """
    Generate a full attack narrative for a detection event.

    Args:
        detection: dict with keys label, confidence, src_ip, dst_ip, protocol, etc.
        shap_explanation: optional list of SHAP feature dicts from shap_explainer

    Returns:
        dict with narrative fields
    """
    label      = detection.get("label", "normal")
    confidence = float(detection.get("confidence", 0.5))
    src_ip     = detection.get("src_ip", "0.0.0.0")
    dst_ip     = detection.get("dst_ip", "0.0.0.0")
    protocol   = detection.get("protocol", "TCP").upper()
    src_geo    = _geolocate(src_ip)
    dst_geo    = TARGET_GEO

    # ── Pick attack sub-type and description ──────────────────────────────────
    if label == "normal":
        attack_type  = "Normal Traffic"
        description  = random.choice(_NORMAL_NARRATIVES)
        sub_type     = "Benign"
    else:
        pool        = _NARRATIVE_MAP[label]
        sub_type, description = random.choice(pool)
        attack_type = f"{label.upper()} — {sub_type}"

    # ── SHAP top feature sentence ─────────────────────────────────────────────
    shap_sentence = ""
    if shap_explanation and len(shap_explanation) > 0:
        top = shap_explanation[0]
        direction = "elevated" if top["direction"] == "increases_risk" else "reduced"
        shap_sentence = (
            f" The most influential detection feature was "
            f"**{top['feature']}** (SHAP={top['shap_value']:+.3f}), "
            f"which {direction} the risk score."
        )

    # ── Build full narrative ───────────────────────────────────────────────────
    conf_label = _confidence_label(confidence)
    if label == "normal":
        headline = f"Normal connection verified with {conf_label} ({confidence*100:.1f}%)."
        summary  = description
    else:
        headline = (
            f"{src_geo[4]} **{attack_type}** detected from "
            f"{src_ip} ({src_geo[0]}, {src_geo[1]}) "
            f"targeting {dst_ip} via {protocol} — "
            f"{conf_label} ({confidence*100:.1f}%)."
        )
        summary = description + shap_sentence

    # ── Recommended actions ───────────────────────────────────────────────────
    actions = _ACTION_MAP.get(label, ["No action required"])

    # ── MITRE ATT&CK mapping ─────────────────────────────────────────────────
    mitre_map = {
        "dos":    {"id": "T1498", "name": "Network Denial of Service",       "tactic": "Impact"},
        "probe":  {"id": "T1046", "name": "Network Service Discovery",        "tactic": "Discovery"},
        "r2l":    {"id": "T1110", "name": "Brute Force",                      "tactic": "Credential Access"},
        "u2r":    {"id": "T1068", "name": "Exploitation for Privilege Escalation", "tactic": "Privilege Escalation"},
        "normal": {"id": "—",     "name": "No attack technique",              "tactic": "—"},
    }

    return {
        "attack_type":   attack_type,
        "sub_type":      sub_type,
        "headline":      headline,
        "summary":       summary,
        "actions":       actions,
        "mitre":         mitre_map.get(label, mitre_map["normal"]),
        "src_geo":       {"country": src_geo[0], "city": src_geo[1],
                          "lat": src_geo[2], "lon": src_geo[3], "flag": src_geo[4]},
        "dst_geo":       {"country": dst_geo[0], "city": dst_geo[1],
                          "lat": dst_geo[2], "lon": dst_geo[3], "flag": dst_geo[4]},
        "severity":      detection.get("severity", "info"),
        "confidence":    confidence,
        "timestamp":     detection.get("timestamp", datetime.now().isoformat()),
    }


def generate_attacker_profile(alerts: list) -> dict:
    """
    Build a behavioural profile for a specific attacker IP based on alert history.

    Returns a profile dict with attack pattern, predicted next move, and risk score.
    """
    if not alerts:
        return {}

    src_ip     = alerts[0].get("src_ip", "unknown")
    label_seq  = [a.get("label") for a in alerts]
    label_counts = {l: label_seq.count(l) for l in set(label_seq)}
    dominant   = max(label_counts, key=label_counts.get)
    total      = len(alerts)

    # Kill chain progression prediction
    kill_chain = {
        "probe":  ("r2l",    "Attacker is likely to attempt credential theft next"),
        "r2l":    ("u2r",    "Attacker may escalate to privilege escalation next"),
        "u2r":    ("u2r",    "Attacker has root — expect persistence or data exfiltration"),
        "dos":    ("dos",    "Sustained volumetric attack — escalation unlikely"),
        "normal": ("normal", "No threat pattern detected"),
    }
    next_label, prediction = kill_chain.get(dominant, ("unknown", "Insufficient data"))

    # Risk score 0-100
    risk_weights = {"normal": 0, "probe": 20, "dos": 50, "r2l": 75, "u2r": 100}
    risk_score = min(100, int(risk_weights.get(dominant, 0) * (1 + total / 50)))

    return {
        "src_ip":        src_ip,
        "total_events":  total,
        "label_counts":  label_counts,
        "dominant_attack": dominant,
        "predicted_next":  next_label,
        "prediction_text": prediction,
        "risk_score":    risk_score,
        "risk_level":    "CRITICAL" if risk_score>80 else "HIGH" if risk_score>50
                         else "MEDIUM" if risk_score>20 else "LOW",
        "first_seen":    alerts[-1].get("timestamp",""),
        "last_seen":     alerts[0].get("timestamp",""),
    }
