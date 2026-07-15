"""
NSL-KDD Dataset Downloader
Downloads the NSL-KDD dataset (train + test) from the Canadian Institute for
Cybersecurity mirror and saves it locally.
"""

import os
import urllib.request
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm

# ── Dataset URLs ──────────────────────────────────────────────────────────────
NSL_KDD_TRAIN_URL = (
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
)
NSL_KDD_TEST_URL = (
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"

# ── NSL-KDD Column Names (41 features + label + difficulty) ──────────────────
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


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest: Path) -> None:
    """Download a file with a progress bar."""
    if dest.exists():
        print(f"  [SKIP] {dest.name} already exists.")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [DOWN] Downloading {dest.name} ...")
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1,
                              desc=dest.name) as t:
        urllib.request.urlretrieve(url, filename=dest,
                                   reporthook=t.update_to)
    print(f"  [ OK ] Saved to {dest}")


def download_nsl_kdd() -> tuple[Path, Path]:
    """Download NSL-KDD train and test sets. Returns (train_path, test_path)."""
    train_path = DATA_DIR / "KDDTrain+.txt"
    test_path  = DATA_DIR / "KDDTest+.txt"

    print("\n+==========================================+")
    print("||      NSL-KDD Dataset Downloader        ||")
    print("+==========================================+\n")

    download_file(NSL_KDD_TRAIN_URL, train_path)
    download_file(NSL_KDD_TEST_URL,  test_path)

    print(f"\n✓ Dataset ready at: {DATA_DIR}\n")
    return train_path, test_path


if __name__ == "__main__":
    download_nsl_kdd()
