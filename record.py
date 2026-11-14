"""Record a labeled CSI session to data/<label>.csv for later training.

Usage:
    python record.py empty --port /dev/ttyACM0 --seconds 180
    python record.py walking --port /dev/ttyACM0 --seconds 180

Run this a few separate times per label (a few minutes each) rather than once
for a long time -- train.py splits train/test by recording session, so having
several distinct sessions per label is what makes that split meaningful.
"""

import argparse
import os
import time

import pandas as pd

from features import (
    DEFAULT_STEP_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    FEATURE_NAMES,
    extract_features,
    sliding_windows,
)
from read_csi import DEFAULT_BAUD, read_csi

LABELS = ("empty", "standing", "walking")


def record_session(
    label: str,
    seconds: int,
    port: str,
    baud: int = DEFAULT_BAUD,
    out_dir: str = "data",
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    step_seconds: float = DEFAULT_STEP_SECONDS,
    session_id: str = None,
) -> None:
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}, got {label!r}")

    os.makedirs(out_dir, exist_ok=True)
    session_id = session_id or f"{label}_{int(time.time())}"
    out_path = os.path.join(out_dir, f"{label}.csv")
    write_header = not os.path.exists(out_path)
    columns = ["label", "session_id", "t"] + list(FEATURE_NAMES)

    deadline = time.time() + seconds
    packets_seen = 0
    rows_written = 0

    def timed_packets():
        nonlocal packets_seen
        for packet in read_csi(port, baud):
            packets_seen += 1
            if packet[0] >= deadline:
                return
            yield packet

    windows = sliding_windows(timed_packets(), window_seconds, step_seconds)
    for window in windows:
        feats = extract_features(window)
        if feats["n_packets"] == 0:
            continue

        row = {"label": label, "session_id": session_id, "t": time.time()}
        row.update(feats)
        pd.DataFrame([row], columns=columns).to_csv(
            out_path, mode="a", header=write_header, index=False
        )
        write_header = False
        rows_written += 1

    print(f"session {session_id}: wrote {rows_written} rows from {packets_seen} packets to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Record a labeled CSI session for training.")
    parser.add_argument("label", choices=LABELS)
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyACM0 or COM5")
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--session-id", default=None)
    args = parser.parse_args()

    record_session(
        args.label,
        args.seconds,
        args.port,
        baud=args.baud,
        out_dir=args.out_dir,
        session_id=args.session_id,
    )


if __name__ == "__main__":
    main()
