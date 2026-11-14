"""Generate extra training rows by bootstrap-resampling real recorded CSI
packets (new random combinations of real packets, plus small jitter), to
give the models more windows to train on.

This does NOT invent new motion signal -- it only recombines noise already
present in the real recordings. Synthetic sessions are tagged with a
"synth_" session_id prefix, which train.py's session_split excludes from
ever being chosen as held-out test data -- reported accuracy still reflects
only real, independently-recorded sessions.

Usage:
    python augment.py --sessions-per-label 5 --packets-per-session 2000
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from features import (
    DEFAULT_STEP_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    extract_features,
    sliding_windows,
)
from read_csi import values_to_amplitudes
from train import FEATURE_CSV_COLUMNS, LABEL_NAMES, is_raw_packet_csv

JITTER_AMPLITUDE_STD = 0.03  # multiplicative noise on amplitudes (3%)
JITTER_RSSI_STD = 0.5  # dB
DEFAULT_RATE_HZ = 95.0


def load_real_packets_for_label(data_dir: str, label: str):
    """Pool every real (non-synthetic) packet recorded for this label."""
    packets = []
    for path in sorted(glob.glob(os.path.join(data_dir, f"{label}_*.csv"))):
        if not is_raw_packet_csv(path):
            continue  # skip already-windowed/synthetic feature CSVs
        raw = pd.read_csv(path)
        value_cols = [c for c in raw.columns if c.startswith("v")]
        for _, row in raw.iterrows():
            amplitudes = values_to_amplitudes(row[value_cols].to_numpy(dtype=int))
            packets.append((float(row["rssi"]), amplitudes))
    return packets


def make_synthetic_stream(packets, n_packets: int, rate_hz: float, seed: int):
    """Bootstrap-resample real packets (with small jitter) into a fake,
    evenly-spaced packet stream, ready for the normal windowing pipeline."""
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, len(packets), size=n_packets)
    stream = []
    for i, idx in enumerate(indices):
        rssi, amplitudes = packets[idx]
        jittered_amp = amplitudes * rng.normal(1.0, JITTER_AMPLITUDE_STD, size=amplitudes.shape)
        jittered_amp = np.clip(jittered_amp, 0, None)
        jittered_rssi = rssi + rng.normal(0, JITTER_RSSI_STD)
        stream.append((i / rate_hz, jittered_rssi, jittered_amp))
    return stream


def main():
    parser = argparse.ArgumentParser(description="Bootstrap-augment training data from real recordings.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sessions-per-label", type=int, default=5)
    parser.add_argument("--packets-per-session", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for label_idx, label in enumerate(LABEL_NAMES):
        packets = load_real_packets_for_label(args.data_dir, label)
        if not packets:
            print(f"no real packets found for label {label!r}, skipping")
            continue
        print(f"{label}: pooled {len(packets)} real packets from existing sessions")

        for n in range(1, args.sessions_per_label + 1):
            session_id = f"synth_{label}_{n}"
            seed = args.seed + label_idx * 1000 + n
            stream = make_synthetic_stream(packets, args.packets_per_session, DEFAULT_RATE_HZ, seed)

            rows = []
            for window in sliding_windows(stream, DEFAULT_WINDOW_SECONDS, DEFAULT_STEP_SECONDS):
                feats = extract_features(window)
                if feats["n_packets"] == 0:
                    continue
                rows.append({"label": label, "session_id": session_id, "t": window[-1][0], **feats})

            out_path = os.path.join(args.data_dir, f"{session_id}.csv")
            pd.DataFrame(rows, columns=FEATURE_CSV_COLUMNS).to_csv(out_path, index=False)
            print(f"  wrote {len(rows)} synthetic feature rows to {out_path}")


if __name__ == "__main__":
    main()
