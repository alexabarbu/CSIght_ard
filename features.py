"""Turn a window of parsed CSI packets into a fixed-size feature vector.

Each packet can carry a different number of subcarriers (see read_csi.py), so
a packet's amplitude array is treated as an unordered distribution rather than
a fixed-position vector: it's first summarized into a small set of
order-independent statistics (mean/std/max/min/percentiles), and those
per-packet statistics are then aggregated across the window. This way
extract_features() never indexes a specific subcarrier position, which would
be unsafe across packets of different lengths.
"""

import argparse
import time
from collections import deque
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np

from read_csi import DEFAULT_BAUD, read_csi

DEFAULT_WINDOW_SECONDS = 1.0
DEFAULT_STEP_SECONDS = 0.25

STAT_NAMES = ("mean", "std", "max", "min", "p25", "p50", "p75", "p90")
PERCENTILES = (25, 50, 75, 90)

Packet = Tuple[float, float, np.ndarray]  # (timestamp, rssi, amplitudes)

FEATURE_NAMES = (
    ["n_packets", "window_duration", "mean_rssi", "std_rssi", "pooled_std_amplitude"]
    + [f"{name}_time_mean" for name in STAT_NAMES]
    + [f"{name}_time_std" for name in STAT_NAMES]
)


def per_packet_stats(amplitudes: np.ndarray) -> Dict[str, float]:
    """Summarize one packet's variable-length amplitude array into the fixed
    set of stats named in STAT_NAMES. NaN-safe on an empty array."""
    if amplitudes.size == 0:
        return {name: float("nan") for name in STAT_NAMES}

    stats = {
        "mean": float(np.mean(amplitudes)),
        "std": float(np.std(amplitudes)),
        "max": float(np.max(amplitudes)),
        "min": float(np.min(amplitudes)),
    }
    for p, v in zip(PERCENTILES, np.percentile(amplitudes, PERCENTILES)):
        stats[f"p{p}"] = float(v)
    return stats


def extract_features(window: List[Packet]) -> Dict[str, float]:
    """window: list of (timestamp, rssi, amplitudes) covering ~1s.

    Always returns the same fixed set of keys (see FEATURE_NAMES), regardless
    of how many packets or subcarriers are present. For an empty window,
    returns n_packets=0 and NaN for every other feature so callers can detect
    an unreliable/empty window without this function ever raising.
    """
    if not window:
        feats = {"n_packets": 0, "window_duration": 0.0, "mean_rssi": float("nan"),
                  "std_rssi": float("nan"), "pooled_std_amplitude": float("nan")}
        for name in STAT_NAMES:
            feats[f"{name}_time_mean"] = float("nan")
            feats[f"{name}_time_std"] = float("nan")
        return feats

    per_packet = [per_packet_stats(amplitudes) for _, _, amplitudes in window]
    rssis = np.array([rssi for _, rssi, _ in window], dtype=np.float64)
    nonempty = [amplitudes for _, _, amplitudes in window if amplitudes.size > 0]
    pooled = np.concatenate(nonempty) if nonempty else np.array([])

    feats: Dict[str, float] = {
        "n_packets": len(window),
        "window_duration": window[-1][0] - window[0][0],
        "mean_rssi": float(np.mean(rssis)),
        "std_rssi": float(np.std(rssis)),
        "pooled_std_amplitude": float(np.std(pooled)) if pooled.size else float("nan"),
    }

    for name in STAT_NAMES:
        series = np.array([p[name] for p in per_packet], dtype=np.float64)
        series = series[~np.isnan(series)]
        feats[f"{name}_time_mean"] = float(np.mean(series)) if series.size else float("nan")
        feats[f"{name}_time_std"] = float(np.std(series)) if series.size else float("nan")

    return feats


def motion_score(features: Dict[str, float]) -> float:
    """Single scalar proxy for 'how much is amplitude changing over time':
    the std, across the window, of each packet's mean amplitude."""
    return features.get("mean_time_std", float("nan"))


def sliding_windows(
    packets: Iterable[Packet],
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    step_seconds: float = DEFAULT_STEP_SECONDS,
) -> Iterator[List[Packet]]:
    """Consume a stream of (timestamp, rssi, amplitudes) packets and yield a
    list of the packets from the last `window_seconds`, once every
    `step_seconds` of wall-clock time (by packet timestamp)."""
    buf: deque = deque()
    next_emit = None

    for packet in packets:
        timestamp = packet[0]
        buf.append(packet)
        while buf and timestamp - buf[0][0] > window_seconds:
            buf.popleft()

        if next_emit is None:
            next_emit = timestamp + step_seconds
        elif timestamp >= next_emit:
            yield list(buf)
            next_emit = timestamp + step_seconds


def main():
    parser = argparse.ArgumentParser(description="Live motion-score monitor from CSI.")
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyACM0 or COM5")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--step-seconds", type=float, default=DEFAULT_STEP_SECONDS)
    parser.add_argument("--threshold", type=float, default=2.0,
                         help="Motion score above this prints MOTION instead of NO MOTION.")
    args = parser.parse_args()

    packets = read_csi(args.port, args.baud)
    windows = sliding_windows(packets, args.window_seconds, args.step_seconds)
    for window in windows:
        feats = extract_features(window)
        score = motion_score(feats)
        is_motion = not np.isnan(score) and score > args.threshold
        status = "MOTION" if is_motion else "NO MOTION"
        print(f"{time.strftime('%H:%M:%S')}  motion_score={score:.2f}  "
              f"n_packets={feats['n_packets']}  -> {status}")


if __name__ == "__main__":
    main()
