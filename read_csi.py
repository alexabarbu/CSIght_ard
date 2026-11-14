"""Read CSI_DATA lines from the ESP32-C3 over serial and yield amplitude data.

Line format, as printed by firmware/main/main.c:
    CSI_DATA,<rssi>,<len>,[v0,v1,v2,...]
The bracketed values are signed bytes, (imaginary, real) pairs, one pair per
subcarrier. `len` is the raw byte count of that packet's CSI buffer, so
n_subcarriers = len // 2 -- this can vary between packets and must never be
assumed fixed by downstream code.
"""

import argparse
import re
import sys
import time
from typing import Iterator, Optional, Tuple

import numpy as np
import serial

DEFAULT_BAUD = 921600

LINE_RE = re.compile(r"^CSI_DATA,(-?\d+),(\d+),\[(.*)\]\s*$")


def values_to_amplitudes(values) -> np.ndarray:
    """Turn a flat list of (imaginary, real) byte pairs into one amplitude
    per subcarrier. Shared by the live serial parser and by code that reads
    the raw per-packet CSVs record.py writes (same value layout either way)."""
    pairs = np.asarray(values, dtype=np.float64)
    imag = pairs[0::2]
    real = pairs[1::2]
    return np.sqrt(imag**2 + real**2)


def parse_csi_line(line: str) -> Optional[Tuple[int, np.ndarray]]:
    """Parse one CSI_DATA line into (rssi, amplitudes).

    Returns None for any malformed line (wrong prefix, bad numbers, value
    count that doesn't match the declared length or isn't even) -- callers
    should skip and keep reading rather than crash.
    """
    match = LINE_RE.match(line.strip())
    if match is None:
        return None

    try:
        rssi = int(match.group(1))
        declared_len = int(match.group(2))
        values_str = match.group(3)
        values = [int(v) for v in values_str.split(",")] if values_str else []
    except ValueError:
        return None

    if len(values) != declared_len or len(values) % 2 != 0:
        return None

    amplitudes = values_to_amplitudes(values)
    return rssi, amplitudes


def read_csi(port: str, baud: int = DEFAULT_BAUD) -> Iterator[Tuple[float, float, np.ndarray]]:
    """Yield (timestamp, rssi, amplitudes) for every valid CSI_DATA line read
    from the serial port. Never raises on bad input -- malformed or unrelated
    lines (e.g. STATUS/log lines) are skipped with a warning on stderr.
    """
    with serial.Serial(port, baud, timeout=1) as ser:
        while True:
            try:
                raw = ser.readline()
            except serial.SerialException as exc:
                print(f"read_csi: serial error, stopping: {exc}", file=sys.stderr)
                return

            if not raw:
                continue  # read timeout, no data yet

            line = raw.decode("utf-8", errors="ignore")
            if not line.startswith("CSI_DATA"):
                continue

            parsed = parse_csi_line(line)
            if parsed is None:
                print(f"read_csi: skipping malformed line: {line.strip()[:80]!r}", file=sys.stderr)
                continue

            rssi, amplitudes = parsed
            yield time.time(), rssi, amplitudes


def main():
    parser = argparse.ArgumentParser(description="Stream parsed CSI amplitudes from the ESP32-C3.")
    parser.add_argument("port", help="Serial port, e.g. /dev/ttyACM0 or COM5")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    args = parser.parse_args()

    print(f"Reading CSI from {args.port} @ {args.baud} baud. Ctrl+C to stop.")
    count = 0
    for _timestamp, rssi, amplitudes in read_csi(args.port, args.baud):
        count += 1
        if count % 20 == 0:
            mean_amp = f"{amplitudes.mean():.1f}" if amplitudes.size else "n/a"
            print(f"packet {count}: rssi={rssi} n_subcarriers={len(amplitudes)} mean_amp={mean_amp}")


if __name__ == "__main__":
    main()
