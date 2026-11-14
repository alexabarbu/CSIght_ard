"""Load the trained model and classify a live CSI window.

This is the only module the dashboard needs to import:

    bundle = load_model("model.pkl")       # once, at startup
    label, confidence, feats = predict(window, bundle)   # every tick

Both functions are safe to call even if no model has been trained yet --
load_model returns None and predict then returns ("unknown", 0.0, feats) so
the dashboard can show a fallback/"no model" state instead of crashing.
"""

import sys
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

from features import Packet, extract_features, rssi_occupancy


def load_model(path: str = "model.pkl") -> Optional[dict]:
    try:
        return joblib.load(path)
    except Exception as exc:
        print(f"infer: could not load model from {path!r}: {exc}", file=sys.stderr)
        return None


def predict(window: List[Packet], model_bundle: Optional[dict]) -> Tuple[str, float, Dict[str, float]]:
    """Returns (label, confidence, feature_dict). Never raises.

    RSSI decides empty vs. occupied first (see features.rssi_occupancy --
    hardcoded thresholds calibrated live on the current setup). The trained
    model only picks between standing/walking once RSSI says someone is
    there, and decides freely in the small uncertain gap between the two
    RSSI thresholds.
    """
    try:
        feats = extract_features(window)
        if model_bundle is None or feats.get("n_packets", 0) == 0:
            return "unknown", 0.0, feats

        occupancy = rssi_occupancy(feats)
        if occupancy == "empty":
            return "empty", 1.0, feats

        feature_names = model_bundle["feature_names"]
        values = [feats[name] for name in feature_names]
        if any(np.isnan(v) for v in values):
            return "unknown", 0.0, feats

        model = model_bundle["model"]
        proba = model.predict_proba([values])[0]
        classes = model.classes_

        if occupancy == "occupied":
            allowed = [i for i, c in enumerate(classes) if c != "empty"]
            best_idx = max(allowed, key=lambda i: proba[i])
        else:  # "uncertain" -- let the model decide freely among all 3 classes
            best_idx = int(np.argmax(proba))

        label = str(classes[best_idx])
        return label, float(proba[best_idx]), feats
    except Exception as exc:
        print(f"infer: prediction failed: {exc}", file=sys.stderr)
        return "unknown", 0.0, {}


def main():
    import argparse
    import time

    from features import DEFAULT_STEP_SECONDS, DEFAULT_WINDOW_SECONDS, sliding_windows
    from read_csi import DEFAULT_BAUD, read_csi

    parser = argparse.ArgumentParser(description="Print live status from the trained model.")
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyACM0 or COM5")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--model", default="model.pkl")
    args = parser.parse_args()

    bundle = load_model(args.model)
    if bundle is None:
        print("no model loaded -- every prediction will be 'unknown'")

    packets = read_csi(args.port, args.baud)
    windows = sliding_windows(packets, DEFAULT_WINDOW_SECONDS, DEFAULT_STEP_SECONDS)
    for window in windows:
        label, confidence, feats = predict(window, bundle)
        print(f"{time.strftime('%H:%M:%S')}  {label:10s}  confidence={confidence:.2f}  "
              f"n_packets={feats.get('n_packets', 0)}")


if __name__ == "__main__":
    main()
