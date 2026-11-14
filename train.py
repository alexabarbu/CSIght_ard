"""Train the empty/standing/walking classifier from recorded CSI data.

Usage:
    python train.py [--data-dir data] [--out model.pkl] [--test-fraction 0.25]

record.py writes one raw per-packet CSV per session, data/<label>_<n>.csv,
with columns time,rssi,v0,...,v127 -- one row per CSI packet, no windowing or
labeling yet. load_data() turns each of those files into windowed feature
rows itself (via features.sliding_windows/extract_features), tagging each
window with the label and session id taken from the filename
(e.g. "walking_2.csv" -> label "walking", session "walking_2"). Session id
matters because train/test is split BY SESSION below (not by row --
overlapping windows within one session are near-duplicates, so a row-level
split would leak and inflate accuracy).
"""

import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import (
    DEFAULT_STEP_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    FEATURE_NAMES,
    extract_features,
    sliding_windows,
)
from read_csi import values_to_amplitudes

LABEL_NAMES = ("empty", "standing", "walking")

FEATURE_CSV_COLUMNS = ["label", "session_id", "t"] + list(FEATURE_NAMES)


def is_raw_packet_csv(path: str) -> bool:
    """record.py's raw per-packet format starts with time,rssi,v0,... --
    anything else is assumed to already be windowed features."""
    with open(path) as f:
        header = f.readline()
    return header.startswith("time,rssi,v0")


def label_and_session_from_filename(path: str) -> tuple:
    stem = os.path.splitext(os.path.basename(path))[0]  # e.g. "walking_2"
    label = stem.rsplit("_", 1)[0] if "_" in stem else stem
    return label, stem


def raw_csv_to_feature_rows(
    path: str,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    step_seconds: float = DEFAULT_STEP_SECONDS,
) -> pd.DataFrame:
    """Read one of record.py's raw per-packet CSVs and turn it into the same
    shape of windowed feature rows record.py's engineered-feature sibling
    would have produced directly."""
    label, session_id = label_and_session_from_filename(path)
    raw = pd.read_csv(path)
    value_cols = [c for c in raw.columns if c.startswith("v")]

    packets = []
    for _, row in raw.iterrows():
        amplitudes = values_to_amplitudes(row[value_cols].to_numpy(dtype=int))
        packets.append((float(row["time"]), float(row["rssi"]), amplitudes))

    rows = []
    for window in sliding_windows(packets, window_seconds, step_seconds):
        feats = extract_features(window)
        if feats["n_packets"] == 0:
            continue
        rows.append({"label": label, "session_id": session_id, "t": window[-1][0], **feats})

    return pd.DataFrame(rows, columns=FEATURE_CSV_COLUMNS)


def load_data(data_dir: str = "data") -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}/ -- run record.py first.")

    frames = []
    for path in paths:
        if is_raw_packet_csv(path):
            frame = raw_csv_to_feature_rows(path)
            print(f"{path}: {len(frame)} windowed feature row(s) from raw packets")
        else:
            frame = pd.read_csv(path)
            print(f"{path}: {len(frame)} feature row(s) (already windowed)")
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)

    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(
            f"data is missing expected feature columns {missing} -- "
            f"was it recorded with an older version of features.py?"
        )

    before = len(df)
    df = df.dropna(subset=list(FEATURE_NAMES))
    dropped = before - len(df)
    if dropped:
        print(f"dropped {dropped} row(s) with missing/NaN features")

    present_labels = sorted(df["label"].unique()) if len(df) else []
    missing_labels = [l for l in LABEL_NAMES if l not in present_labels]
    if missing_labels:
        print(f"warning: no usable data yet for label(s) {missing_labels} -- "
              f"record more with record.py before training a useful model")

    return df


def session_split(df: pd.DataFrame, test_fraction: float = 0.25, seed: int = 0):
    """Hold out whole sessions per label for test, never splitting a session
    across train and test."""
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []

    for label, group in df.groupby("label"):
        sessions = sorted(group["session_id"].unique())
        if len(sessions) < 2:
            print(f"warning: label {label!r} has only {len(sessions)} session(s); "
                  f"can't hold one out -- all its rows go to train")
            train_idx.extend(group.index)
            continue

        shuffled = list(sessions)
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_fraction))
        test_sessions = set(shuffled[:n_test])

        for session_id, session_group in group.groupby("session_id"):
            target = test_idx if session_id in test_sessions else train_idx
            target.extend(session_group.index)

    return df.loc[train_idx], df.loc[test_idx]


def build_candidates():
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=0
        ),
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=0),
    }


def train_and_compare(train_df: pd.DataFrame, test_df: pd.DataFrame):
    x_train = train_df[list(FEATURE_NAMES)].values
    y_train = train_df["label"].values
    x_test = test_df[list(FEATURE_NAMES)].values
    y_test = test_df["label"].values

    results = {}
    for name, model in build_candidates().items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds, labels=LABEL_NAMES)
        results[name] = (model, acc, cm)

        print(f"\n[{name}] accuracy on held-out sessions: {acc:.3f}")
        print(f"confusion matrix (rows=true, cols=pred), labels={LABEL_NAMES}:")
        print(cm)

    best_name = max(results, key=lambda n: results[n][1])
    print(f"\nbest model: {best_name} (accuracy {results[best_name][1]:.3f})")
    return best_name, results[best_name][0]


def main():
    parser = argparse.ArgumentParser(description="Train the CSI motion classifier.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="model.pkl")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = load_data(args.data_dir)

    n_classes = df["label"].nunique()
    if n_classes < 2:
        present = sorted(df["label"].unique())
        raise SystemExit(
            f"Only {n_classes} class with usable data so far ({present}) -- "
            f"need at least 2 of {LABEL_NAMES} to train a classifier. "
            f"Record the missing labels with record.py (e.g. 'make standing', "
            f"'make walking') and try again."
        )

    train_df, test_df = session_split(df, args.test_fraction, args.seed)
    print(f"train rows: {len(train_df)}, test rows: {len(test_df)}")

    if len(test_df) == 0:
        print("warning: no held-out test rows (not enough distinct sessions yet); "
              "evaluating on training rows -- these accuracy numbers are optimistic")
        test_df = train_df

    best_name, best_model = train_and_compare(train_df, test_df)

    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "feature_names": list(FEATURE_NAMES),
            "feature_version": "v1",
            "label_names": list(LABEL_NAMES),
        },
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
