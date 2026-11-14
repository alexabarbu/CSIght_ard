"""Train the empty/standing/walking classifier from recorded CSI feature data.

Usage:
    python train.py [--data-dir data] [--out model.pkl] [--test-fraction 0.25]

Loads every data/<label>.csv written by record.py, splits train/test BY
RECORDING SESSION (not by row -- overlapping sliding windows within one
session are near-duplicates, so a row-level split would leak and inflate
accuracy), trains a few candidate models on the same split, and saves the
best one.
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

from features import FEATURE_NAMES

LABEL_NAMES = ("empty", "standing", "walking")


def load_data(data_dir: str = "data") -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}/ -- run record.py first.")

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

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
