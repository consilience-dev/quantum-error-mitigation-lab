"""Evaluate four upstream-style forests on the fresh nested calibration curve."""
import csv
import gzip
import hashlib
import json
import time
from importlib.metadata import version

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from fetch_rf_batches import CACHE, ROOT


def clean_metrics(y, prediction, depths):
    delta = np.asarray(prediction, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    l2 = np.linalg.norm(delta, axis=1)
    return {"pooled_rmse": float(np.sqrt(np.mean(delta**2))),
            "mean_circuit_l2": float(np.mean(l2)),
            "mean_circuit_l2_per_depth": {int(d): float(l2[depths == d].mean()) for d in np.unique(depths)}}


def main():
    split_file = ROOT / "data/clean_split.json"
    plan = json.loads(split_file.read_text())
    data = np.load(CACHE / "rf_features.npz")
    X, y, ideal, depth = [data[name] for name in ("X", "y", "ideal", "depth")]
    test = np.array(plan["test_ids"], dtype=int)
    report = {"split_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
              "versions": {k: version(k) for k in ("numpy", "scikit-learn")},
              "n_test": len(test), "estimator": "four 100-tree RFs; sklearn defaults; same model RNG per run",
              "reference_baselines": {name: clean_metrics(ideal[test], data[name][test], depth[test])
                                      for name in ("noisy", "zne")}, "runs": []}
    slices = {"full": slice(None), "descriptors_only": slice(0, -4), "measurements_only": slice(-4, None)}
    with gzip.open(ROOT / "reports/rf_clean_predictions.csv.gz", "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["budget", "calibration_seed", "inputs", "unshuffled_row_id"] + [f"predicted_{i}" for i in range(4)])
        for run in plan["runs"]:
            train = np.asarray(run["calibration_ids"], dtype=int)
            if np.intersect1d(train, test).size or np.unique(train).size != run["budget"]:
                raise ValueError("Duplicate or overlapping rows in clean plan")
            for name, columns in slices.items():
                # Full input learning curve and both controls at the main 500-circuit comparison.
                if name != "full" and run["budget"] != 500:
                    continue
                start = time.perf_counter()
                prediction = np.column_stack([
                    RandomForestRegressor(n_estimators=100, random_state=i, n_jobs=2)
                    .fit(X[train, columns], y[train, i]).predict(X[test, columns]) for i in range(4)])
                stats = {"calibration_seed": run["seed"], "budget": run["budget"], "inputs": name,
                         "elapsed_seconds": time.perf_counter() - start,
                         **clean_metrics(ideal[test], prediction, depth[test])}
                report["runs"].append(stats)
                for row, pred in zip(test, prediction):
                    writer.writerow([run["budget"], run["seed"], name, row, *pred])
            print(run["seed"], run["budget"], flush=True)
    (ROOT / "reports/rf_clean.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
