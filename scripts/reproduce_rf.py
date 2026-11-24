"""Refit the ML-QEM four-qubit hardware demo on its original 500/2,500 row split."""
import argparse
import csv
import hashlib
import io
import json
import pickle
import platform
import re
from importlib.metadata import version

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from audit_measurements import NumericUnpickler
from fetch_rf_batches import CACHE, ROOT, SOURCES, check_blob
from fetch_upstream import verify


def circuit_features(circuit, edges):
    """Implement encode_data_v2_ecr with cx gate family and four noisy features."""
    counts = circuit.count_ops()
    gates = [counts.get(key, 0) for key in ("cx", "sx", "x", "id", "rz")]
    angles = []
    ordered_ops = []
    for op, qubits, clbits in circuit.data:
        params = [float(p) for p in op.params]
        if op.name in {"rx", "ry", "rz"} and len(qubits) == 1:
            angles.append(params[0])
        if op.name != "barrier":
            ordered_ops.append((op.name, [float.hex(v) for v in params],
                                [circuit.find_bit(q).index for q in qubits],
                                [circuit.find_bit(c).index for c in clbits]))
    hist, _ = np.histogram(angles, bins=edges)
    features = np.array(gates + hist.tolist(), dtype=np.float32) * np.float32(0.01)
    fingerprint = hashlib.sha256(json.dumps(ordered_ops, separators=(",", ":")).encode()).hexdigest()
    return features, fingerprint


def build_inputs():
    manifest = json.loads((CACHE / "manifest.json").read_text())
    for entry in manifest["files"]:
        verify((ROOT / entry["local_path"]).read_bytes(), entry)
    order = np.array(json.loads((CACHE / "index_order.json").read_text()))
    assert np.array_equal(np.sort(order), np.arange(3000))
    # Only read upstream pickles after validating pinned artifact identities.
    m = NumericUnpickler(io.BytesIO((CACHE / "results.pk").read_bytes())).load()
    inverse = np.argsort(order)
    ideal, noisy, zne = [np.asarray(m[key]).reshape(-1, 4)[inverse] for key in ("ideal", "noisy", "zne_mitigated")]
    X = np.empty((3000, 169), dtype=np.float32)
    X[:, -4:] = noisy
    y = ideal.astype(np.float32)
    logical_bytes_hash = np.empty(3000, dtype="U64")
    batch_ids = np.empty(3000, dtype=np.int32)
    edges = np.arange(-2 * np.pi, 2 * np.pi + 0.025 * np.pi, 0.025 * np.pi)
    sources = json.loads(SOURCES.read_text())
    assert [r["name"] for r in sources] == [f"batch_{b:02d}.pk" for b in range(60)]
    for i, entry in enumerate(sources):
        raw = (CACHE / entry["name"]).read_bytes()
        check_blob(raw, entry)
        batch = pickle.loads(raw)
        ids = order[50*i:50*(i+1)]
        assert len(batch["circuit_batch"]) == 50 and np.array_equal(ids, batch["index_order_batch"])
        for index, circuit in zip(ids, batch["circuit_batch"]):
            X[index, :-4], logical_bytes_hash[index] = circuit_features(circuit, edges)
            batch_ids[index] = i
    depth = np.concatenate([np.repeat(np.arange(10), 200), np.repeat(np.arange(10), 100)])
    ids_by_depth = [np.flatnonzero(depth == d) for d in range(10)]
    train_ids = np.concatenate([ids[:50] for ids in ids_by_depth])
    test_ids = np.concatenate([ids[50:] for ids in ids_by_depth])
    assert np.array_equal(np.sort(np.r_[train_ids, test_ids]), np.arange(3000))
    assert all(np.isfinite(array).all() for array in (ideal, X, zne, noisy))
    np.savez_compressed(CACHE / "rf_features.npz", X=X, y=y, ideal=ideal, noisy=noisy, zne=zne,
                        depth=depth, train_ids=train_ids, test_ids=test_ids,
                        content_hash=logical_bytes_hash, hardware_batch=batch_ids)
    return X, y, noisy, zne, depth, train_ids, test_ids, logical_bytes_hash


def summarize(y, predicted, depth):
    errors = predicted.astype(np.float64) - y.astype(np.float64)
    l2 = np.linalg.norm(errors, axis=1)
    return {"pooled_rmse": float(np.sqrt(np.square(errors).mean())),
            "mean_circuit_l2": float(l2.mean()),
            "rmse_per_observable": np.sqrt(np.square(errors).mean(axis=0)).tolist(),
            "mean_circuit_l2_per_depth": [float(l2[depth == d].mean()) for d in range(10)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    X, y, noisy, zne, depth, train, test, hashes = build_inputs()
    notebook = json.loads((CACHE / "demo2_ising_4q_hardware_plot.ipynb").read_text())
    output = "".join(line for cell in notebook["cells"] for o in cell.get("outputs", [])
                     for line in o.get("text", []))
    saved_rmse = float(re.search(r"^RMSE_mitigated: ([0-9.]+)$", output, flags=re.M).group(1))
    # Figure 4 source-data workbook, four-qubit hardware RF means, depths 0..9.
    fig4 = [0.008884, 0.053985, 0.080179, 0.091810, 0.114600,
            0.119991, 0.125018, 0.137034, 0.117355, 0.151538]
    report = {
        "dataset": "ML-QEM published four-qubit hardware row split",
        "upstream_revision": json.loads((CACHE / "manifest.json").read_text())["commit"],
        "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "batch_manifest_sha256": hashlib.sha256(SOURCES.read_bytes()).hexdigest(),
        "feature_matrix_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
        "target_matrix_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
        "python": platform.python_version(),
        "versions": {k: version(k) for k in ["numpy", "scikit-learn", "qiskit-terra", "symengine", "scipy"]},
        "n_train": len(train), "n_test": len(test), "features": X.shape[1],
        "n_trees_per_target": 100, "n_targets": 4,
        "depth0_unique_compiled_train": len(set(hashes[train[depth[train] == 0]])),
        "depth0_unique_compiled_test": len(set(hashes[test[depth[test] == 0]])),
        "unique_compiled_train_test_overlap_per_depth": [
            len(set(hashes[train[depth[train] == d]]) & set(hashes[test[depth[test] == d]])) for d in range(10)],
        "reference_notebook_pooled_rmse": saved_rmse,
        "reference_figure4_mean_circuit_l2_per_depth_6dp": fig4,
        "reference_figure4_mean_circuit_l2": float(np.mean(fig4)),
        "reference_figure4_source": "docs/paper_figures/ML-QEM Source data.xlsx",
        "split": "first 50 / last 250 rows in each depth, upstream order",
        "estimators": "4 independent RandomForestRegressor(n_estimators=100), other sklearn defaults",
        "randomness": "NumPy global RNG seeded before each refit; upstream notebook leaves this RNG unspecified",
        "raw": summarize(y[test], noisy[test].astype(np.float32), depth[test]),
        "zne": summarize(y[test], zne[test], depth[test]), "rf_refits": []}
    out = ROOT / "reports"
    with (out / "rf_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "unshuffled_row_id", "depth"] + [f"predicted_{i}" for i in range(4)])
        for seed in args.seeds:
            np.random.seed(seed)
            predictions = np.column_stack([
                RandomForestRegressor(n_estimators=100, n_jobs=-1).fit(X[train], y[train, i]).predict(X[test])
                for i in range(4)])
            stats = summarize(y[test], predictions, depth[test])
            stats.update({"seed": seed, "pooled_rmse_minus_notebook": stats["pooled_rmse"] - saved_rmse,
                          "mean_circuit_l2_minus_figure4": stats["mean_circuit_l2"] - float(np.mean(fig4))})
            report["rf_refits"].append(stats)
            for row_id, prediction in zip(test, predictions):
                writer.writerow([seed, row_id, depth[row_id], *prediction])
            print(seed, stats["pooled_rmse"], stats["mean_circuit_l2"], flush=True)
    (out / "rf_reproduction.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
