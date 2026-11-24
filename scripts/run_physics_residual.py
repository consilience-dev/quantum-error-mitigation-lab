"""Exploratory cross-fitted measurement correction to the known-coupling interpolation baseline."""
import csv
import gzip
import hashlib
import json
import pickle

import numpy as np

from freeze_clean_split import CACHE, ROOT, SOURCES, fetch_one


def read_couplings(revision):
    values, axes = [], []
    for entry in json.loads(SOURCES.read_bytes()):
        rows = pickle.loads(fetch_one(entry, revision).read_bytes())
        values.extend(float(row["J"]) for row in rows)
        axes.extend(str(row["basis"]).upper() for row in rows)
    expected = json.loads((ROOT / "data/measurement_basis.json").read_bytes())["basis_by_unshuffled_row_id"]
    if "".join(axes) != expected:
        raise ValueError("Calibration metadata order does not match the basis manifest")
    return np.asarray(values), np.asarray(axes)


def interpolate(j, labels, train_ids, query_j):
    ids = train_ids[np.argsort(j[train_ids])]
    return np.column_stack([np.interp(query_j, j[ids], labels[ids, q]) for q in range(4)])


def independent_test_reference():
    rows = {}
    with gzip.open(ROOT / "reports/physics_control_predictions.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            if r["split"] == "clean":
                rows[(int(r["seed"]), int(r["row_id"]))] = [float(r[f"pred_{q}"]) for q in range(4)]
    return rows


def main():
    plan_bytes = (ROOT / "data/clean_split.json").read_bytes()
    plan = json.loads(plan_bytes)
    revision = plan["upstream_revision"]
    j, axes = read_couplings(revision)
    with np.load(CACHE / "rf_features.npz") as a:
        labels, ideal, noisy, depth, batches = [a[k] for k in ("y", "ideal", "noisy", "depth", "hardware_batch")]
    test = np.asarray(plan["test_ids"], dtype=int)
    base_reference = independent_test_reference()
    out = ROOT / "reports"
    arms = ["parameter_interpolation", "intercept_only", "affine_measurement_residual"]
    loss = {name: [] for name in arms}
    runs = []
    with gzip.open(out / "physics_residual_predictions.csv.gz", "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "arm", "row_id", *[f"pred_{q}" for q in range(4)]])
        for run in plan["runs"]:
            if run["budget"] != 500:
                continue
            train = np.asarray(run["calibration_ids"], dtype=int)
            if np.intersect1d(train, test).size or len(np.unique(train)) != 500:
                raise ValueError("Unexpected calibration deduplication or overlap")
            loo = np.full((len(train), 4), np.nan)
            base = np.full((len(test), 4), np.nan)
            for d in range(1, 10):
                for axis in "XYZ":
                    tr = np.flatnonzero((depth[train] == d) & (axes[train] == axis))
                    te = np.flatnonzero((depth[test] == d) & (axes[test] == axis))
                    if len(tr) < 2:
                        raise ValueError("Too few training circuits in stratum for leave-one-out")
                    base[te] = interpolate(j, labels, train[tr], j[test[te]])
                    for i in tr:
                        loo[i] = interpolate(j, labels, train[tr[tr != i]], j[train[i:i+1]])[0]
            if not np.isfinite(loo).all():
                raise ValueError("Missing leave-one-out predictions")
            np.testing.assert_array_equal(base, [base_reference[run["seed"], int(index)] for index in test])
            target_residual = labels[train] - loo
            noise_residual = noisy[train] - loo
            slopes, intercepts = [], []
            for q in range(4):
                coef, _, rank, _ = np.linalg.lstsq(
                    np.column_stack([np.ones(len(train)), noise_residual[:, q]]), target_residual[:, q], rcond=None)
                if rank != 2:
                    raise ValueError("Singular affine calibration fit")
                intercepts.append(float(coef[0]))
                slopes.append(float(coef[1]))
            corrected = base + np.asarray(intercepts) + (noisy[test] - base)*np.asarray(slopes)
            summary = {"seed": run["seed"], "calibration_rows": len(train), "affine_intercept": intercepts,
                       "affine_slope": slopes, "mean_circuit_l2": {}}
            for name, p in zip(arms, [base, base + target_residual.mean(axis=0), corrected]):
                circuit_errors = np.linalg.norm(p-ideal[test], axis=1)
                loss[name].append(circuit_errors)
                summary["mean_circuit_l2"][name] = float(circuit_errors.mean())
                writer.writerows((run["seed"], name, int(i), *v) for i,v in zip(test,p))
            runs.append(summary)
    if {r["seed"] for r in runs} != set(range(5)):
        raise ValueError("Expected all five calibration draws")
    # Resample entire acquisition batches and average across the five fixed calibration draws.
    _, clusters = np.unique(batches[test], return_inverse=True)
    sizes = np.bincount(clusters)
    rng = np.random.default_rng(20261124)
    draws = rng.integers(len(sizes), size=(10000, len(sizes)))
    contrasts = {}
    for reference in arms[:2]:
        delta = np.mean(np.asarray(loss[arms[2]]) - np.asarray(loss[reference]), axis=0)
        totals = np.bincount(clusters, weights=delta)
        replicates = totals[draws].sum(axis=1)/sizes[draws].sum(axis=1)
        contrasts[reference] = {"mean_corrected_minus_reference": float(delta.mean()),
                                "paired_acquisition_batch_bootstrap_95pct": np.quantile(replicates, [.025,.975]).tolist()}
    report = {
        "status": "exploratory_post_hoc_control",
        "method": "Within each (depth,basis) linearly interpolate in J, with constant endpoints. For the calibration set, leave each circuit out of interpolation when computing its residual. Fit one global slope and intercept per observable by OLS. Test correction uses all 500 calibration rows for interpolation plus intercept + slope*(noisy - interpolated).",
        "intercept_only_control": "Add the mean leave-one-out target residual per observable without reading hardware outputs",
        "test_circuits": len(test), "calibration_draws": len(runs), "upstream_revision": revision,
        "split_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "seeds": runs,
        "mean_circuit_l2_across_draws": {name:float(np.mean(by_seed)) for name,by_seed in loss.items()},
        "contrasts": contrasts,
        "uncertainty": "Nominal percentile intervals resample the observed acquisition batches 10,000 times together across methods and seeds, conditional on these calibration draws. Negative differences favor using the measurement."
    }
    (out / "physics_residual.json").write_text(json.dumps(report, indent=2)+"\n")
    print(report["mean_circuit_l2_across_draws"], contrasts)


if __name__ == "__main__":
    main()
