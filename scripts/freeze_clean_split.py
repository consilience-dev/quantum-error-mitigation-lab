"""Audit logical and compiled identities and build the predetermined nonzero-depth split."""
import hashlib
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen

import numpy as np

from fetch_rf_batches import CACHE, ROOT, check_blob
from plan_splits import plan
from reproduce_rf import circuit_features

SOURCES = ROOT / "data/rf_logical_sources.json"
PLAN = ROOT / "data/clean_split.json"
HOLDOUT_SEED = "ml-qem-clean-v1-holdout"


def fetch_one(entry, revision):
    partition, name = entry["path"].split("/")[-2:]
    path = CACHE / f"logical_{partition}_{name}"
    if path.exists():
        raw = path.read_bytes()
    else:
        url = f"https://raw.githubusercontent.com/qiskit-community/ml-qem/{revision}/{entry['path']}"
        with urlopen(url, timeout=60) as response:
            raw = response.read()
        check_blob(raw, {"name": name, **entry})
        path.write_bytes(raw)
    check_blob(raw, {"name": name, **entry})
    return path


def main():
    with np.load(CACHE / "rf_features.npz", allow_pickle=False) as inputs:
        expected_hashes, depths = inputs["content_hash"], inputs["depth"]
    revision = json.loads((CACHE / "manifest.json").read_text())["commit"]
    entries = json.loads(SOURCES.read_text())
    expected = [f"{part}/step_{depth:02d}.pk" for part in ("train", "val") for depth in range(10)]
    if ["/".join(e["path"].split("/")[-2:]) for e in entries] != expected:
        raise ValueError("Expected 20 ordered source depth files")
    with ThreadPoolExecutor(max_workers=6) as pool:
        paths = list(pool.map(lambda e: fetch_one(e, revision), entries))
    # Fresh holdout selection reads only circuit identity and depth.
    compiled_keys, logical_keys = [], []
    edges = np.arange(-2 * np.pi, 2 * np.pi + 0.025 * np.pi, 0.025 * np.pi)
    for i, path in enumerate(paths):
        depth = i % 10
        source = pickle.loads(path.read_bytes())
        if len(source) != (200 if i < 10 else 100):
            raise ValueError(f"Unexpected number of original circuits in {path.name}")
        for row in source:
            row_id = len(logical_keys)
            _, fingerprint = circuit_features(row["circuit"], edges)
            if fingerprint != expected_hashes[row_id] or depth != depths[row_id]:
                raise ValueError(f"Original source differs from hardware batch at row {row_id}")
            # At depth 0, no operation depends on the coupling J.
            params = [depth, float(row["J"]).hex() if depth else None, str(row["basis"])]
            logical_keys.append(hashlib.sha256(json.dumps(params).encode()).hexdigest())
            compiled_keys.append(fingerprint)
    # Union by either definition before splitting. Keep lowest original row ID in each group.
    root = list(range(len(compiled_keys)))
    def representative(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i
    for keys in (compiled_keys, logical_keys):
        first = {}
        for row_id, key in enumerate(keys):
            a, b = representative(row_id), representative(first.setdefault(key, row_id))
            root[max(a, b)] = min(a, b)
    records, by_depth = [], []
    for depth in range(10):
        all_ids = np.flatnonzero(depths == depth).tolist()
        retained = [i for i in all_ids if representative(i) == i]
        by_depth.append({"depth": depth, "total_rows": len(all_ids), "unique_groups": len(retained),
                         "unique_compiled": len({compiled_keys[i] for i in all_ids}),
                         "unique_logical": len({logical_keys[i] for i in all_ids})})
        if depth == 0:
            continue
        # Stable digest ordering, independent of noisy values, labels, and previous scores.
        held_out = sorted(retained, key=lambda i: hashlib.sha256(
            f"{HOLDOUT_SEED}:{logical_keys[i]}".encode()).digest())[:100]
        if len(held_out) != 100:
            raise ValueError(f"Fewer than 100 eligible test circuits at depth {depth}")
        for i in retained:
            records.append({"id": str(i), "depth": depth, "partition": "test" if i in held_out else "calibration",
                            "circuit_hash": logical_keys[i]})
    config = json.loads((ROOT / "configs/followup.json").read_text())["clean_curve"]
    result = plan(records, config["budgets"], config["seeds"])
    result.update({"upstream_revision": revision, "logical_manifest_sha256": hashlib.sha256(SOURCES.read_bytes()).hexdigest(),
                   "holdout_seed_text": HOLDOUT_SEED, "identity_audit": by_depth,
                   "id_definition": "0-based index in original unshuffled hardware order"})
    PLAN.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(PLAN, by_depth, flush=True)


if __name__ == "__main__":
    main()
