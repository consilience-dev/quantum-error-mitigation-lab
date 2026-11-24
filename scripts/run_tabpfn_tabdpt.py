"""Run TabPFN or TabDPT on the published or deduplicated ML-QEM split.

Install tabpfn==6.4.1 (v2) or tabpfn==10.2.0 (newer versions), tabdpt==1.3.2
and torch==2.15.0 in a Python 3.12 environment. Place the checkpoint in
weight_dir from https://huggingface.co/{repository}/resolve/{revision}/{filename}
using the corresponding tuple below. The runner verifies SHA-256 before loading.

Example: python scripts/run_tabpfn_tabdpt.py tabpfn_v2 weights --device mps
Add --clean-seed 0 to use 500 verified unique nonzero-depth circuits, or
--fine-tune to apply the specified three-epoch TabPFN v2 adaptation.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "tabpfn_v2_5": (
        "Prior-Labs/tabpfn_2_5", "6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b",
        "tabpfn-v2.5-regressor-v2.5_default.ckpt", "6e0308b7e5f3a8325b798b96d986b2b282c3608b1cda6e2991425a2a261d40b8",
    ),
    "tabpfn_v2_6": (
        "Prior-Labs/tabpfn_2_6", "24148873a9a5992b429833d07363690dcffd02a8",
        "tabpfn-v2.6-regressor-v2.6_default.ckpt", "c855169116d79d2c957a29107ffc04464555f39aec0eef90c90390f2c2b2008b",
    ),
    "tabpfn_v3_5": (
        "Prior-Labs/tabpfn_3_5", "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e",
        "tabpfn-v3.5-20260909.safetensors", "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
    ),
    "tabpfn_v3": (
        "Prior-Labs/tabpfn_3", "24a16a89d245878b846555110985634aa2e656d7",
        "tabpfn-v3-regressor-v3_default.ckpt", "311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301",
    ),
    "tabpfn_v2": (
        "Prior-Labs/TabPFN-v2-reg", "02a44705ff7790b50cded404bc59adb94da456a1",
        "tabpfn-v2-regressor-v2_default.ckpt",
        "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736",
    ),
    "tabdpt_v1_3": (
        "Layer6/TabDPT", "a5ca6e01c0fa09ec68c73e958e5199d1932abb3a", "tabdpt1_3.safetensors",
        "97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb",
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=CHECKPOINTS)
    parser.add_argument("weight_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clean-seed", type=int, choices=range(5), help="Use the frozen unique nonzero-depth split at 500")
    parser.add_argument("--clean-budget", type=int, default=500, choices=(18, 45, 90, 180, 270, 450, 500))
    parser.add_argument("--initialization-seed", type=int, choices=(0, 1, 2), help="Use the official freshly initialized TabPFN v2 architecture")
    parser.add_argument("--inputs", choices=("all", "descriptors", "noisy", "shuffle_train_noisy"), default="all")
    parser.add_argument("--fine-tune", action="store_true", help="Three fixed TabPFN fine-tuning epochs")
    parser.add_argument("--features", type=Path, default=ROOT / "data/upstream/rf_features.npz")
    args = parser.parse_args()
    repo, revision, filename, expected = CHECKPOINTS[args.model]
    weights = args.weight_dir / filename
    with weights.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != expected:
        raise ValueError("Unexpected checkpoint digest")
    if args.clean_budget != 500 and (args.clean_seed is None or args.fine_tune):
        parser.error("Learning-curve budgets require a clean split and frozen weights")
    if args.fine_tune and args.model != "tabpfn_v2":
        parser.error("--fine-tune requires tabpfn_v2")
    name = args.model + ("_tuned" if args.fine_tune else "")
    initialization_audit = None
    if args.initialization_seed is not None:
        if args.model != "tabpfn_v2" or args.clean_seed != 0 or args.fine_tune or args.clean_budget != 500 or args.inputs != "all":
            parser.error("The weight ablation uses clean split 0 at 500 and full inputs")
        name += f"_init{args.initialization_seed}"
    initial_tensor = None
    deltas = []
    if args.fine_tune:
        initial_tensor = torch.load(weights, map_location="cpu", weights_only=False)["state_dict"]["decoder_dict.standard.0.weight"]
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    with np.load(args.features) as data:
        X, y = data["X"], data["y"]
        train, test, depth = data["train_ids"], data["test_ids"], data["depth"]
    if X.shape != (3000, 169) or len(train) != 500 or len(test) != 2500 or np.intersect1d(train, test).size:
        raise ValueError("Expected original 500/2500 ML-QEM split")
    split_digest = None
    if args.clean_seed is not None:
        manifest = (ROOT / "data/clean_split.json").read_bytes()
        split_digest = hashlib.sha256(manifest).hexdigest()
        split = json.loads(manifest)
        run = next(r for r in split["runs"] if r["budget"] == args.clean_budget and r["seed"] == args.clean_seed)
        train = np.array(run["calibration_ids"], dtype=int)
        test = np.array(split["test_ids"], dtype=int)
        if len(np.unique(train)) != args.clean_budget or len(np.unique(test)) != 900 or np.intersect1d(train, test).size:
            raise ValueError("Invalid clean split")
        name += f"_clean_s{args.clean_seed}"
        if args.clean_budget != 500:
            name += f"_b{args.clean_budget}"
    donor_digest = None
    if args.inputs != "all":
        if args.clean_seed is None or args.fine_tune:
            parser.error("Input controls use the clean frozen comparisons")
        name += "_" + args.inputs
        if args.inputs == "descriptors":
            X = X[:, :-4]
        elif args.inputs == "noisy":
            X = X[:, -4:]
        else:
            # Jointly permute the four noisy values in calibration only. A cyclic
            # rotation of a shuffled index list deranges every non-singleton stratum.
            strata = json.loads((ROOT / "data/measurement_basis.json").read_text())
            basis = np.array(list(strata["basis_by_unshuffled_row_id"]))
            rng = np.random.default_rng(20261124)
            donors = np.arange(len(X))
            for d in np.unique(depth[train]):
                for b in np.unique(basis[train]):
                    ids = train[(depth[train] == d) & (basis[train] == b)]
                    ordered = rng.permutation(ids)
                    donors[ordered] = np.roll(ordered, 1)
            if np.any(donors[train] == train) or not np.array_equal(np.sort(donors[train]), np.sort(train)):
                raise ValueError("Calibration permutation has a fixed point or changes membership")
            X = X.copy()
            X[train, -4:] = X[donors[train], -4:]
            donor_digest = hashlib.sha256(donors[train].astype("<i8").tobytes()).hexdigest()
    preds = np.empty((len(test), 4), dtype=float)
    start = time.monotonic()
    for q in range(4):
        if args.model == "tabdpt_v1_3":
            from tabdpt import TabDPTRegressor
            model = TabDPTRegressor(model_weight_path=str(weights), device=args.device,
                                    use_flash=False, compile=False, verbose=False)
            model.fit(X[train], y[train, q])
            preds[:, q] = model.predict(X[test], context_size=len(train),
                                        n_ensembles=1, batch_size=256, seed=0)
        elif args.fine_tune:
            from tabpfn.finetuning import FinetunedTabPFNRegressor
            model = FinetunedTabPFNRegressor(
                device=args.device, epochs=3, learning_rate=1e-5, random_state=0,
                validation_split_ratio=0.1, early_stopping=False, use_lr_scheduler=False,
                n_finetune_ctx_plus_query_samples=128, n_estimators_finetune=1,
                n_estimators_validation=1, n_estimators_final_inference=1,
                use_activation_checkpointing=False,
                extra_regressor_kwargs={"model_path": weights, "inference_precision": torch.float32},
            )
            model.fit(X[train], y[train, q])
            final_tensor = model.finetuned_estimator_.model_.state_dict()["decoder_dict.standard.0.weight"].cpu()
            delta = (final_tensor - initial_tensor).abs().max().item()
            if not np.isfinite(delta) or delta == 0:
                raise RuntimeError("No verified finite update to the regression decoder")
            deltas.append(delta)
            # The host library's tuned-clone API uses fit_preprocessors mode.
            preds[:, q] = np.concatenate([model.predict(X[ids]) for ids in np.array_split(test, 5)])
        else:
            from tabpfn import TabPFNRegressor
            model_input = weights
            if args.initialization_seed is not None:
                from tabpfn_initialization import initialized_spec
                model_input, audit = initialized_spec(weights, args.initialization_seed)
                if initialization_audit is not None and initialization_audit != audit:
                    raise ValueError("Target-specific models must share the same initial parameters")
                initialization_audit = audit
            model = TabPFNRegressor(model_path=model_input, device=args.device, random_state=0,
                                    n_estimators=1, fit_mode="fit_with_cache",
                                    inference_precision=torch.float32)
            model.fit(X[train], y[train, q])
            chunks = 2 if args.clean_budget < 500 else 20
            preds[:, q] = np.concatenate([model.predict(X[ids]) for ids in np.array_split(test, chunks)])
        print(f"{name} completed observable {q + 1}/4 in {time.monotonic()-start:.1f} s", flush=True)
    if not np.isfinite(preds).all():
        raise ValueError("Nonfinite predictions")
    error = preds - y[test]
    distance = np.linalg.norm(error, axis=1)
    report = {
        "model": name, "weights_updated": args.fine_tune, "device": args.device,
        "checkpoint_url": f"https://huggingface.co/{repo}/resolve/{revision}/{filename}",
        "checkpoint_sha256": expected, "feature_archive_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "split": "published" if args.clean_seed is None else "clean", "clean_manifest_sha256": split_digest,
        "inputs": args.inputs, "training_noise_donors_sha256": donor_digest,
        "split_seed": args.clean_seed, "calibration_rows": len(train), "test_rows": len(test), "features": X.shape[1],
        "ensemble_members": 1, "seed": 0, "seconds": time.monotonic()-start,
        "pooled_rmse": float(np.mean(error**2)**0.5),
        "mean_circuit_l2": float(np.mean(distance)),
        "mean_circuit_l2_by_depth": {int(d): float(np.mean(distance[depth[test]==d])) for d in np.unique(depth[test])},
        "python": platform.python_version(),
        "versions": {p: version(p) for p in ["torch", "tabpfn", "tabdpt", "numpy", "scikit-learn"]},
    }
    if initialization_audit is not None:
        report["initialization_control"] = {
            **initialization_audit,
            "per_target_prediction_sd": np.std(preds, axis=0).tolist(),
            "calibration_target_mean_reference_l2": float(np.linalg.norm(
                y[test] - y[train].mean(axis=0), axis=1).mean()),
        }
    if args.fine_tune:
        report["tuning"] = {"epochs": 3, "learning_rate": 1e-5, "training_rows": 450, "validation_rows": 50, "final_context_rows": 500, "episode_size": 128, "fixed_schedule": True, "decoder_max_abs_updates": deltas}
    with (ROOT / f"reports/{name}_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["circuit_id", "depth", *[f"prediction_{q}" for q in range(4)]])
        writer.writerows([int(i), int(depth[i]), *values] for i, values in zip(test, preds))
    (ROOT / f"reports/{name}.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
