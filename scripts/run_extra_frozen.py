"""Evaluate pinned Causilo, TabAbax and Mitra regression checkpoints at 500 calibration rows."""
import argparse
import csv
import hashlib
import json
import os
import time
from importlib.metadata import version
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/extra_model_sources.json"


def checkpoints(name):
    record = json.loads(MANIFEST.read_text())[name]
    local = ROOT / ".model-cache/extra" / name
    for filename, digest in record["files"].items():
        path = local / filename
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/{record['repo']}/resolve/{record['revision']}/{filename}"
            with urlopen(url, timeout=120) as response:
                raw = response.read()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("Invalid downloaded model digest")
            path.write_bytes(raw)
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != digest:
                raise ValueError("Unexpected local checkpoint")
    return local, record


def predict_mitra(local, x_train, targets, x_test, device, batch_size):
    from tabtune.models.mitra.tab2d import Tab2D
    model = Tab2D.from_pretrained(str(local), device="cpu").eval().to(device)
    keep = np.ptp(x_train, axis=0) > 0
    support = torch.from_numpy(np.ascontiguousarray(x_train[:, keep])).to(device).unsqueeze(0)
    low, high = float(targets.min()), float(targets.max())
    y = torch.tensor((targets - low) / (high - low), device=device)[None, :]
    bins = torch.linspace(-0.5, 1.5, model.dim_output + 1, device=device)
    if model.dim_output > 1:
        y = torch.bucketize(y, bins).sub(1).clamp(0, model.dim_output - 1).long()
    def mask(size):
        return torch.zeros((1, size), dtype=torch.bool, device=device)
    pieces = []
    for offset in range(0, len(x_test), batch_size):
        batch = x_test[offset:offset + batch_size, keep]
        query = torch.from_numpy(batch.copy()).to(device)[None, :]
        with torch.autocast("mps", dtype=torch.float16, enabled=(device == "mps")):
            out = model(support.clone(), y, query, mask(support.shape[-1]), mask(y.shape[-1]), mask(len(batch)))
        if model.dim_output > 1:
            out = bins[out.argmax(-1)] + (bins[1] - bins[0]) / 2
        pieces.append(out.flatten().float().cpu().numpy() * (high-low) + low)
    return np.concatenate(pieces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("causilo", "tababax", "mitra", "mitra_v2"))
    parser.add_argument("--split", choices=("published", "clean"), default="published")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--inputs", choices=("full","descriptors_only","measurements_only"), default="full")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch.set_num_threads(2)
    torch.manual_seed(0)
    local, record = checkpoints(args.model)
    with np.load(ROOT / "data/upstream/rf_features.npz") as a:
        X, y, depth, train, test = [a[k] for k in ("X", "y", "depth", "train_ids", "test_ids")]
    split_sha = None
    if args.split == "clean":
        raw = (ROOT / "data/clean_split.json").read_bytes()
        split_sha = hashlib.sha256(raw).hexdigest()
        plan = json.loads(raw)
        run = next(r for r in plan["runs"] if r["budget"] == 500 and r["seed"] == args.seed)
        train = np.asarray(run["calibration_ids"], dtype=int)
        test = np.asarray(plan["test_ids"], dtype=int)
    assert len(train) == 500 and not np.intersect1d(train, test).size
    X = X[:, {"full":slice(None),"descriptors_only":slice(None,-4),"measurements_only":slice(-4,None)}[args.inputs]]
    start = time.monotonic()
    preds = np.empty((len(test),4))
    with torch.inference_mode():
        for q in range(4):
            if args.model in ("mitra","mitra_v2"):
                preds[:, q] = predict_mitra(local, X[train], y[train,q], X[test], args.device, args.batch_size)
            else:
                if args.model == "causilo":
                    from tabtune.models.causilo import CausiloRegressor
                    from tabtune.models.causilo.checkpoints import load_checkpoint
                    from tabtune.models.causilo.engine import Engine
                    estimator = CausiloRegressor(n_estimators=1, random_state=0, use_kv_cache=True, device=args.device)
                    estimator._engine = Engine("regression", args.device)
                    estimator._engine.model = load_checkpoint(local / "regressor").to(args.device)
                else:
                    from tabtune.models.tababax import TabAbaxRegressor
                    estimator = TabAbaxRegressor(n_estimators=1, model_path=local / "tababax-regressor.ckpt",
                                                 allow_auto_download=False, device=args.device, random_state=0,
                                                 n_jobs=1, use_amp=False)
                estimator.fit(X[train], y[train,q])
                preds[:,q] = np.concatenate([estimator.predict(X[test[i:i+args.batch_size]])
                                             for i in range(0, len(test), args.batch_size)])
            print(args.model, args.split, q, round(time.monotonic()-start, 2), flush=True)
    if not np.isfinite(preds).all():
        raise ValueError("Nonfinite predictions")
    errors = preds - y[test]
    l2 = np.linalg.norm(errors, axis=1)
    name = f"extra_{args.model}_{args.split}" + (f"_seed{args.seed}" if args.split=="clean" else "")
    if args.inputs != "full":
        name += "_"+args.inputs
    out = ROOT / "reports"
    with (out / (name+"_predictions.csv")).open("w", newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["circuit_id","depth",*[f"prediction_{q}" for q in range(4)]])
        writer.writerows([i, depth[i], *v] for i,v in zip(test,preds))
    report = {"model": args.model, "split": args.split, "clean_split_sha256": split_sha,
              "calibration_seed": args.seed if args.split=="clean" else None, "n_train":len(train),
              "n_test":len(test), "inputs": args.inputs, "input_columns":X.shape[1], "checkpoint":record, "weights_updated":False,
              "device":args.device, "mitra_mps_autocast_fp16": args.model.startswith("mitra") and args.device == "mps", "query_batch_size":args.batch_size, "ensemble_members":1,
              "seconds":float(time.monotonic()-start), "tabtune_version":version("tabtune"),
              "torch_version":version("torch"),"numpy_version":version("numpy"),
              "scikit_learn_version":version("scikit-learn"),"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "mean_circuit_l2":float(l2.mean()), "pooled_rmse":float(np.sqrt(np.mean(errors**2))),
              "mean_circuit_l2_by_depth":{int(d):float(l2[depth[test]==d].mean()) for d in np.unique(depth[test])}}
    (out/(name+".json")).write_text(json.dumps(report,indent=2)+"\n")
    print(name,report["mean_circuit_l2"],flush=True)


if __name__ == "__main__":
    main()
