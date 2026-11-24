"""Fetch the 60 original transpiled circuit batches at the pinned Git revision."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/upstream"
SOURCES = ROOT / "data/rf_batch_sources.json"
UPSTREAM_DIR = "docs/tutorials/data/ising_init_from_qasm_hardware/shuffled"


def check_blob(raw, entry):
    header = f"blob {len(raw)}\0".encode()
    if len(raw) != entry["bytes"] or hashlib.sha1(header + raw).hexdigest() != entry["blob"]:
        raise ValueError(f"Git blob mismatch: {entry['name']}")


def fetch_batches():
    revision = json.loads((CACHE / "manifest.json").read_text())["commit"]
    entries = json.loads(SOURCES.read_text())
    expected = [f"batch_{i:02d}.pk" for i in range(60)]
    if [e["name"] for e in entries] != expected:
        raise ValueError("Expected all 60 ordered circuit batches")
    def fetch_one(entry):
        path = CACHE / entry["name"]
        if path.exists():
            check_blob(path.read_bytes(), entry)
            return
        url = f"https://raw.githubusercontent.com/qiskit-community/ml-qem/{revision}/{UPSTREAM_DIR}/{path.name}"
        with urlopen(url, timeout=60) as response:
            raw = response.read()
        check_blob(raw, entry)
        path.write_bytes(raw)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch_one, entries))
    print(f"Verified {len(entries)} circuit batches from {revision}")


if __name__ == "__main__":
    fetch_batches()
