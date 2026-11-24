"""Retrieve exact upstream files using the committed provenance manifest."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def verify(raw, record):
    if len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
        raise ValueError(f"Upstream checksum mismatch: {record['upstream_path']}")


def main():
    manifest = json.loads((ROOT / 'data/upstream/manifest.json').read_text())
    for record in manifest['files']:
        destination = ROOT / record['local_path']
        raw = destination.read_bytes() if destination.exists() else urlopen(record['url'], timeout=60).read()
        verify(raw, record)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        print(f"Verified {record['local_path']}")


if __name__ == '__main__':
    main()
