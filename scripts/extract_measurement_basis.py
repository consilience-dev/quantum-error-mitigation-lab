"""Save measured Pauli basis by original row ID from the pinned logical sources."""
import hashlib
import json
import pickle

from freeze_clean_split import CACHE, ROOT, SOURCES, fetch_one


def main():
    revision = json.loads((CACHE / "manifest.json").read_text())["commit"]
    entries = json.loads(SOURCES.read_text())
    values = []
    for i, entry in enumerate(entries):
        expected = ("train" if i < 10 else "val") + f"/step_{i % 10:02d}.pk"
        if not entry["path"].endswith("/" + expected):
            raise ValueError("Unexpected original row order")
        path = fetch_one(entry, revision)
        rows = pickle.loads(path.read_bytes())
        if len(rows) != (200 if i < 10 else 100):
            raise ValueError("Unexpected depth population")
        values.extend(str(row["basis"]).upper() for row in rows)
    if len(values) != 3000 or set(values) != {"X", "Y", "Z"}:
        raise ValueError(f"Unexpected measurement basis: {set(values)}")
    report = {
        "source_manifest_sha256": hashlib.sha256(SOURCES.read_bytes()).hexdigest(),
        "upstream_commit": revision, "basis_by_unshuffled_row_id": "".join(values),
    }
    (ROOT / "data/measurement_basis.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
