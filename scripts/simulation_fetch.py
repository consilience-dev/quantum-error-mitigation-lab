"""Download the immutable h12 tutorial and its original train/validation circuit records."""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/upstream/simulation'
MANIFEST = ROOT / 'data/simulation_h12_sources.json'


def verified(path, source):
    if not path.is_file() or path.stat().st_size != source['size']:
        return False
    blob = hashlib.sha1(f"blob {source['size']}\0".encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            blob.update(chunk)
    return blob.hexdigest() == source['git_blob_sha1']


def fetch(source, revision):
    path = CACHE / source['path']
    if verified(path, source):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + '.part')
    url = f"https://raw.githubusercontent.com/qiskit-community/ml-qem/{revision}/{source['path']}"
    for attempt in range(3):
        try:
            with urlopen(url, timeout=75) as stream, part.open('wb') as out:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    out.write(chunk)
            if not verified(part, source):
                raise ValueError(f"Source digest mismatch: {source['path']}")
            part.replace(path)
            print('Verified', source['path'], flush=True)
            return
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def main():
    manifest = json.loads(MANIFEST.read_bytes())
    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(lambda src: fetch(src, manifest['revision']), manifest['files']))


if __name__ == '__main__':
    main()
