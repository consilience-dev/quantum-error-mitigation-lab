"""Verify and cache the released Figure 3 no-readout test files for the within-depth comparison."""
import json
from concurrent.futures import ThreadPoolExecutor

from simulation_fetch import ROOT, fetch

MANIFEST = ROOT / 'data/simulation_fig3_sources.json'


if __name__ == '__main__':
    sources = json.loads(MANIFEST.read_bytes())
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda entry: fetch(entry, sources['revision']), sources['files']))
