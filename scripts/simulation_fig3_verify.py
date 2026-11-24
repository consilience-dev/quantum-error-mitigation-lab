"""Rescore saved Figure 3 predictions, check identities, and estimate the paired TabPFN–RF difference."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np

from simulation_fetch import ROOT
from simulation_fig3_metrics import metrics
from verify_figure4 import DIGEST as WORKBOOK_SHA256

OUT = ROOT / 'reports/simulation/fig3'


def load_rows(filename, expected_hash):
    path = OUT/filename
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f'Unexpected prediction hash for {filename}')
    with gzip.open(path,'rt',newline='') as f:
        return list(csv.DictReader(f))


def matrix(rows, prefix):
    return np.array([[float(row[f'{prefix}_{q}']) for q in range(4)] for row in rows])


def check_scores(actual, expected):
    if set(actual) != set(expected):
        raise ValueError('Missing score field')
    for key, value in actual.items():
        if isinstance(value,dict):
            check_scores(value,expected[key])
        else:
            np.testing.assert_allclose(value,expected[key],rtol=1e-12,atol=1e-14)


def verify_additional_models(target, depth, novel, ids, reference_hash):
    for name in ('nori', 'tabicl'):
        report = json.loads((OUT / f'{name}.json').read_bytes())
        if (report['reference_artifact_sha256'] != reference_hash or report['train_rows'] != 4500
                or report['test_rows'] != len(ids) or report['inputs'] != 58
                or not report['full_context_verified_each_query_batch']):
            raise ValueError(f'Unexpected original-split configuration for {name}')
        rows = load_rows(f'{name}.csv.gz', report['predictions_sha256'])
        if [(r['row_id'], int(r['paper_row_id'])) for r in rows] != ids:
            raise ValueError(f'Misaligned prediction rows for {name}')
        recomputed = metrics(matrix(rows, 'pred'), target, depth, novel)
        check_scores(json.loads(json.dumps(recomputed)), report['scores'])


def workbook_check(path, ref):
    if hashlib.sha256(path.read_bytes()).hexdigest() != WORKBOOK_SHA256:
        raise ValueError('Unexpected upstream workbook')
    with ZipFile(path) as archive:
        names = [''.join(c.itertext()) for c in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        sheet = ET.fromstring(archive.read('xl/worksheets/sheet2.xml'))
        cells = {c.get('r'):names[int(c.findtext('{*}v'))] for c in sheet.findall('.//{*}c') if c.get('t') == 's'}
    if cells['A1'] != 'Top row; left column':
        raise ValueError('Expected the Figure 3 top-left panel')
    for source, start in [('noisy',3),('zne',35),('rfr_list',67)]:
        if cells[f'A{start+30}'].split(',')[0] != 'Name: L2_'+source:
            raise ValueError('Unexpected Figure 3 workbook series')
        for d in range(15):
            step, value = cells[f'A{start+d}'].split()
            expected = ref['scores'][source]['by_depth'][str(d)]['mean_circuit_l2']
            if int(step) != d or abs(float(value)-expected) > 5.01e-7:
                raise ValueError(f'Workbook disagrees for {source} at depth {d}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook',type=Path,help='Also validate all 45 published panel values in the ML-QEM source workbook')
    parser.add_argument('--check',action='store_true',help='Check the committed paired report in place')
    args = parser.parse_args()
    ref = json.loads((OUT/'reference.json').read_bytes())
    pfn = json.loads((OUT/'tabpfn_v2.json').read_bytes())
    if hashlib.sha256((OUT/'reference.json').read_bytes()).hexdigest() != pfn['reference_artifact_sha256']:
        raise ValueError('TabPFN report uses a different reference')
    if args.workbook:
        workbook_check(args.workbook,ref)
    a = load_rows('reference.csv.gz',ref['predictions_sha256'])
    b = load_rows('tabpfn_v2.csv.gz',pfn['predictions_sha256'])
    ids = [(r['row_id'],int(r['paper_row_id'])) for r in a]
    if ids != [(r['row_id'],int(r['paper_row_id'])) for r in b] or len(set(ids)) != 4500:
        raise ValueError('Misaligned prediction IDs')
    depth = np.array([int(k.split('/')[1].split('_')[1]) for k,_ in ids])
    novel = np.array([r['seen_in_train'] == '0' for r in a])
    groups = {r['compiled_hash'] for r in a if r['seen_in_train'] == '0'}
    if not np.array_equal(novel,depth != 0) or len(groups) != 4200 or novel.sum() != 4200:
        raise ValueError('Unexpected identity-based sensitivity population')
    y, tab, rf = matrix(a,'ideal'), matrix(b,'pred'), matrix(a,'rfr_list')
    for k in ref['scores']:
        actual = metrics(matrix(a,k),y,depth,novel)
        check_scores(json.loads(json.dumps(actual)),ref['scores'][k])
    check_scores(json.loads(json.dumps(metrics(tab,y,depth,novel))),pfn['scores'])
    verify_additional_models(y, depth, novel, ids, pfn['reference_artifact_sha256'])
    # Keep the paired loss differences together when resampling a circuit.
    ae = np.abs(tab-y).mean(axis=1)-np.abs(rf-y).mean(axis=1)
    l2 = np.linalg.norm(tab-y,axis=1)-np.linalg.norm(rf-y,axis=1)
    loss_differences = np.column_stack((ae,l2))
    rng = np.random.default_rng(0)
    draws = 10000
    boot = np.zeros((draws,2))
    for d in range(1,15):
        sample = loss_differences[(depth == d) & novel]
        if len(sample) != 300:
            raise ValueError('Unbalanced sensitivity population')
        for start in range(0,draws,100):
            index = rng.integers(0,len(sample),size=(100,len(sample)))
            boot[start:start+100] += sample[index].sum(axis=1) / int(novel.sum())
    confidence = np.quantile(boot,[.025,.975],axis=0)
    result = dict(contrast='Frozen TabPFN v2 minus original archived random forest; lower favors TabPFN',
                  population='4,200 distinct test circuit fingerprints unseen in training, depths 1 to 14',
                  procedure='Paired circuit bootstrap within each depth, conditioned on training data, model and simulated noise setting',
                  bootstrap_seed=0, draws=draws,
                  mean_circuit_mae=dict(estimate=ae[novel].mean(),ci95=confidence[:,0].tolist()),
                  mean_circuit_l2=dict(estimate=l2[novel].mean(),ci95=confidence[:,1].tolist()),
                  official_mae_difference=ae.mean(),
                  official_relative_mae_reduction_vs_rf=float(1-pfn['scores']['official']['mean_circuit_mae']/ref['scores']['rfr_list']['official']['mean_circuit_mae']),
                  official_depths_lower_mae_than_rf=[int(d) for d in np.unique(depth) if ae[depth==d].mean() < 0])
    path = OUT/'paired_rf.json'
    if args.check:
        expected = json.loads(path.read_bytes())
        for key in ('mean_circuit_mae','mean_circuit_l2','official_mae_difference','official_relative_mae_reduction_vs_rf'):
            cur, stored = result.pop(key), expected.pop(key)
            if isinstance(cur, dict):
                for stat in ('estimate','ci95'):
                    np.testing.assert_allclose(cur[stat],stored[stat],rtol=1e-12,atol=1e-14)
            else:
                np.testing.assert_allclose(cur,stored,rtol=1e-12,atol=1e-14)
        if result != expected:
            raise ValueError('Paired comparison metadata changed')
    else:
        path.write_text(json.dumps(result,indent=2)+'\n')
    print('Verified six original methods, three frozen models and the paired TabPFN–RF comparison.',flush=True)


if __name__ == '__main__':
    main()
