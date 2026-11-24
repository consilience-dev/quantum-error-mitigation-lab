"""Read the published Song et al. single-qubit GST bars and block histograms."""
import argparse
import hashlib
import io
import json
import math
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/pec_song/single_qubit_inputs.json'
SOURCE = 'https://arxiv.org/src/1812.10903v1'
ARCHIVE_SHA = 'ebd54f0b6f2de5661bb2e7bb16e08d1ef9cf44da2a20196a92367116f5eb3f38'
BLOCKS = 100
SHOTS = 3000
BIN_TOL = 0.00001
IDEAL = np.array([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, -1], [1, -1, 0, 0]])
PRINTED = {
    'Q1': {'q': [-0.0181, -0.0050, 0.0085, 1.0959],
           'raw': {'mean': 0.027, 'sd': 0.016}, 'pec': {'mean': 0.003, 'sd': 0.022}},
    'Q2': {'q': [-0.0288, 0.0043, 0.0078, 1.1101],
           'raw': {'mean': 0.034, 'sd': 0.019}, 'pec': {'mean': 0.007, 'sd': 0.024}},
}


def gram_matrix(page, crop, grid_origin, per_axis):
    """Read both horizontal faces of each bar on the projected 4 x 4 grid."""
    lx, ly, ux, uy = crop
    x0, y0 = grid_origin
    (row_x, row_y), (col_x, col_y) = per_axis
    cells = {(r, c): [] for r in range(4) for c in range(4)}
    for shape in page.curves:
        if not (lx < shape['x0'] and shape['x1'] < ux and ly < shape['top'] < uy
                and isinstance(shape['non_stroking_color'], tuple) and len(shape['pts']) == 5):
            continue
        pts = np.array(shape['pts'][:4])
        # A horizontal face has four different projected x coordinates.
        if np.ptp(pts[:, 0]) < 6 or len(set(np.round(pts[:, 0], 2))) != 4:
            continue
        x, y = pts.mean(axis=0)
        distances = {ij: abs(x - (x0 + ij[0]*row_x + ij[1]*col_x)) for ij in cells}
        nearest = min(distances, key=distances.get)
        if distances[nearest] > 0.03:
            raise ValueError('Unrecognized horizontal face')
        cells[nearest].append(y)
    heights = np.empty((4, 4))
    for (r, c), faces in cells.items():
        if len(faces) != 2:
            raise ValueError(f'Expected two horizontal faces for element {(r,c)}')
        zero_level = y0 + r*row_y + c*col_y
        base, tip = sorted(faces, key=lambda v: abs(v - zero_level))
        if abs(base - zero_level) > 0.02:
            raise ValueError('Bar base disagrees with the projected zero plane')
        heights[r,c] = base - tip
    # The identity observable is exactly one for every input preparation.
    heights /= heights[0].mean()
    np.testing.assert_allclose(heights[0], 1, atol=0.0002, rtol=0)
    heights[0] = 1
    return heights


def histogram(page, base, color, is_raw):
    # S3c axis ticks at -0.1, -0.05, 0, 0.05, 0.1 on the two panels.
    ticks = sorted({round((o['x0'] + o['x1'])/2, 3) for o in page.curves
                    if o['non_stroking_color'] == 0.149 and abs(o['bottom']-base) < 0.01
                    and abs(o['width']-0.1314) < 0.006 and abs(o['height']-0.838) < 0.01})
    if len(ticks) != 5:
        raise ValueError(f'Expected five horizontal axis ticks: {ticks}')
    scale = (ticks[-1]-ticks[0]) / 0.2
    np.testing.assert_allclose((np.array(ticks)-ticks[2])/scale, [-.1,-.05,0,.05,.1], atol=0.00001)
    bars = sorted((o for o in page.curves if o['non_stroking_color'] == color
                   and abs(o['bottom']-base) < 0.01 and 150 < o['x0'] and len(o['pts']) == 5),
                  key=lambda o: o['x0'])
    if not bars:
        raise ValueError('Missing histogram bars')
    height_per_block = sum(o['height'] for o in bars) / BLOCKS
    bins = []
    for bar in bars:
        lower, upper = [(bar[k]-ticks[2])/scale for k in ('x0','x1')]
        height = bar['height']/height_per_block
        if abs(height-round(height)) > .01:
            raise ValueError('A histogram height does not correspond to an integer count')
        b = dict(lower=round(lower, 8), upper=round(upper, 8), blocks=round(height))
        if is_raw:
            b.update(raw_plus_count_min=math.ceil((1 + lower - BIN_TOL)*SHOTS/2),
                     raw_plus_count_max=math.floor((1 + upper + BIN_TOL)*SHOTS/2))
        bins.append(b)
    if sum(b['blocks'] for b in bins) != BLOCKS:
        raise ValueError('Expected one hundred experimental blocks')
    return bins


def score_bounds(bins, arm, printed):
    """Conservative ranges from plotted intervals and the precision of the reported mean."""
    if arm == 'raw':
        intervals = [(2*b['raw_plus_count_min']/SHOTS-1, 2*b['raw_plus_count_max']/SHOTS-1)
                     for b in bins]
    else:
        intervals = [(b['lower']-BIN_TOL,b['upper']+BIN_TOL) for b in bins]
    n = np.array([b['blocks'] for b in bins])
    lo, hi = np.array(intervals).T
    mean_band = np.array([n@lo, n@hi])/BLOCKS
    reported_lo = printed['mean']-.0005
    reported_hi = printed['mean']+.0005
    if mean_band[0] > reported_hi or mean_band[1] < reported_lo:
        raise ValueError('Reported mean falls outside reconstructed histogram')
    # Enclose both population and sample standard deviations. For each block,
    # bound its distance from any mean rounding to the authors' printed value.
    near = np.maximum(0, np.maximum(lo-reported_hi, reported_lo-hi))
    far = np.maximum(abs(hi-reported_lo),abs(lo-reported_hi))
    sd_band = [np.sqrt(n @ near**2 / BLOCKS), np.sqrt(n @ far**2 / (BLOCKS-1))]
    if sd_band[0] > printed['sd']+.0005 or sd_band[1] < printed['sd']-.0005:
        raise ValueError('Reported SD falls outside reconstructed histogram')
    return dict(mean_bounds=np.round(mean_band,8).tolist(), sd_bounds=np.round(sd_band,8).tolist())


def extract(path):
    import pdfplumber

    if hashlib.sha256(path.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError('Unexpected arXiv source archive')
    with tarfile.open(path) as tar:
        figs = {name: tar.extractfile(name).read() for name in ('fig3_supp.pdf','figure2.pdf')}
    qubits = {}
    with pdfplumber.open(io.BytesIO(figs['fig3_supp.pdf'])) as sup:
        page = sup.pages[0]
        for qubit, vertical_offset, baseline in [('Q1',0,130.052), ('Q2',75.82,163.153)]:
            gram = gram_matrix(page, (10,vertical_offset,110,vertical_offset+83),
                               (32.73,51.5263+vertical_offset), ((9.6725,4.1629),(9.505,-4.2356)))
            q = np.linalg.solve(gram.T, IDEAL[3])
            np.testing.assert_allclose(q, PRINTED[qubit]['q'], rtol=0, atol=.0004)
            item = {'gram':np.round(gram,7).tolist(), 'q_from_figure_gram':np.round(q,7).tolist(),
                    'reported':PRINTED[qubit]}
            for arm, color in [('raw',(0.,0.,1.)), ('pec',(1.,0.,0.))]:
                bins = histogram(page, baseline, color, arm=='raw')
                item[f'{arm}_bins'] = bins
                item[f'{arm}_histogram_check'] = score_bounds(bins, arm, PRINTED[qubit][arm])
            qubits[qubit] = item
    with pdfplumber.open(io.BytesIO(figs['figure2.pdf'])) as main:
        second = gram_matrix(main.pages[0], (135,25,248,115), (151.66,63.14),
                             ((13.38,5.7643),(13.15,-5.866)))
    difference = float(np.max(abs(np.array(qubits['Q1']['gram'])-second)))
    if difference > .00035:
        raise ValueError('Two independently rendered Q1 matrices disagree')
    return dict(paper='https://doi.org/10.1126/sciadv.aaw5686', source=SOURCE, source_sha256=ARCHIVE_SHA,
                figure_sha256={n:hashlib.sha256(b).hexdigest() for n,b in figs.items()},
                shots_per_block=SHOTS, blocks_per_histogram=BLOCKS, histogram_axis_tolerance=BIN_TOL,
                qubits=qubits, ideal_gram=IDEAL.tolist(), gram_rows=['I','X','Y','Z'],
                gram_columns=['|0>','|1>','|+X>','|-Y>'],
                q1_max_gram_difference_between_main_and_supplement=round(difference,8),
                provenance='Gram entries are reconstructed from vector bar geometry in supplementary figure S3a. '
                           'The Q1 matrix is cross-checked against the independently rendered main figure 2b. '
                           'Histogram bin counts are exact integers recovered from S3c. The printed means, '
                           'standard deviations and measurement decomposition weights come from S3b/c. '
                           'Per-block values and their ordering are unavailable. The single-qubit test state '
                           'is also present as the fourth GST calibration state; repeated blocks are not distinct targets.')


def verify_publication(archive, supplement):
    import pdfplumber
    from pypdf import PdfReader

    official_url = 'https://pmc-oa-opendata.s3.amazonaws.com/PMC6731091.1/aaw5686_SM.pdf'
    official_sha = hashlib.sha256(supplement.read_bytes()).hexdigest()
    if official_sha != '0c766a315bb3cd1e3d48b87b5aabeb597c61a4ac8167ad5ef8e22522c65fc3cd':
        raise ValueError('Unexpected publisher-deposited supplement')
    with tarfile.open(archive) as tar:
        source_s3 = PdfReader(io.BytesIO(tar.extractfile('fig3_supp.pdf').read())).pages[0]
        main_pdf = tar.extractfile('figure2.pdf').read()
    # Final article's supplementary page 3 contains both figures S3 and S4.
    journal_s3 = PdfReader(supplement).pages[2]['/Resources']['/XObject']['/Im3']

    def graphic_hashes(node, base=''):
        streams = {}
        for name, ref in node['/Resources'].get('/XObject', {}).items():
            graphic = ref.get_object()
            key = base + name
            streams[key] = hashlib.sha256(graphic.get_data()).hexdigest()
            if graphic['/Subtype'] == '/Form':
                streams.update(graphic_hashes(graphic, key))
        return streams

    native_stream = journal_s3.get_data()
    source_stream = source_s3.get_contents().get_data()
    streams = graphic_hashes(source_s3)
    if source_stream != native_stream or graphic_hashes(journal_s3) != streams or len(streams) != 6:
        raise ValueError('The published and arXiv figure S3 drawing streams differ')
    with pdfplumber.open(io.BytesIO(main_pdf)) as main:
        matrix = gram_matrix(main.pages[0], (135,25,248,115), (151.66,63.14),
                             ((13.38,5.7643),(13.15,-5.866)))
    return dict(published_supplement=official_url, published_supplement_sha256=official_sha,
                figure_page=3, figure='S3', arxiv_source=SOURCE,
                shared_drawing_stream_bytes=len(native_stream),
                shared_drawing_stream_sha256=hashlib.sha256(native_stream).hexdigest(),
                shared_graphic_streams_sha256=dict(sorted(streams.items())),
                interpretation='The source and published supplements contain identical S3 drawing instructions '
                               'and identical nested graphic streams for the Gram bars, histogram bins, axes '
                               'and plotted PEC coefficients. The main figure 2b independently renders Q1.',
                q1_gram_from_main_figure=np.round(matrix,7).tolist())


def save_or_check(path, report, check):
    if check:
        if json.loads(path.read_bytes()) != report:
            raise ValueError(f'Committed reconstruction differs in {path.name}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('archive', type=Path)
    cli.add_argument('--published-supplement', type=Path, help='Journal supplement in the PubMed Central open archive')
    cli.add_argument('--check', action='store_true')
    args = cli.parse_args()
    report = extract(args.archive)
    save_or_check(OUT, report, args.check)
    if args.published_supplement:
        journal = verify_publication(args.archive, args.published_supplement)
        save_or_check(OUT.with_name('publication_check.json'), journal, args.check)
    print('Verified both published GST matrices and four histograms of 100 experimental blocks.')
