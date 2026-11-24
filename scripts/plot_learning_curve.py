"""Plot ML-QEM calibration curves on the shared 900-circuit test set."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = [18, 45, 90, 180, 270, 450, 500]
NORI_BUDGETS = [18, 90, 500]


def curve(ax, rows, model, arm, budgets, color, marker, label):
    y = []
    for budget in budgets:
        draws = sorted((int(r['seed']), float(r['mean_circuit_l2'])) for r in rows
                       if r['model'] == model and int(r['budget']) == budget and r['inputs'] == arm)
        if [seed for seed, _ in draws] != list(range(5)):
            raise ValueError(f'Expected five draws for {model}/{arm} at {budget}')
        y.append([value for _, value in draws])
    values = np.array(y)
    ax.fill_between(budgets, values.min(axis=1), values.max(axis=1), color=color, alpha=0.10, linewidth=0)
    ax.plot(budgets, values.mean(axis=1), color=color, marker=marker, lw=1.7, label=label)


def main():
    with (ROOT / 'reports/scores.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == 'clean']
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharey=True, layout='constrained')
    axes[0].set_title('Circuit descriptors and measurements')
    axes[1].set_title('Measurements alone')
    curve(axes[0], rows, 'Random forest', 'full', BUDGETS, '#46628b', 'o', 'Random forest')
    curve(axes[0], rows, 'TabPFN v2', 'full', BUDGETS, '#a04529', 's', 'TabPFN v2')
    curve(axes[0], rows, 'Nori 1.5', 'full', NORI_BUDGETS, '#007d72', '^', 'Nori 1.5')
    curve(axes[1], rows, 'Affine', 'full', BUDGETS, '#77538b', 'o', 'Per-observable affine')
    curve(axes[1], rows, 'Multivariate affine', 'full', BUDGETS, '#ad6500', 's', 'Four-input affine')
    curve(axes[1], rows, 'Nori 1.5', 'measurements_only', NORI_BUDGETS, '#007d72', '^', 'Nori 1.5')
    axes[0].set_ylabel('Mean four-observable circuit error')
    for ax in axes:
        for name, style in [('Hardware', '--'), ('ZNE', ':')]:
            row, = [r for r in rows if r['model'] == name]
            ax.axhline(float(row['mean_circuit_l2']), ls=style, color='#555555', lw=1.1, label=name)
        ax.set_xlabel('Unique calibration circuits (log scale)')
        ax.set_xscale('log')
        ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
        ax.tick_params(axis='x', labelrotation=45)
        ax.minorticks_off()
        ax.spines[['right','top']].set_visible(False)
        ax.grid(axis='y', color='#dddddd', lw=.7)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)
    fig.savefig(ROOT / 'reports/learning_curve.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
