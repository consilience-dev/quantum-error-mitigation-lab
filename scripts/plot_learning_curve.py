"""Plot the completed unique-circuit learning curves from the verified scorecard."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = [18, 45, 90, 180, 270, 450, 500]


def main():
    with (ROOT / 'reports/scores.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == 'clean']
    fig, ax = plt.subplots(figsize=(8.0,4.5), layout='constrained')
    for model, color, marker in [('Random forest','#46628b','o'),('TabPFN v2','#a04529','s')]:
        y = []
        for budget in BUDGETS:
            draws = sorted((int(r['seed']),float(r['mean_circuit_l2'])) for r in rows
                           if r['model'] == model and int(r['budget']) == budget and r['inputs'] == 'full')
            if [seed for seed,_ in draws] != list(range(5)):
                raise ValueError(f'Expected all five draws for {model} at {budget}')
            y.append([value for _,value in draws])
        values = np.asarray(y)
        ax.fill_between(BUDGETS,values.min(axis=1),values.max(axis=1),color=color,alpha=0.14,linewidth=0)
        ax.plot(BUDGETS,values.mean(axis=1),color=color,marker=marker,lw=2,label=model)
    for model, style, color in [('Hardware','--','#555555'),('ZNE',':','#737373')]:
        row, = [r for r in rows if r['model'] == model]
        ax.axhline(float(row['mean_circuit_l2']),ls=style,color=color,lw=1.5,label=model)
    ax.set(xlabel='Unique calibration circuits (log scale)',ylabel='Mean four-observable circuit error')
    ax.set_xscale('log')
    ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
    ax.tick_params(axis='x',labelrotation=35)
    ax.minorticks_off()
    ax.spines[['right','top']].set_visible(False)
    ax.grid(axis='y',color='#dadada',lw=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False,ncol=2)
    fig.savefig(ROOT / 'reports/learning_curve.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
