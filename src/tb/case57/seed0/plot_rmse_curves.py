"""Replot exported GENCO TensorBoard curves, with boxed upper-right legends."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = pd.read_csv(ROOT / 'rmse_curves_long.csv')

SERIES = [
    ('k2p5_s40_d10_a60_g2_normal', 'aging: weak, perturbation: enabled', '#e377c2'),
    ('k2p5_s40_d50_a60_g2_normal', 'aging: normal, perturbation: enabled', '#8c564b'),
    ('k2p5_s40_d10_a60_g2_all_out', 'aging: weak, perturbation: disabled', '#bcbd22'),
    ('k2p5_s40_d50_a60_g2_all_out', 'aging: normal, perturbation: disabled', '#7f7f7f'),
]
PLOTS = [
    ('epoch/model_rmse', 'tb_model_rmse', 'GENCO model RMSE'),
    ('epoch/reactance_mismatch_rmse', 'tb_reactance_mismatch_rmse', 'Reactance mismatch RMSE (p.u.)'),
    ('epoch/resistance_mismatch_rmse', 'tb_resistance_mismatch_rmse', 'Resistance mismatch RMSE (p.u.)'),
]

for metric, stem, ylabel in PLOTS:
    fig, ax = plt.subplots(figsize=(11.8, 4.7))
    for experiment, legend_label, color in SERIES:
        df = DATA.loc[(DATA['metric'] == metric) & (DATA['experiment'] == experiment)]
        df = df.sort_values('epoch')
        ax.plot(df['epoch'], df['value'], label=legend_label, color=color, linewidth=2.5)
    ax.set_xlim(0, 250)
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, color='#d8d8d8', alpha=0.43, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.9)
    ax.spines['bottom'].set_linewidth(0.9)
    leg = ax.legend(loc='upper right', ncol=1, fontsize=10, frameon=True,
                    facecolor='white', edgecolor='black', framealpha=1,
                    fancybox=False, borderpad=0.65, labelspacing=0.42,
                    handlelength=2.5, borderaxespad=0.7)
    leg.get_frame().set_linewidth(0.85)
    fig.tight_layout(pad=1.0)
    fig.savefig(ROOT / (stem + '.pdf'), bbox_inches='tight')
    fig.savefig(ROOT / (stem + '.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(ROOT / (stem + '.pdf'))
