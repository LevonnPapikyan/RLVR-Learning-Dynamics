"""
Draws the two figures of the report from the saved results.
  figures/mechanism.{pdf,png} : the mechanism, from results/mechanism.json (run mechanism.py)
  figures/results.{pdf,png}   : the training runs, from results/results.json (run run_single.py, run_mixtures.py)
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'mathtext.fontset': 'stix',
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#888888',
                     'axes.linewidth': 0.6, 'legend.frameon': False, 'lines.linewidth': 1.6,
                     'savefig.bbox': 'tight'})
BLUE, ORANGE = '#2a78d6', '#eb6834'
COLOR = {5: '#86b4ea', 15: '#2a78d6', 35: '#eb6834', 45: '#123c71'}   # one colour per length

def tidy(ax):
    ax.grid(True, color='#e6e6e6', lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=2)

def save(fig, name):
    fig.savefig(f'figures/{name}.pdf'); fig.savefig(f'figures/{name}.png', dpi=200)

# ---------------- the mechanism ----------------
m = json.load(open('results/mechanism.json'))
attn = np.array(m['attn'])
acc = {int(L): np.array(v) for L, v in m['acc'].items()}
s = np.linspace(-1, 6, 400)                                   # score gap between right and wrong pairs
J = {L: np.interp(np.exp(s) / (np.exp(s) + L - 1), attn, acc[L]) ** L for L in acc}
smooth = lambda x: np.convolve(x, np.ones(9) / 9, mode='same')
signal = {L: smooth(np.gradient(J[L], s)) / L for L in acc}   # length normalised learning signal

fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.95))
for L in acc:
    ax[0].plot(attn, acc[L], color=COLOR[L], label=f'$L={L}$')
    ax[1].plot(s, J[L], color=COLOR[L], label=f'$L={L}$')
ax[2].plot(s, (signal[5] + signal[15] + signal[45]) / 3, color=BLUE, label='$\\{5,15,45\\}$')
ax[2].plot(s, (signal[5] + signal[35]) / 2, color=ORANGE, label='$\\{5,35\\}$')
ax[0].set(xlabel='attention on the right instruction', ylabel='one-step accuracy')
ax[1].set(xlabel='score gap $s=q-r$', ylabel='predicted success $J_L$')
ax[2].set(xlabel='score gap $s=q-r$', ylabel='learning signal')
ax[0].set_title('(a) the frozen box is a soft threshold', loc='left', fontsize=8)
ax[1].set_title('(b) longer chains need a larger gap', loc='left', fontsize=8)
ax[2].set_title('(c) gaps in signal become plateaus', loc='left', fontsize=8)
ax[1].legend(loc='center right', fontsize=7); ax[2].legend(loc='upper right', fontsize=7)
for a in ax: tidy(a)
fig.tight_layout(w_pad=1.2); save(fig, 'mechanism')

# ---------------- the training runs ----------------
R = json.load(open('results/results.json'))
runs = lambda prefix: [R[k] for k in sorted(R) if k.startswith(prefix)]

fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.85), sharey=True)
for i, r in enumerate(runs('L5_')):
    ax[0].plot(r['step'], r['success']['5'], color=COLOR[5], lw=1.1, label=f'$L=5$ ({len(runs("L5_"))} seeds)' if i == 0 else None)
for i, r in enumerate(runs('L15_')):
    ax[0].plot(r['step'], r['success']['15'], color=ORANGE, lw=1.1, label=f'$L=15$ ({len(runs("L15_"))} seeds)' if i == 0 else None)
ax[0].set_title('(a) one length at a time', loc='left', fontsize=8)
for a, prefix, title in [(ax[1], 'mix3', '(b) ratio 3, $\\{5,15,45\\}$'), (ax[2], 'mix7', '(c) ratio 7, $\\{5,35\\}$')]:
    for j, r in enumerate(runs(prefix)):
        for L, v in r['success'].items():
            a.plot(r['step'], v, color=COLOR[int(L)], lw=1.3, ls='-' if j == 0 else '--', label=f'$L={L}$' if j == 0 else None)
    a.set_title(title, loc='left', fontsize=8)
for a in ax:
    a.set_xlabel('training step'); a.set_ylim(-0.02, 1.02); a.legend(loc='center right', fontsize=6.5); tidy(a)
ax[0].set_ylabel('sampled success')
fig.tight_layout(w_pad=0.8); save(fig, 'results')
print('saved figures/mechanism and figures/results (pdf and png)')
