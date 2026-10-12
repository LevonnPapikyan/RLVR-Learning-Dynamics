"""Two small plotting helpers for the notebooks."""
import matplotlib.pyplot as plt
import torch


def plot_history(hist, title='', key='success', ax=None):
    """One line per length: success (or hit rate) over training steps."""
    ax = ax or plt.gca()
    for L, values in hist[key].items():
        ax.plot(hist['step'], values, lw=2, label=f'L = {L}')
    ax.set_title(title, loc='left'); ax.set_xlabel('training step')
    ax.set_ylabel('sampled success' if key == 'success' else 'attention hit rate')
    ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.legend()
    return ax


def plot_Q(Q, codebook, ax=None):
    """Q with its columns reordered so that the right pairs sit on the diagonal."""
    ax = ax or plt.gca()
    partner = torch.argsort(codebook)            # partner[answer tag] = its prompt tag
    im = ax.imshow(Q.detach()[:, partner], cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_title('Q, reordered so true pairs are on the diagonal', loc='left')
    ax.set_xlabel('prompt tag (reordered)'); ax.set_ylabel('answer tag')
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ax
