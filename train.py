"""
REINFORCE on the picker Q with outcome-only reward, plus rules for which lengths to train on.

rules (how the B problems of each step get their lengths):
  'fixed'    : uniform over the given lengths (the paper's mixed-difficulty training)
  'oracle'   : in proportion to p(1-p), using the true sampled pass rates from the last evaluation
  'thompson' : no oracle. Discounted success/failure counts per length from the training
               rollouts; each problem goes to the length with the largest sampled p(1-p)
p(1-p) is the variance of a 0/1 reward, a simple measure of how much a length can teach right now.
"""
import numpy as np
import torch
from data import get_batch, make_codebook
from model import rollout


@torch.no_grad()
def pass_rates(Q, box, codebook, lengths, n=256, sharp=2.3):
    """Sampled success rate and attention hit rate for each length."""
    out = {}
    for L in lengths:
        batch = get_batch(torch.full((n,), L), codebook)
        final, _, hit = rollout(Q, box, batch, sharp)
        out[L] = ((final == batch['y_final']).float().mean().item(), hit.mean().item())
    return out


def train(box, lengths, steps=600, rule='fixed', B=512, lr=20.0, momentum=0.95, gamma=0.95,
          sharp=2.3, seed=0, eval_every=25, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    codebook = make_codebook(seed)
    n_tags = len(codebook)
    Q = torch.zeros(n_tags, n_tags, requires_grad=True)
    lengths = list(lengths)
    baseline = {L: 0.0 for L in lengths}                      # one running baseline per length
    succ, fail = np.zeros(len(lengths)), np.zeros(len(lengths))
    hist = dict(step=[], success={L: [] for L in lengths}, hit={L: [] for L in lengths},
                share={L: [] for L in lengths})
    p = np.zeros(len(lengths))
    for step in range(steps + 1):
        if step % eval_every == 0:
            ev = pass_rates(Q, box, codebook, lengths, sharp=sharp)
            p = np.array([ev[L][0] for L in lengths])
            hist['step'].append(step)
            for L in lengths:
                hist['success'][L].append(ev[L][0]); hist['hit'][L].append(ev[L][1])
            if verbose:
                print(f'step {step:4d}  ' + '  '.join(f'L={L}: {ev[L][0]:.2f}' for L in lengths))
        if step == steps:
            break

        # 1) choose a length for each of the B problems
        if rule == 'fixed':
            idx = np.random.randint(len(lengths), size=B)
        elif rule == 'oracle':
            w = p * (1 - p) + 1e-3
            idx = np.random.choice(len(lengths), B, p=w / w.sum())
        elif rule == 'thompson':
            pt = np.random.beta(succ + 1, fail + 1, size=(B, len(lengths)))
            idx = (pt * (1 - pt)).argmax(1)
        counts = np.bincount(idx, minlength=len(lengths))
        if step % eval_every == 0:
            for i, L in enumerate(lengths):
                hist['share'][L].append(counts[i] / B)
        Ls = torch.tensor(np.array(lengths)[idx])

        # 2) roll out and score
        batch = get_batch(Ls, codebook)
        final, logp, _ = rollout(Q, box, batch, sharp)
        reward = (final == batch['y_final']).float()

        # 3) REINFORCE with a per-length baseline and the paper's length normalisation
        b = torch.tensor([baseline[L] for L in Ls.tolist()])
        objective = ((reward - b) * logp / Ls).mean()
        Q.grad = None
        objective.backward()
        with torch.no_grad():
            Q += lr * Q.grad                                   # gradient ascent: more reward

        # 4) bookkeeping: baselines and Thompson counts
        succ *= gamma; fail *= gamma
        for i, L in enumerate(lengths):
            r = reward[Ls == L]
            if len(r):
                baseline[L] = momentum * baseline[L] + (1 - momentum) * r.mean().item()
                succ[i] += r.sum().item(); fail[i] += len(r) - r.sum().item()
    return Q, hist


def takeoff(hist, L, level=0.5):
    """First evaluated step at which length L's success reaches `level` (nan if never)."""
    for s, v in zip(hist['step'], hist['success'][L]):
        if v >= level:
            return s
    return float('nan')
