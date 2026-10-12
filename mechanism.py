"""
The mechanism without any RL: how accurate is the frozen box when only part of the
attention sits on the right transition? Saves results/mechanism.json for make_figures.py.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
from model import load_box, box_input
from data import d

box = load_box()
torch.manual_seed(0)

@torch.no_grad()
def step_accuracy(a, L, n=4000, sharp=2.3):
    """P(next state is right) when attention a is on the right action and 1 - a is spread over L - 1 others."""
    acts = torch.rand(n, d).argsort(1)[:, :L]                  # column 0 is the right action
    w = torch.full((n, L), (1 - a) / max(L - 1, 1)); w[:, 0] = a
    blend = (w[:, :, None] * F.one_hot(acts, 2 * d).float()).sum(1)
    y = torch.randint(d, (n,))
    p = F.softmax(sharp * box(box_input(y, blend)), -1)
    return p[torch.arange(n), (y + acts[:, 0]) % d].mean().item()

attn = np.linspace(0, 1, 41)
out = {'attn': attn.tolist(), 'acc': {L: [step_accuracy(a, L) for a in attn] for L in [5, 15, 35, 45]}}
json.dump(out, open('results/mechanism.json', 'w'))
print('saved results/mechanism.json')
