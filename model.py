"""
The model from the paper's theory and synthetic experiments.

  box    : an MLP that does one step, (state, action) -> next state. Trained once, then frozen.
  picker : a table Q[answer tag, prompt tag], starting at zero. At slot k the model attends over
           the instructions with weights softmax(Q[a_tag[k], p_tag]), blends their actions, and
           feeds the blend to the box. Q is the only thing RL trains.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import d, n_tags


def box_input(state, action_vec):
    """What the box reads: half the state, half the action, in one space of size 2d."""
    return 0.5 * (F.one_hot(state + d, 2 * d).float() + action_vec)


def train_box(hidden=512, steps=1000, lr=1e-3, seed=1337, verbose=True):
    """Teach the MLP every (state, action) pair: 96 * 96 = 9216 examples."""
    torch.manual_seed(seed)
    ys, gs = torch.meshgrid(torch.arange(d), torch.arange(d), indexing='ij')
    ys, gs = ys.reshape(-1), gs.reshape(-1)
    X = box_input(ys, F.one_hot(gs, 2 * d).float())
    labels = (ys + gs) % d
    box = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU(), nn.Linear(hidden, d))
    opt = torch.optim.Adam(box.parameters(), lr=lr)
    for step in range(steps + 1):
        logits = box(X)
        loss = F.cross_entropy(logits, labels)
        if verbose and step % 200 == 0:
            acc = (logits.argmax(1) == labels).float().mean().item()
            print(f'step {step:4d}  loss {loss.item():.4f}  accuracy {acc:.4f}')
        opt.zero_grad(); loss.backward(); opt.step()
    return box


def load_box(path='box.pt'):
    """Load the frozen box, training and saving it the first time."""
    box = nn.Sequential(nn.Linear(2 * d, 512), nn.ReLU(), nn.Linear(512, d))
    if os.path.exists(path):
        box.load_state_dict(torch.load(path))
    else:
        box = train_box()
        torch.save(box.state_dict(), path)
    for p in box.parameters():
        p.requires_grad = False
    return box.eval()


def rollout(Q, box, batch, sharp=2.3, greedy=False):
    """The model writes its chain for every problem.
    Returns the final state (B,), the log-probability of the chain (B,), and the attention
    hit rate (B,): how often the biggest attention weight landed on the right instruction."""
    L, p_tag, a_tag, mask = batch['L'], batch['p_tag'], batch['a_tag'], batch['mask']
    B, T = p_tag.shape
    act_vec = F.one_hot(batch['act'], 2 * d).float()               # (B, T, 2d)
    state = batch['y'][:, 0].clone()
    logp, hits = torch.zeros(B), torch.zeros(B)
    for k in range(T):
        alive = k < L                                              # problems that still have a step k
        scores = Q[a_tag[:, k]].gather(1, p_tag).masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=1)                            # (B, T) where slot k looks
        mix = (attn[:, :, None] * act_vec).sum(1)                  # (B, 2d) the blended action
        logits = sharp * box(box_input(state, mix))                # (B, d) next-state scores
        dist = torch.distributions.Categorical(logits=logits)
        nxt = logits.argmax(1) if greedy else dist.sample()
        logp = logp + dist.log_prob(nxt) * alive
        state = torch.where(alive, nxt, state)
        hits = hits + ((attn.argmax(1) == k) & alive).float()
    return state, logp, hits / L
