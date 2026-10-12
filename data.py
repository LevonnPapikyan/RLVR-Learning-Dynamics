"""
The state-tracking task from the paper (arXiv:2602.14872, Definitions 3.2/3.3, Section 7.1).

One problem of length L:
  prompt : L instructions (p_tag[k], act[k])      k = 0..L-1
  answer : L+1 slots      (a_tag[k], y[k])        k = 0..L
  y[0] is the start state, y[k+1] = (y[k] + act[k]) % d, and y[L] is the only thing graded.
  a_tag[k] = codebook[p_tag[k]]: the secret matching rule the model has to discover.

The order of the instructions carries no information. To find the action for slot k the model
must look up the instruction whose tag is the codebook partner of a_tag[k].
"""
import torch

d = 96        # states and actions live in Z_96, as in the paper
n_tags = 64   # number of tags (our choice; must be > the longest length)


def make_codebook(seed=0):
    """A fixed random pairing of tags: codebook[prompt tag] = answer tag."""
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n_tags, generator=g)


def get_batch(Ls, codebook):
    """A batch of problems. Ls is a (B,) tensor of lengths; shorter problems are padded."""
    B, T = len(Ls), int(Ls.max())
    mask = torch.arange(T)[None, :] < Ls[:, None]             # (B, T) True where an instruction exists
    act = torch.rand(B, d).argsort(1)[:, :T]                 # (B, T) all different actions
    tags = torch.rand(B, n_tags).argsort(1)[:, :T + 1]       # (B, T+1) all different tags
    p_tag = tags[:, :T]                                      # prompt tags
    a_tag = codebook[tags]                                   # slot k: partner of tag k. slot L: a spare tag
    y0 = torch.randint(d, (B, 1))
    y = torch.cat([y0, (y0 + (act * mask).cumsum(1)) % d], 1)   # (B, T+1) all true states
    y_final = y[torch.arange(B), Ls]
    return dict(L=Ls, p_tag=p_tag, act=act, a_tag=a_tag, y=y, mask=mask, y_final=y_final)


def show(batch, b=0, max_steps=6):
    """Print problem b of a batch the way the notes write it."""
    L = batch['L'][b].item()
    k = min(L, max_steps)
    more = '  ...' if L > k else ''
    p, g, a, y = (batch[key][b].tolist() for key in ('p_tag', 'act', 'a_tag', 'y'))
    print(f'L = {L}')
    print('prompt (tag, action):', '  '.join(f'(t{p[i]}, +{g[i]})' for i in range(k)) + more)
    print('answer (tag, state) :', '  '.join(f'(t{a[i]}, {y[i]})' for i in range(k + 1)) + more)
    print(f'the model is given the start state {y[0]} and the answer tags; it is graded on {y[L]}')
