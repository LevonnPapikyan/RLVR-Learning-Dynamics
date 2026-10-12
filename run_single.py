"""
Experiment 1: train on one length at a time.
  L = 5  alone, five seeds  -> plateau, then a sudden jump
  L = 15 alone, two seeds   -> stays flat
Results are added to results/results.json. About ten minutes on a laptop CPU.
"""
import json, os, time
from model import load_box
from train import train, takeoff

path = 'results/results.json'
results = json.load(open(path)) if os.path.exists(path) else {}
box = load_box()

def record(name, hist, seconds):
    results[name] = dict(step=hist['step'], seconds=seconds,
                         success={str(L): v for L, v in hist['success'].items()},
                         hit={str(L): v for L, v in hist['hit'].items()})
    json.dump(results, open(path, 'w'))
    print(name, 'take-off steps:', {L: takeoff(hist, L) for L in hist['success']}, f'({seconds:.0f}s)', flush=True)

for s in range(5):
    t = time.time(); _, h = train(box, [5], steps=600, seed=s, verbose=False); record(f'L5_seed{s}', h, time.time() - t)
for s in range(2):
    t = time.time(); _, h = train(box, [15], steps=1000, seed=s, eval_every=50, verbose=False); record(f'L15_seed{s}', h, time.time() - t)
