"""
Experiment 2: train on mixtures of lengths, as in Section 7.1 of the paper.
  ratio 3: lengths {5, 15, 45}  -> progress is relayed from length to length
  ratio 7: lengths {5, 35}      -> the long length stalls
Results are added to results/results.json. Each run takes 20 to 30 minutes on a laptop CPU.
Usage: python run_mixtures.py [number of seeds, default 1]
"""
import json, os, sys, time
from model import load_box
from train import train, takeoff

path = 'results/results.json'
results = json.load(open(path)) if os.path.exists(path) else {}
box = load_box()
n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 1

for s in range(n_seeds):
    for name, lengths in [('mix3', [5, 15, 45]), ('mix7', [5, 35])]:
        t = time.time()
        _, h = train(box, lengths, steps=3000, seed=s, eval_every=50, verbose=False)
        results[f'{name}_seed{s}'] = dict(step=h['step'], seconds=time.time() - t,
                                          success={str(L): v for L, v in h['success'].items()},
                                          hit={str(L): v for L, v in h['hit'].items()})
        json.dump(results, open(path, 'w'))
        print(f'{name}_seed{s}', 'take-off steps:', {L: takeoff(h, L) for L in h['success']}, flush=True)
