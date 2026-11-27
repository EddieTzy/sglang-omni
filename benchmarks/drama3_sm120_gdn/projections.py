"""Compare dual-stream and packed BF16 GDN projections at TP=1 with rotating weights."""
import argparse
import contextlib
import json
import statistics
from pathlib import Path

import torch
from safetensors import safe_open
from torch.nn.functional import linear

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--trace-only', action='store_true')
args = parser.parse_args()
root = args.model
work = args.output
work.mkdir(exist_ok=True, parents=True)
weight_map = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
torch.manual_seed(482)
torch.set_grad_enabled(False)
side = torch.cuda.Stream()
weights = []
with contextlib.ExitStack() as stack:
    files = {}
    for layer in [0, 1, 2, 4, 5, 6]:
        blocks = []
        for suffix in ['qkv', 'z', 'b', 'a']:
            key = f'model.language_model.layers.{layer}.linear_attn.in_proj_{suffix}.weight'
            file = weight_map[key]
            if file not in files:
                files[file] = stack.enter_context(safe_open(root / file, framework='pt', device='cpu'))
            blocks.append(files[file].get_tensor(key))
        large = torch.cat(blocks[:2]).to('cuda')
        small = torch.cat(blocks[2:]).to('cuda')
        packed = torch.cat([large, small])
        weights.append((large, small, packed))

def project(x, w, packed):
    if packed:
        out = linear(x, w[2])
        return out.split([w[0].size(0), w[1].size(0)], dim=1)
    main = torch.cuda.current_stream()
    side.wait_stream(main)
    a = linear(x, w[0])
    with torch.cuda.stream(side):
        b = linear(x, w[1])
    main.wait_stream(side)
    return a, b

rows = {}
for n in ([4] if args.trace_only else [1, 4, 8, 16]):
    x = torch.randn(n, weights[0][0].size(1), device='cuda', dtype=torch.bfloat16)
    graphs = []
    errors = []
    for w in weights:
        a = project(x, w, False)
        b = project(x, w, True)
        errors.append([((p.float()-q.float()).norm()/p.float().norm()).item() for p,q in zip(a,b)])
    for packed in [False, True]:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outputs = [project(x, w, packed) for w in weights]
        graphs.append(g)
    for g in graphs:
        for _ in range(20):
            g.replay()
    torch.cuda.synchronize()
    if args.trace_only:
        for label, graph in zip(['dual', 'packed'], graphs):
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            with torch.profiler.profile(activities=activities) as profile:
                for _ in range(12):
                    graph.replay()
                torch.cuda.synchronize()
            profile.export_chrome_trace(str(work / f'{label}.json.gz'))
        continue
    times = [[], []]
    for trial in range(10):
        for variant in ([0, 1] if trial % 2 == 0 else [1, 0]):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(50):
                graphs[variant].replay()
            end.record()
            end.synchronize()
            times[variant].append(start.elapsed_time(end) * 1000 / (50 * len(weights)))
    rows[n] = dict(dual_us=statistics.median(times[0]), packed_us=statistics.median(times[1]),
                   samples_us=times, relative_errors=errors)
    print(n, rows[n]['dual_us'],rows[n]['packed_us'],flush=True)
if not args.trace_only:
    (work/'projection-results.json').write_text(json.dumps(rows,indent=2)+'\n')
