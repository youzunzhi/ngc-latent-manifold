"""Time to score one CEM pool with the L2 cost and with NGC.

The work of either cost does not depend on the candidate values (NGC
measures the distance from every candidate to every graph node, sorts it,
and takes a minimum over anchor pairs), so the pools are graph nodes
perturbed by Gaussian noise of the scale of a nearest-neighbor distance.
Each cost scores the same pools after a warm-up, timed with CUDA events;
the goal changes every ``POOLS_PER_GOAL`` pools, as it does once per
episode in planning. The time of a whole solve is recorded by
``scripts/evaluate.py`` (``solve_seconds``).

    python -m analysis.timing --nodes nodes/cube-seed0.npz --neighbors 15 \\
        --anchors 4 --samples 300 --out results/timing/cube-seed0.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch

from ngc import NGCCost, build_graph

WARMUP = 20
POOLS = 200
POOLS_PER_GOAL = 20
SEED = 0


def median_milliseconds(cost, pools, goals):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for i, pool in enumerate(pools):
        info = {
            'predicted_emb': pool[None, :, None],
            'goal_emb': goals[i // POOLS_PER_GOAL][None, None],
        }
        start.record()
        cost(info)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(np.median(times[WARMUP:]))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--nodes', type=Path, required=True)
    parser.add_argument('--neighbors', type=int, required=True)
    parser.add_argument('--anchors', type=int, required=True)
    parser.add_argument('--samples', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), f'{args.out} exists'
    assert torch.cuda.is_available(), 'timing runs on CUDA'
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    latents = np.load(args.nodes)['latents']
    nodes, graph_distance, _ = build_graph(latents, args.neighbors, 'mutual')
    scale = float(
        np.median(
            np.sort(
                np.linalg.norm(latents[:256, None] - latents[None], axis=-1),
                axis=1,
            )[:, 1]
        )
    )
    rng = np.random.default_rng(SEED)
    count = WARMUP + POOLS
    picks = rng.integers(0, len(latents), size=(count, args.samples))
    pools = latents[picks] + rng.normal(
        0,
        scale / np.sqrt(latents.shape[1]),
        size=(*picks.shape, latents.shape[1]),
    )
    pools = torch.as_tensor(pools, dtype=torch.float32, device='cuda')
    goal_ids = rng.integers(0, len(latents), size=count // POOLS_PER_GOAL + 1)
    goals = torch.as_tensor(
        latents[goal_ids], dtype=torch.float32, device='cuda'
    )
    ngc = NGCCost(nodes, graph_distance, args.neighbors, args.anchors).cuda()
    result = {
        'nodes': str(args.nodes),
        'node_count': len(nodes),
        'samples': args.samples,
        'pools': POOLS,
        'device': torch.cuda.get_device_name(),
        'l2_ms_per_pool': median_milliseconds(
            swm.planning.GoalMSE(), pools, goals
        ),
        'ngc_ms_per_pool': median_milliseconds(ngc, pools, goals),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(
        f'{result["device"]}: L2 {result["l2_ms_per_pool"]:.3f} ms, '
        f'NGC {result["ngc_ms_per_pool"]:.3f} ms per pool of {args.samples}'
    )


if __name__ == '__main__':
    main()
