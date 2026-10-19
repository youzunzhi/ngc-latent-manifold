"""NGC minus the L2 cost by the physical distance from start to goal.

The pairs of one task's evaluations (every offset and pair set, the same
planner seeds for both costs) are grouped by the physical distance of
``analysis.physical`` from the start state to the goal state: by quintile
over all offsets, and by tercile within each offset, where the budget is
fixed. Each group reports both success rates and their paired difference;
its interval resamples episodes, since one episode can supply a pair at
several offsets.

    python -m analysis.strata --task cube --dataset cube_single_expert.h5 \\
        --l2 results/cube/l2-*.json --ngc results/cube/ngc-*.json \\
        --out results/strata/cube.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from analysis import physical, success

POOLED_BINS = 5
WITHIN_OFFSET_BINS = 3


def bins(distance, offsets, episodes, l2, ngc, count, rng):
    edges = np.quantile(distance, np.linspace(0, 1, count + 1))
    rows = []
    for b in range(count):
        last = b == count - 1
        upper = distance <= edges[b + 1] if last else distance < edges[b + 1]
        mask = (distance >= edges[b]) & upper
        rows.append(
            {
                'range': [float(edges[b]), float(edges[b + 1])],
                'median': float(np.median(distance[mask])),
                'pairs_by_offset': {
                    str(o): int((offsets[mask] == o).sum())
                    for o in np.unique(offsets)
                },
                **success.difference(l2[mask], ngc[mask], episodes[mask], rng),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--task', choices=physical.TASKS, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--l2', nargs='+', type=Path, required=True)
    parser.add_argument('--ngc', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), f'{args.out} exists'
    l2_records = success.pair_success(args.l2)
    keys, l2, ngc = success.paired(l2_records, success.pair_success(args.ngc))
    records = [l2_records[k] for k in keys]
    episodes = np.array([r['episode_id'] for r in records])
    starts = np.array([r['start_step'] for r in records])
    offsets = np.array([r['goal_offset'] for r in records])
    rows = physical.episode_rows(args.dataset, episodes, starts)
    start = physical.read_states(args.dataset, args.task, rows)
    goal = physical.read_states(args.dataset, args.task, rows + offsets)
    distance = np.array(
        [
            physical.distance(
                args.task,
                {c: v[[i]] for c, v in start.items()},
                {c: v[[i]] for c, v in goal.items()},
            )[0, 0]
            for i in range(len(records))
        ]
    )
    rng = np.random.default_rng(success.BOOTSTRAP_SEED)
    result = {
        'task': args.task,
        'unit': physical.UNIT[args.task],
        'pairs': len(records),
        'distance_range': [float(distance.min()), float(distance.max())],
        'pooled': bins(distance, offsets, episodes, l2, ngc, POOLED_BINS, rng),
        'within_offset': {
            str(o): bins(
                distance[offsets == o],
                offsets[offsets == o],
                episodes[offsets == o],
                l2[offsets == o],
                ngc[offsets == o],
                WITHIN_OFFSET_BINS,
                rng,
            )
            for o in np.unique(offsets)
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
    for row in result['pooled']:
        print(
            f'{args.task} distance {row["median"]:.3g} {result["unit"]}: '
            f'L2 {row["first"]:.1f}, NGC {row["second"]:.1f}, '
            f'delta {row["delta"]:+.1f} {row["ci95"]}'
        )


if __name__ == '__main__':
    main()
