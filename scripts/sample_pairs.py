"""Draw a pair set from a dataset for ``scripts/evaluate.py``.

``--rule protocol``: the draw of the main protocol, ``--rule lewm``: the
draw of the LeWM evaluation, both defined in ``evaluation.pairs``. The LeWM
evaluation draws with its planner seed.

``scripts/build_nodes.py --exclude`` reads the same file to keep the pair
episodes out of the graph nodes.

    python scripts/sample_pairs.py --dataset cube_single_expert.h5 \\
        --task cube --rule protocol --goal-offset 50 --count 100 --seed 0 \\
        --out pairs/cube/offset50-seed0.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import stable_worldmodel as swm

from evaluation.pairs import (
    PREDICATE_COLUMN,
    kept_starts,
    lewm_pairs,
    protocol_pairs,
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--task', choices=PREDICATE_COLUMN, required=True)
    parser.add_argument('--rule', choices=('protocol', 'lewm'), required=True)
    parser.add_argument('--goal-offset', type=int, required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), f'{args.out} exists'
    assert args.goal_offset > 0
    if args.rule == 'protocol':
        column = PREDICATE_COLUMN[args.task]
        dataset = swm.data.load_dataset(args.dataset, keys_to_load=[column])
        kept = kept_starts(
            args.task,
            np.asarray(dataset.get_col_data(column), dtype=np.float64),
            np.asarray(dataset.lengths),
            np.asarray(dataset.offsets),
            args.goal_offset,
        )
        pairs = protocol_pairs(kept, args.count, args.seed)
        print(f'{len(kept)} episodes keep a pair')
    else:
        # Every column, so the episode column is found as in the LeWM
        # evaluation's ``episode_col``.
        dataset = swm.data.load_dataset(args.dataset)
        names = set(dataset.column_names)
        column = 'episode_idx' if 'episode_idx' in names else 'ep_idx'
        pairs = lewm_pairs(
            dataset.get_col_data(column),
            dataset.get_col_data('step_idx'),
            args.goal_offset,
            args.count,
            args.seed,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(
            {
                'goal_offset': args.goal_offset,
                'pairs': [
                    {'pair_id': i, **pair} for i, pair in enumerate(pairs)
                ],
            },
            stream,
            indent=2,
        )
    print(f'{len(pairs)} pairs at goal offset {args.goal_offset}: {args.out}')


if __name__ == '__main__':
    main()
