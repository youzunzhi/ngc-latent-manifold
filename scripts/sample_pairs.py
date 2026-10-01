"""Sample start-goal pairs from a dataset for ``scripts/evaluate.py``.

Each pair comes from a different episode, drawn uniformly among the
episodes longer than ``goal_offset`` steps. Its start step is uniform over
the steps whose goal, ``goal_offset`` steps later, lies in the same
episode. ``scripts/build_nodes.py --exclude`` reads the same file to keep
the pair episodes out of the graph nodes.

    python scripts/sample_pairs.py --dataset cube_single_expert.h5 \\
        --goal-offset 50 --count 100 --seed 0 --out pairs/cube.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import stable_worldmodel as swm


def sample_pairs(lengths, goal_offset, count, seed):
    """``count`` pairs ``{episode_id, start_step}`` in episode order."""
    lengths = np.asarray(lengths)
    eligible = np.flatnonzero(lengths > goal_offset)
    assert 1 <= count <= len(eligible), (
        f'{count} pairs requested from {len(eligible)} eligible episodes'
    )
    rng = np.random.default_rng(seed)
    episodes = np.sort(rng.choice(eligible, count, replace=False))
    return [
        {
            'episode_id': int(episode),
            'start_step': int(rng.integers(lengths[episode] - goal_offset)),
        }
        for episode in episodes
    ]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--goal-offset', type=int, required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), f'{args.out} exists'
    assert args.goal_offset > 0
    dataset = swm.data.load_dataset(args.dataset, keys_to_load=['action'])
    pairs = sample_pairs(
        dataset.lengths, args.goal_offset, args.count, args.seed
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump({'goal_offset': args.goal_offset, 'pairs': pairs}, stream)
    print(f'{len(pairs)} pairs at goal offset {args.goal_offset}: {args.out}')


if __name__ == '__main__':
    main()
