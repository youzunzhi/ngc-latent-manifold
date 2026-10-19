"""Sample and encode the graph nodes of NGC for one dataset and checkpoint.

The training split is redrawn from the fraction and seed of the world
model's training run, over clips of the training frameskip and number of
steps. Episodes of the given pair sets are excluded. The output holds the
dataset rows of the nodes and their float32 latents.

    python scripts/build_nodes.py --dataset cube_single_expert.h5 \\
        --checkpoint lewm-cube --train-fraction 0.9 --train-seed 3072 \\
        --frameskip 5 --num-steps 4 --exclude pairs/cube/*.json \\
        --count 5000 --seed 0 --image-size 224 --batch-size 32 \\
        --out nodes/cube.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
import stable_worldmodel as swm

from ngc.nodes import encode, load_model, sample_rows, training_clips


def excluded_episodes(pair_sets):
    """Every episode a pair of the given pair sets starts from."""
    return {
        pair['episode_id']
        for path in pair_sets
        for pair in json.loads(Path(path).read_text())['pairs']
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--train-fraction', type=float, required=True)
    parser.add_argument('--train-seed', type=int, required=True)
    parser.add_argument('--frameskip', type=int, required=True)
    parser.add_argument('--num-steps', type=int, required=True)
    parser.add_argument('--exclude', nargs='*', type=Path, required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), f'{args.out} exists'

    clips = swm.data.load_dataset(
        args.dataset,
        frameskip=args.frameskip,
        num_steps=args.num_steps,
        keys_to_load=['pixels'],
    )
    train = training_clips(clips, args.train_fraction, args.train_seed)
    excluded = excluded_episodes(args.exclude)
    rows = sample_rows(clips, train, excluded, args.count, args.seed)
    model = load_model(args.checkpoint)
    latents = encode(model, clips, rows, args.image_size, args.batch_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, rows=rows, latents=latents)
    print(
        f'{len(rows)} nodes from {len(train)} training clips, '
        f'{len(excluded)} episodes excluded: {args.out}'
    )


if __name__ == '__main__':
    main()
