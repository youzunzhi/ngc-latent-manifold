"""Write nested subsets of a graph-node file.

One permutation of the nodes, drawn from ``--seed``, orders them; the subset
of size ``n`` is the first ``n`` nodes of that order, so every subset is
contained in every larger one. Each output keeps the rows and latents of its
nodes in ascending row order.

    python scripts/subset_nodes.py --nodes nodes/cube.npz \\
        --counts 625 1250 2500 --seed 0 --out-dir nodes/cube-subsets
"""

import argparse
from pathlib import Path

import numpy as np


def nested_subsets(count, sizes, seed):
    """Ascending node indices of every size, all from one permutation."""
    assert all(1 <= size <= count for size in sizes)
    order = np.random.default_rng(seed).permutation(count)
    return {size: np.sort(order[:size]) for size in sizes}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--nodes', type=Path, required=True)
    parser.add_argument('--counts', nargs='+', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.nodes) as data:
        rows, latents = data['rows'], data['latents']
    assert len(rows) == len(latents) and np.all(np.diff(rows) > 0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for size, ids in nested_subsets(len(rows), args.counts, args.seed).items():
        out = args.out_dir / f'{args.nodes.stem}-{size}.npz'
        assert not out.exists(), f'{out} exists'
        np.savez(out, rows=rows[ids], latents=latents[ids])
        print(f'{size} of {len(rows)} nodes: {out}')


if __name__ == '__main__':
    main()
