"""Latent costs against the physical distance on one planning graph.

The costs are measured as planning applies them. The frames are FRAMES
dataset frames drawn uniformly among the frames of the episodes that hold
no graph node (``scripts/build_nodes.py``), so NGC (``ngc.cost.NGCCost``)
attaches both ends of every pair to the graph online, as it does in
planning. Every ordered pair of distinct frames is scored, the first frame
as the query and the second as the goal. States are read from the dataset
rows, and ``analysis.physical`` gives the distance from every query to
every goal. Against that distance the study reports:

- cost against distance: the Spearman correlation of the L2 cost and of
  NGC with the physical distance over all pairs, and the median and
  10th--90th percentiles of each cost in BINS bins of equal width from zero
  to the largest distance.

On the graph nodes themselves, it reports:

- edge lengths: the physical distance spanned by the edges of the graph
  rule before bridging;
- neighborhood faithfulness: the fraction of each node's latent nearest
  neighbors among its physically nearest nodes, and how much farther, in
  physical distance, the latent neighbors lie than the physical ones;
- distance concentration: the moments of the latent marginal along random
  directions, its participation ratio, the distance of two independent
  draws of a Gaussian with the latent covariance, and the Euclidean
  distance by decile of the physical distance.

Correlations over all pairs share their points and are descriptive.

    python -m analysis.geometry --task cube --nodes nodes/cube-seed0.npz \\
        --dataset cube_single_expert.h5 --checkpoint lewm-cube \\
        --image-size 224 --batch-size 32 --neighbors 15 --anchors 4 \\
        --rule mutual --out results/geometry/cube-seed0.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from analysis import physical
from ngc import NGCCost, build_graph
from ngc.graph import RULES, neighbor_links
from ngc.nodes import encode, load_model

FRAMES = 10000
FRAME_SEED = 0
GOAL_CHUNK = 16
BINS = 30
FAITHFUL_NEIGHBORS = 15
PROJECTIONS = 1024
PROJECTION_SEED = 0
DECILES = 10


def episode_of(clips, rows):
    """Episode index of every dataset row."""
    return np.searchsorted(clips.offsets, rows, side='right') - 1


def sampled_rows(clips, node_rows):
    """FRAMES rows drawn uniformly, without replacement, from the episodes
    that hold no node, ascending."""
    eligible = np.setdiff1d(
        np.arange(len(clips.lengths)), episode_of(clips, node_rows)
    )
    pool = np.concatenate(
        [clips.offsets[e] + np.arange(clips.lengths[e]) for e in eligible]
    )
    assert len(pool) >= FRAMES, (len(pool), FRAMES)
    rng = np.random.default_rng(FRAME_SEED)
    return np.sort(rng.choice(pool, FRAMES, replace=False))


def ngc_costs(nodes, queries, goals, neighbors, anchors, rule):
    """NGC [Q, G] from every query latent to every goal latent, on CUDA."""
    nodes, graph_distance, bridges = build_graph(nodes, neighbors, rule)
    model = NGCCost(nodes, graph_distance, neighbors, anchors).cuda()
    queries = torch.as_tensor(queries, dtype=torch.float32, device='cuda')
    goals = torch.as_tensor(goals, dtype=torch.float32, device='cuda')
    cost = np.empty((len(queries), len(goals)))
    for start in range(0, len(goals), GOAL_CHUNK):
        chunk = goals[start : start + GOAL_CHUNK]
        info = {
            'predicted_emb': queries[None, :, None].expand(
                len(chunk), -1, -1, -1
            ),
            'goal_emb': chunk[:, None],
        }
        cost[:, start : start + len(chunk)] = model(info).cpu().numpy().T
    return cost, bridges


def spearman(a, b):
    return float(spearmanr(a, b).correlation)


def cost_against_distance(cost, truth):
    """Cost [F, F] against the physical distance [F, F] over every ordered
    pair of distinct frames."""
    off = ~np.eye(len(cost), dtype=bool)
    cost, truth = cost[off], truth[off]
    edges = np.linspace(0.0, truth.max(), BINS + 1)
    binned = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        upper = truth <= hi if i == BINS - 1 else truth < hi
        values = cost[(truth >= lo) & upper]
        assert len(values) > 0, (lo, hi)
        q10, q50, q90 = np.quantile(values, [0.1, 0.5, 0.9])
        binned.append(
            {
                'lo': float(lo),
                'hi': float(hi),
                'n': int(len(values)),
                'q10': float(q10),
                'median': float(q50),
                'q90': float(q90),
            }
        )
    return {'spearman': spearman(cost, truth), 'binned': binned}


def edge_lengths(euclidean, truth, neighbors, rule):
    """Physical distance spanned by the edges of ``rule`` between nodes."""
    latent = euclidean.copy()
    np.fill_diagonal(latent, np.inf)
    linked = np.triu(neighbor_links(latent, neighbors, rule), k=1)
    spans = truth[linked]
    q50, q99 = np.quantile(spans, [0.5, 0.99])
    return {
        'edges': int(linked.sum()),
        'median': float(q50),
        'q99': float(q99),
        'max': float(spans.max()),
    }


def neighborhood(euclidean, truth):
    """Latent against physical nearest neighbors of every node."""
    latent, physical_distance = euclidean.copy(), truth.copy()
    np.fill_diagonal(latent, np.inf)
    np.fill_diagonal(physical_distance, np.inf)
    k = FAITHFUL_NEIGHBORS
    latent_nn = np.argsort(latent, axis=1, kind='stable')[:, :k]
    physical_nn = np.argsort(physical_distance, axis=1, kind='stable')[:, :k]
    overlap = np.array(
        [len(set(a) & set(b)) / k for a, b in zip(latent_nn, physical_nn)]
    )
    to_latent = np.take_along_axis(physical_distance, latent_nn, axis=1)
    to_physical = np.take_along_axis(physical_distance, physical_nn, axis=1)
    ratio = to_latent.mean(axis=1) / to_physical.mean(axis=1)
    finite = np.isfinite(ratio)
    return {
        'neighbors': k,
        'overlap_median': float(np.median(overlap)),
        'distance_ratio_median': float(np.median(ratio[finite])),
        'nodes_with_zero_physical_neighbor_distance': int((~finite).sum()),
    }


def concentration(latents, euclidean, truth):
    d = latents.shape[1]
    centered = latents - latents.mean(axis=0)
    covariance = centered.T @ centered / (len(latents) - 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    trace = float(eigenvalues.sum())
    participation = trace**2 / float((eigenvalues**2).sum())
    directions = np.random.default_rng(PROJECTION_SEED).standard_normal(
        (d, PROJECTIONS)
    )
    directions /= np.linalg.norm(directions, axis=0)
    projected = latents @ directions
    std = projected.std(axis=0)
    standardized = (projected - projected.mean(axis=0)) / std
    kurtosis = (standardized**4).mean(axis=0) - 3.0
    off = ~np.eye(len(latents), dtype=bool)
    e, t = euclidean[off], truth[off]
    edges = np.quantile(t, np.linspace(0, 1, DECILES + 1))
    deciles = []
    for i in range(DECILES):
        upper = t <= edges[i + 1] if i == DECILES - 1 else t < edges[i + 1]
        mask = (t >= edges[i]) & upper
        q10, q50, q90 = np.quantile(e[mask], [0.1, 0.5, 0.9])
        deciles.append(
            {
                'truth_median': float(np.median(t[mask])),
                'euclid_q10': float(q10),
                'euclid_median': float(q50),
                'euclid_q90': float(q90),
            }
        )
    far = t > np.median(t)
    beyond = [b['euclid_median'] for b in deciles[1:]]
    return {
        'dimension': d,
        'participation_ratio': participation,
        'random_projection_median_std': float(np.median(std)),
        'random_projection_median_excess_kurtosis': float(np.median(kurtosis)),
        'independent_pair_distance': float(np.sqrt(2 * trace)),
        'far_half': {
            'euclid_median': float(np.median(e[far])),
            'euclid_cv': float(e[far].std() / e[far].mean()),
            'truth_cv': float(t[far].std() / t[far].mean()),
        },
        'beyond_first_decile': {
            'truth_median_growth': deciles[-1]['truth_median']
            / deciles[1]['truth_median'],
            'euclid_median_spread': max(beyond) / min(beyond) - 1.0,
        },
        'deciles': deciles,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--task', choices=physical.TASKS, required=True)
    parser.add_argument('--nodes', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--neighbors', type=int, required=True)
    parser.add_argument('--anchors', type=int, required=True)
    parser.add_argument('--rule', choices=RULES, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), f'{args.out} exists'
    with np.load(args.nodes) as data:
        node_rows = data['rows']
        nodes = data['latents'].astype(np.float64)

    clips = swm.data.load_dataset(args.dataset, keys_to_load=['pixels'])
    rows = sampled_rows(clips, node_rows)
    model = load_model(args.checkpoint)
    frames = encode(model, clips, rows, args.image_size, args.batch_size)
    states, node_states = (
        physical.read_states(args.dataset, args.task, r)
        for r in (rows, node_rows)
    )
    truth = physical.distance(args.task, states, states)
    l2 = cdist(frames, frames, 'sqeuclidean')
    ngc, bridges = ngc_costs(
        nodes, frames, frames, args.neighbors, args.anchors, args.rule
    )
    node_truth = physical.distance(args.task, node_states, node_states)
    node_euclidean = cdist(nodes, nodes)
    result = {
        'task': args.task,
        'nodes': str(args.nodes),
        'node_count': len(nodes),
        'rows': rows.tolist(),
        'neighbors': args.neighbors,
        'anchors': args.anchors,
        'rule': args.rule,
        'bridges': bridges,
        'unit': physical.UNIT[args.task],
        'stats': {
            'l2': cost_against_distance(l2, truth),
            'ngc': cost_against_distance(ngc, truth),
        },
        'edge_lengths': edge_lengths(
            node_euclidean, node_truth, args.neighbors, args.rule
        ),
        'neighborhood': neighborhood(node_euclidean, node_truth),
        'concentration': concentration(nodes, node_euclidean, node_truth),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
    for key in ('l2', 'ngc'):
        print(
            f'{args.task} {key}: Spearman over all pairs '
            f'{result["stats"][key]["spearman"]:.3f}'
        )


if __name__ == '__main__':
    main()
