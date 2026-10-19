"""TwoRoom cost maps and two-dimensional embeddings of the grid latents.

A TwoRoom observation is fixed by the agent position. The grid renders the
agent at every position of a 46 x 46 grid with a 4 px stride where the
collision rule can place the agent center (``reachable``), 1,882
positions, and encodes each rendering.

``grid``: render and encode the grid with one checkpoint (CUDA).

``maps``: for every pair of one pair set, the L2 cost (squared Euclidean
distance) and NGC from every grid latent to the latent of the pair's goal
frame, the start and goal positions, and the executed rollouts of one L2
and one NGC result of ``scripts/evaluate.py`` on that pair set (CUDA). A
rollout is replayed from the start position with the recorded executed
actions; positions are kept at the start and after every action block, and
the replay must reach the goal exactly when the evaluation recorded a
success.

``embed``: PCA, metric MDS, Isomap, t-SNE and UMAP of the grid latents,
each fit once, and the Spearman correlation between embedding distance and
physical (x, y) distance over all pairs of grid positions (CPU, dependency
group ``paper``).

    python -m analysis.tworoom_maps grid --checkpoint lewm-tworoom \\
        --out grid.npz
    python -m analysis.tworoom_maps maps --grid grid.npz \\
        --checkpoint lewm-tworoom --dataset tworoom.h5 \\
        --pairs pairs/tworoom/offset25-seed0.json --nodes nodes.npz \\
        --neighbors 15 --anchors 4 --l2 l2.json --ngc ngc.json \\
        --out maps.npz
    python -m analysis.tworoom_maps embed --grid grid.npz --out embed.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

GRID_STRIDE = 4.0
# The collision rule lets the agent center cross the wall wherever the
# coordinate along the wall lies within a door widened by this margin.
DOOR_MARGIN = 1.75
ACTION_BLOCK = 5
ENCODER_BATCH = 32
GOAL_CHUNK = 16
# Embedding settings.
ISOMAP_NEIGHBORS = 8
MDS_MAX_ITER = 20000
MDS_EPS = 1e-9
TSNE_SETTINGS = {
    'perplexity': 50,
    'initialization': 'pca',
    'random_state': 42,
    'early_exaggeration_iter': 250,
    'n_iter': 1250,
    'neighbors': 'exact',
    'negative_gradient_method': 'bh',
    'n_jobs': 1,
}
UMAP_SETTINGS = {'n_neighbors': 15, 'min_dist': 0.1, 'random_state': 0}


def scene(env):
    """Geometry of the TwoRoom scene of a reset environment."""
    assert env.wall_axis == 1 and env.num_doors == 1
    return {
        'image_size': float(env.IMG_SIZE),
        'border_size': float(env.BORDER_SIZE),
        'wall_center': float(env.WALL_CENTER),
        'wall_thickness': float(env.wall_thickness),
        'agent_radius': float(
            env.variation_space['agent']['radius'].value.item()
        ),
        'door_center': float(env.door_positions[0]),
        'door_half': float(env.door_sizes[0]),
    }


def center_bounds(geometry):
    """Lowest and highest coordinate the agent center can take."""
    low = geometry['border_size'] + geometry['agent_radius']
    return low, geometry['image_size'] - low


def reachable(geometry, px, py):
    """Where the collision rule can place the agent center.

    The rule clamps the coordinate across the wall whenever a step would
    end with the center inside the wall inflated by the agent radius,
    unless the coordinate along the wall lies within the door widened by
    ``DOOR_MARGIN``.
    """
    radius = geometry['agent_radius']
    half = geometry['wall_thickness'] // 2
    low, high = center_bounds(geometry)
    band = (px > geometry['wall_center'] - half - radius) & (
        px < geometry['wall_center'] + half + radius
    )
    door = np.abs(py - geometry['door_center']) <= (
        geometry['door_half'] + DOOR_MARGIN
    )
    inside = (px >= low) & (px <= high) & (py >= low) & (py <= high)
    return (~band | door) & inside


def position_colors(xy, low, high):
    """RGB of each position: red grows with x, green with y."""
    u = (xy[:, 0] - low) / (high - low)
    v = (xy[:, 1] - low) / (high - low)
    return np.stack([0.15 + 0.8 * u, 0.2 + 0.7 * v, 0.85 - 0.6 * u], axis=1)


def make_env():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  registers swm/TwoRoom-v1

    env = gym.make('swm/TwoRoom-v1').unwrapped
    env.reset(seed=0)
    assert not env.render_target_flag
    assert not bool(env.variation_space['rendering']['render_target'].value)
    return env


def grid(args):
    import torch

    from ngc.nodes import image_transform, load_model

    env = make_env()
    geometry = scene(env)
    low, high = center_bounds(geometry)
    xs = np.arange(low, high + 1e-9, GRID_STRIDE)
    gx, gy = np.meshgrid(xs, xs)
    feasible = reachable(geometry, gx, gy)
    xy = np.stack([gx[feasible], gy[feasible]], axis=1)
    model = load_model(args.checkpoint)
    transform = image_transform(env.IMG_SIZE)
    latents = []
    with torch.inference_mode():
        for start in range(0, len(xy), ENCODER_BATCH):
            frames = torch.stack(
                [
                    env._render_frame(torch.tensor(p, dtype=torch.float32))
                    for p in xy[start : start + ENCODER_BATCH]
                ]
            )
            info = {'pixels': transform(frames)[:, None].cuda()}
            latents.append(model.encode(info)['emb'][:, 0].float().cpu())
    latent = torch.cat(latents).numpy()
    assert np.isfinite(latent).all()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        x=xs,
        y=xs,
        feasible=feasible,
        xy=xy,
        latent=latent,
        rgb=position_colors(xy, low, high),
        geometry=json.dumps(geometry),
    )
    print(f'{len(xy)} grid positions: {args.out}')


def replay(env, start, goal, actions):
    """Positions of a rollout and whether it reached the goal.

    The positions are the start and the position after every action block;
    the last one is the final position.
    """
    env._set_state(start)
    env._set_goal_state(goal)
    positions, reached = [np.array(start, dtype=np.float64)], False
    for step, action in enumerate(actions, start=1):
        assert not reached, 'actions after the goal was reached'
        _, _, reached, _, _ = env.step(np.asarray(action, dtype=np.float32))
        if step % ACTION_BLOCK == 0 or step == len(actions):
            positions.append(env.agent_position.numpy().astype(np.float64))
    return np.stack(positions), reached


def rollouts(env, result, pair_ids, starts, goals):
    """Replayed positions of every pair of one evaluation result."""
    pairs = {p['pair_id']: p for p in result['pairs']}
    positions, successes = [], []
    for pair_id, start, goal in zip(pair_ids, starts, goals, strict=True):
        pair = pairs[pair_id]
        path, reached = replay(env, start, goal, pair['executed_actions'])
        assert reached == pair['success'], f'pair {pair_id} replays wrong'
        positions.append(path)
        successes.append(pair['success'])
    lengths = np.array([len(p) for p in positions])
    return np.concatenate(positions), lengths, np.array(successes)


def maps(args):
    import stable_worldmodel as swm
    import torch

    from analysis.physical import episode_rows, read_states
    from ngc import NGCCost, build_graph
    from ngc.nodes import encode, load_model

    pair_set = json.loads(args.pairs.read_text())
    pairs = pair_set['pairs']
    pair_ids = np.array([p['pair_id'] for p in pairs])
    episodes = [p['episode_id'] for p in pairs]
    steps = np.array([p['start_step'] for p in pairs])
    start_rows = episode_rows(args.dataset, episodes, steps)
    goal_rows = episode_rows(
        args.dataset, episodes, steps + pair_set['goal_offset']
    )
    starts = read_states(args.dataset, 'tworoom', start_rows)['proprio']
    goals = read_states(args.dataset, 'tworoom', goal_rows)['proprio']
    results = {
        name: json.loads(path.read_text())
        for name, path in (('l2', args.l2), ('ngc', args.ngc))
    }
    assert results['l2']['cost'] == 'l2' and results['ngc']['cost'] == 'ngc'
    assert results['l2']['planner_seed'] == results['ngc']['planner_seed']
    for result in results.values():
        assert result['pair_set'] == str(args.pairs.resolve())

    model = load_model(args.checkpoint)
    clips = swm.data.load_dataset(args.dataset, keys_to_load=['pixels'])
    goal_latents = encode(model, clips, goal_rows, 224, ENCODER_BATCH)
    grid_data = np.load(args.grid)
    latent = grid_data['latent']
    l2 = ((latent[None] - goal_latents[:, None]) ** 2).sum(axis=-1)
    nodes, graph_distance, bridges = build_graph(
        np.load(args.nodes)['latents'], args.neighbors, 'mutual'
    )
    ngc = NGCCost(nodes, graph_distance, args.neighbors, args.anchors).cuda()
    queries = torch.as_tensor(latent, device='cuda')
    targets = torch.as_tensor(goal_latents, device='cuda')
    ngc_cost = np.empty_like(l2)
    with torch.inference_mode():
        for start in range(0, len(targets), GOAL_CHUNK):
            chunk = targets[start : start + GOAL_CHUNK]
            info = {
                'predicted_emb': queries[None, :, None].expand(
                    len(chunk), -1, -1, -1
                ),
                'goal_emb': chunk[:, None],
            }
            ngc_cost[start : start + len(chunk)] = ngc(info).cpu().numpy()

    env = make_env()
    assert json.loads(str(grid_data['geometry'])) == scene(env)
    record = {}
    for name, result in results.items():
        positions, lengths, successes = rollouts(
            env, result, pair_ids, starts, goals
        )
        record[f'{name}_positions'] = positions
        record[f'{name}_lengths'] = lengths
        record[f'{name}_success'] = successes
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        pair_id=pair_ids,
        start=starts,
        goal=goals,
        l2_cost=l2,
        ngc_cost=ngc_cost,
        bridges=bridges,
        planner_seed=results['l2']['planner_seed'],
        **record,
    )
    print(
        f'{len(pairs)} pairs, L2 {record["l2_success"].sum()} and NGC '
        f'{record["ngc_success"].sum()} successes: {args.out}'
    )


def classical_mds(distance):
    """Classical MDS [N, 2] of a distance matrix."""
    n = len(distance)
    center = np.eye(n) - 1.0 / n
    gram = -0.5 * center @ (distance**2) @ center
    values, vectors = np.linalg.eigh(gram)
    top = np.argsort(values)[::-1][:2]
    return vectors[:, top] * np.sqrt(values[top])


def embed(args):
    import umap
    from openTSNE import TSNE
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS, Isomap

    grid_data = np.load(args.grid)
    latent = grid_data['latent'].astype(np.float64)
    physical = pdist(grid_data['xy'])
    distance = squareform(pdist(latent))
    mds = MDS(
        n_components=2,
        metric=True,
        n_init=1,
        max_iter=MDS_MAX_ITER,
        eps=MDS_EPS,
        dissimilarity='precomputed',
        normalized_stress=False,
    )
    embeddings = {
        'pca': PCA(n_components=2).fit_transform(latent),
        'mds': mds.fit_transform(distance, init=classical_mds(distance)),
        'isomap': Isomap(
            n_neighbors=ISOMAP_NEIGHBORS, n_components=2
        ).fit_transform(latent),
        'tsne': np.asarray(TSNE(n_components=2, **TSNE_SETTINGS).fit(latent)),
        'umap': umap.UMAP(n_components=2, **UMAP_SETTINGS)
        .fit_transform(latent)
        .astype(np.float64),
    }
    assert mds.n_iter_ < MDS_MAX_ITER
    stats = {
        name: float(spearmanr(pdist(coords), physical).correlation)
        for name, coords in embeddings.items()
    }
    residual = pdist(embeddings['mds']) - squareform(distance)
    stats['mds_kruskal_stress1'] = float(
        np.sqrt((residual**2).sum() / (squareform(distance) ** 2).sum())
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, xy=grid_data['xy'], **embeddings)
    args.out.with_suffix('.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('grid')
    command.add_argument('--checkpoint', required=True)
    command.add_argument('--out', type=Path, required=True)
    command = commands.add_parser('maps')
    command.add_argument('--grid', type=Path, required=True)
    command.add_argument('--checkpoint', required=True)
    command.add_argument('--dataset', required=True)
    command.add_argument('--pairs', type=Path, required=True)
    command.add_argument('--nodes', type=Path, required=True)
    command.add_argument('--neighbors', type=int, required=True)
    command.add_argument('--anchors', type=int, required=True)
    command.add_argument('--l2', type=Path, required=True)
    command.add_argument('--ngc', type=Path, required=True)
    command.add_argument('--out', type=Path, required=True)
    command = commands.add_parser('embed')
    command.add_argument('--grid', type=Path, required=True)
    command.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), f'{args.out} exists'
    {'grid': grid, 'maps': maps, 'embed': embed}[args.command](args)


if __name__ == '__main__':
    main()
