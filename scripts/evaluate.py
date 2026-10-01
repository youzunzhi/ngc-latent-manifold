"""Plan with a LeWM checkpoint on fixed start-goal pairs and report success.

Each pair starts an environment from a recorded dataset state at rest and
takes as goal the frame ``goal_offset`` steps later in the same episode.
CEM plans five action blocks of five steps, executes all of them, and
replans with up to three real frames of the executed rollout. The terminal
cost is NGC or the L2 cost of LeWM. A pair succeeds when the task's success
rule holds at any step within ``--eval-budget`` steps.

The pair file holds ``goal_offset`` and a list of ``{episode_id,
start_step}``; ``scripts/sample_pairs.py`` writes one.

    python scripts/evaluate.py --config configs/cube.yaml \\
        --pairs pairs/cube.json --dataset cube_single_expert.h5 \\
        --checkpoint lewm-cube --cost ngc --nodes nodes/cube.npz \\
        --neighbors 15 --anchors 4 --eval-budget 100 --planner-seed 42 \\
        --out results/cube.json
"""

import os

os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import argparse
import json
import random
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
import yaml
from sklearn import preprocessing
from stable_worldmodel.planning.solver import CEMSolver

import evaluation.reacher  # noqa: F401  registers ngc/ReacherJointMatch-v0
from evaluation import HistoryCostEvaluator, HistoryPolicy, RestStartDataset
from ngc import NGCCost, build_graph
from ngc.nodes import image_transform

PLAN = swm.PlanConfig(
    horizon=5,
    receding_horizon=5,
    history_len=3,
    action_block=5,
    warm_start=True,
)
CEM = {
    'batch_size': 1,
    'num_samples': 300,
    'var_scale': 1.0,
    'n_steps': 30,
    'topk': 30,
}


def seed_runtime(seed):
    """Seed every global generator and run on one CPU thread."""
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_scalers(dataset, columns):
    """Standardize each non-image column over the whole dataset.

    A scaler also normalizes the goal counterpart ``goal_<column>`` of
    every column but ``action``.
    """
    process = {}
    for column in columns:
        values = dataset.get_col_data(column)
        scaler = preprocessing.StandardScaler()
        scaler.fit(values[~np.isnan(values).any(axis=1)])
        process[column] = scaler
        if column != 'action':
            process[f'goal_{column}'] = scaler
    return process


def make_objective(args):
    if args.cost == 'l2':
        assert args.nodes is None and args.neighbors is None
        assert args.anchors is None
        return swm.planning.GoalMSE(), None
    latents = np.load(args.nodes)['latents']
    nodes, graph_distance, bridges = build_graph(latents, args.neighbors)
    objective = NGCCost(nodes, graph_distance, args.neighbors, args.anchors)
    return objective.cuda(), {'nodes': len(nodes), 'bridges': bridges}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cost', choices=('ngc', 'l2'), required=True)
    parser.add_argument('--nodes', type=Path)
    parser.add_argument('--neighbors', type=int)
    parser.add_argument('--anchors', type=int)
    parser.add_argument('--eval-budget', type=int, required=True)
    parser.add_argument('--planner-seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), f'{args.out} exists'
    assert torch.cuda.is_available(), 'planning runs on CUDA'
    # Full float32 kernels, so a latent does not depend on its batch.
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    config = yaml.safe_load(args.config.read_text())
    pair_set = json.loads(args.pairs.read_text())
    goal_offset = pair_set['goal_offset']
    episodes = [pair['episode_id'] for pair in pair_set['pairs']]
    start_steps = [pair['start_step'] for pair in pair_set['pairs']]
    assert PLAN.horizon * PLAN.action_block <= args.eval_budget
    seed_runtime(args.planner_seed)

    world = swm.World(
        **config['world'],
        num_envs=len(episodes),
        max_episode_steps=2 * args.eval_budget,
        image_shape=(config['image_size'], config['image_size']),
        add_pixels=True,
    )
    dataset = swm.data.load_dataset(
        args.dataset, keys_to_cache=config['keys_to_cache']
    )
    for episode, start in zip(episodes, start_steps, strict=True):
        assert 0 <= start < start + goal_offset < dataset.lengths[episode], (
            f'episode {episode} has no goal {goal_offset} steps after {start}'
        )
    process = fit_scalers(dataset, config['keys_to_cache'])
    transform = {
        'pixels': image_transform(config['image_size']),
        'goal': image_transform(config['image_size']),
    }

    model = swm.wm.utils.load_pretrained(args.checkpoint).cuda().eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    objective, graph = make_objective(args)
    cost = HistoryCostEvaluator(model, objective)
    solver = CEMSolver(cost, **CEM, device='cuda', seed=args.planner_seed)
    policy = HistoryPolicy(
        solver=solver, config=PLAN, process=process, transform=transform
    )
    world.set_policy(policy)
    metrics = world.evaluate(
        dataset=RestStartDataset(dataset, config['task']),
        episodes_idx=episodes,
        start_steps=start_steps,
        goal_offset=goal_offset,
        eval_budget=args.eval_budget,
        callables=config['callables'],
    )

    successes = [bool(s) for s in metrics['episode_successes']]
    result = {
        'task': config['task'],
        'cost': args.cost,
        'planner_seed': args.planner_seed,
        'goal_offset': goal_offset,
        'eval_budget': args.eval_budget,
        'graph': graph,
        'pairs': [
            {**pair, 'success': success}
            for pair, success in zip(pair_set['pairs'], successes, strict=True)
        ],
        'success_rate': sum(successes) / len(successes),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(f'{args.cost}: {sum(successes)}/{len(successes)} pairs succeeded')
    world.close()


if __name__ == '__main__':
    main()
