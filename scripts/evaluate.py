"""Plan with a LeWM checkpoint on fixed start-goal pairs and report success.

Each pair starts an environment from a recorded dataset state at rest and
takes as goal the frame ``goal_offset`` steps later in the same episode.
The planner plans five action blocks of five steps, executes all of them,
and replans with up to three real frames of the executed rollout.

CEM draws ``--num-samples`` candidates per iteration and refits to the
``--topk`` best under ``--cost``: the L2 cost of LeWM, NGC, or NGC_Euc
(``ngc-euc``, ``ngc.ablations``). With ``--dynamics oracle`` the
candidates are executed in the simulator instead of predicted by the world
model (``evaluation.oracle``).

A pair succeeds when the task's success rule holds at any step within
``--eval-budget`` steps.

``--protocol comparability`` is the LeWM evaluation instead: the start is
the recorded state with its velocity, and CEM plans from the current frame
alone with the policy of stable-worldmodel (pairs from
``scripts/sample_pairs.py --rule lewm``, Reacher with its released success
rule in ``configs/reacher-native.yaml``).

The pair file holds ``goal_offset`` and a list of ``{episode_id,
start_step}``; ``scripts/sample_pairs.py`` writes one.

    python scripts/evaluate.py --config configs/cube.yaml \\
        --pairs pairs/cube.json --dataset cube_single_expert.h5 \\
        --checkpoint lewm-cube --dynamics learned \\
        --cost ngc --nodes nodes/cube.npz --neighbors 15 --anchors 4 \\
        --rule mutual --num-samples 300 --topk 30 --eval-budget 100 \\
        --planner-seed 42 \\
        --out results/cube.json
"""

import os

os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
import yaml
from sklearn import preprocessing
from stable_worldmodel.planning.solver import CEMSolver

import evaluation.pusht_replay  # noqa: F401  registers ngc/PushTReplay-v1
import evaluation.reacher  # noqa: F401  registers ngc/ReacherJointMatch-v0
from evaluation import HistoryCostEvaluator, HistoryPolicy, RestStartDataset
from evaluation.history import IndexedHistoryPolicy
from evaluation.oracle import SimulatorOracle
from ngc import NGCCost, build_graph
from ngc.ablations import euclidean_table
from ngc.graph import RULES
from ngc.nodes import checkpoint_location, image_transform

PLAN = swm.PlanConfig(
    horizon=5,
    receding_horizon=5,
    history_len=3,
    action_block=5,
    warm_start=True,
)
# The planning loop of the LeWM evaluation: the current frame alone.
COMPARABILITY_PLAN = swm.PlanConfig(
    horizon=5, receding_horizon=5, action_block=5
)
CEM = {'batch_size': 1, 'var_scale': 1.0, 'n_steps': 30}
COSTS = ('l2', 'ngc', 'ngc-euc')
ORACLE_ENCODER_BATCH = 256


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
    """The terminal cost and a record of its graph (None without one)."""
    if args.cost == 'l2':
        assert args.nodes is None and args.neighbors is None
        assert args.anchors is None and args.rule is None
        return swm.planning.GoalMSE(), None
    latents = np.load(args.nodes)['latents']
    if args.cost == 'ngc-euc':
        assert args.rule is None
        objective = NGCCost(
            latents, euclidean_table(latents), args.neighbors, args.anchors
        )
        return objective.cuda(), {'nodes': len(latents)}
    nodes, graph_distance, bridges = build_graph(
        latents, args.neighbors, args.rule
    )
    objective = NGCCost(nodes, graph_distance, args.neighbors, args.anchors)
    return objective.cuda(), {'nodes': len(nodes), 'bridges': bridges}


def cem_policy(args, config, world, model, process, transform):
    """The CEM policy, its graph record and the oracle's scratch world."""
    objective, graph = make_objective(args)
    if args.protocol == 'comparability':
        cost = swm.planning.ShootingCostEvaluator(model, objective)
        solver = CEMSolver(
            cost,
            **CEM,
            num_samples=args.num_samples,
            topk=args.topk,
            device='cuda',
            seed=args.planner_seed,
        )
        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=COMPARABILITY_PLAN,
            process=process,
            transform=transform,
        )
        return policy, graph, None
    dynamics, policy_class, scratch = model, HistoryPolicy, None
    if args.dynamics == 'oracle':
        assert CEM['batch_size'] == 1
        scratch = swm.World(
            **config['world'],
            num_envs=CEM['batch_size'],
            max_episode_steps=2 * args.eval_budget,
            image_shape=(config['image_size'], config['image_size']),
            add_pixels=True,
        )
        dynamics = SimulatorOracle(
            model,
            world.envs.envs,
            scratch.envs.envs,
            action_block=PLAN.action_block,
            action_process=process['action'],
            pixel_transform=transform['pixels'],
            encoder_batch_size=ORACLE_ENCODER_BATCH,
        )
        policy_class = IndexedHistoryPolicy
    cost = HistoryCostEvaluator(dynamics, objective)
    solver = CEMSolver(
        cost,
        **CEM,
        num_samples=args.num_samples,
        topk=args.topk,
        device='cuda',
        seed=args.planner_seed,
    )
    policy = policy_class(
        solver=solver, config=PLAN, process=process, transform=transform
    )
    return policy, graph, scratch


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument(
        '--protocol', choices=('main', 'comparability'), required=True
    )
    parser.add_argument(
        '--dynamics', choices=('learned', 'oracle'), required=True
    )
    parser.add_argument('--cost', choices=COSTS, required=True)
    parser.add_argument('--nodes', type=Path)
    parser.add_argument('--neighbors', type=int)
    parser.add_argument('--anchors', type=int)
    parser.add_argument('--rule', choices=RULES)
    parser.add_argument('--num-samples', type=int, required=True)
    parser.add_argument('--topk', type=int, required=True)
    parser.add_argument('--eval-budget', type=int, required=True)
    parser.add_argument('--planner-seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), f'{args.out} exists'
    assert 1 <= args.topk <= args.num_samples
    if args.protocol == 'comparability':
        assert args.dynamics == 'learned'
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

    model = swm.wm.utils.load_pretrained(
        checkpoint_location(args.checkpoint)
    ).cuda().eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    policy, graph, scratch = cem_policy(
        args, config, world, model, process, transform
    )
    world.set_policy(policy)
    if args.protocol == 'main':
        dataset = RestStartDataset(dataset, config['task'])
    began = time.perf_counter()
    metrics = world.evaluate(
        dataset=dataset,
        episodes_idx=episodes,
        start_steps=start_steps,
        goal_offset=goal_offset,
        eval_budget=args.eval_budget,
        callables=config['callables'],
    )

    successes = [bool(s) for s in metrics['episode_successes']]
    pairs = [
        {**pair, 'success': success}
        for pair, success in zip(pair_set['pairs'], successes, strict=True)
    ]
    timing = {'evaluation_seconds': time.perf_counter() - began}
    if args.protocol == 'main':
        for pair, executed in zip(pairs, policy.executed_actions, strict=True):
            pair['executed_actions'] = np.asarray(executed).tolist()
        timing['solve_seconds'] = policy.solve_seconds
    result = {
        'task': config['task'],
        'protocol': args.protocol,
        'pair_set': str(args.pairs.resolve()),
        'cost': args.cost,
        'dynamics': args.dynamics,
        'neighbors': args.neighbors,
        'anchors': args.anchors,
        'rule': args.rule,
        'num_samples': args.num_samples,
        'topk': args.topk,
        'planner_seed': args.planner_seed,
        'goal_offset': goal_offset,
        'eval_budget': args.eval_budget,
        'graph': graph,
        'pairs': pairs,
        'success_rate': sum(successes) / len(successes),
        **timing,
        'device': torch.cuda.get_device_name(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(f'{sum(successes)}/{len(successes)} pairs succeeded')
    world.close()
    if scratch is not None:
        scratch.close()


if __name__ == '__main__':
    main()
