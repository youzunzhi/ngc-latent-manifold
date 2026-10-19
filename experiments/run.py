"""Print a paper stage, or execute it sequentially with --execute."""

import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

import yaml

from experiments.plan import Plan

STAGES = (
    'train-smoke',
    'train',
    'replicate-train',
    'replicate-nodes',
    'replicate-cells',
    'pairs',
    'prepare',
    'nodes',
    'subsets',
    'cells',
    'analysis',
    'main-nodes',
    'main-cells',
)


def materialize_pair(plan, command):
    if '--pairs' not in command.argv:
        return
    path = Path(command.argv[command.argv.index('--pairs') + 1])
    if path.exists() or path.parent.parent.name != 'pairs-single':
        return
    match = re.fullmatch(r'offset(\d+)-seed(\d+)-pair(\d+)', path.stem)
    if match is None:
        raise ValueError(f'Invalid single-pair path: {path}')
    offset, seed, pair_id = map(int, match.groups())
    source = json.loads(plan.pairs(path.parent.name, offset, seed).read_text())
    pairs = [p for p in source['pairs'] if p['pair_id'] == pair_id]
    assert len(pairs) == 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump({'goal_offset': offset, 'pairs': pairs}, stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('outputs/paper'))
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(args.config.resolve().read_text())
    root = args.root.resolve()
    os.chdir(repo)
    plan = Plan(config, repo, root)
    if args.stage == 'main-cells':
        commands = plan.ladder_cells(
            'ladder', 'project', lambda t, s: plan.arms(t, s, 'project')
        )
    elif args.stage == 'main-nodes':
        commands = [
            c
            for c in plan.nodes_stage()
            if c.out.parent == root / 'nodes/project'
        ]
    else:
        commands = plan.stage(args.stage)
    for command in commands:
        argv = list(map(str, command.argv))
        print(shlex.join(argv), flush=True)
        if args.execute:
            # Existing output is never overwritten or assumed to be complete.
            if command.out.exists():
                raise FileExistsError(command.out)
            materialize_pair(plan, command)
            env = dict(
                os.environ,
                MUJOCO_GL='egl',
                PYOPENGL_PLATFORM='egl',
                MPLBACKEND='Agg',
            )
            # Script entry points must resolve sibling repository modules.
            env['PYTHONPATH'] = (
                str(repo) + os.pathsep + env.get('PYTHONPATH', '')
            )
            subprocess.run(argv, cwd=repo, env=env, check=True)


if __name__ == '__main__':
    main()
