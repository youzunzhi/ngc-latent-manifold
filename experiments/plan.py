"""Every computation behind the paper, as commands grouped in stages.

A stage is a list of ``Command``: the file it writes and its arguments.
Stages run in order, each after the previous one finished:

- ``replicate-train``, ``replicate-nodes``, ``replicate-cells``: LeWM with
  the replicate training seeds on every task, their graph nodes, and L2 cost
  and NGC on the pairs of every offset (on demand, apart from the other
  stages).

- ``train-smoke``: a few batches of the training recipe on one HDF5 task
  and the Lance task, written under ``smoke/`` of the run root; gates the
  training stages.
- ``train``: the project checkpoint of the tasks in ``RETRAINED_TASKS``;
  the other tasks use the project checkpoint of the configuration.
- ``pairs``: the pair sets of the main protocol, written into ``pairs/`` of
  the repository and committed.
- ``prepare``: draw the pairs of the LeWM evaluation.
- ``nodes``: graph nodes of every task, pair seed and checkpoint, of the
  LeWM evaluation's graph, the redrawn nodes of the study pair seed, and the
  latents of the TwoRoom grid.
- ``subsets``: nested node subsets of the graph-size study.
- ``cells``: every evaluation.
- ``analysis``: geometry (also with the union rule on the study pair seed),
  strata, timing and the TwoRoom cost maps of the
  offset-25 pairs of the study pair seed (planner seed 42). The embeddings of the
  TwoRoom grid (``analysis.tworoom_maps embed``) run on a CPU with the
  dependency group ``paper``.

Arms of the paper, by study (pairs: 4 tasks x 3 offsets x 3 pair seeds,
planner seeds 42, 43, 44 unless noted):

- ``ladder``: L2 cost and NGC on the project checkpoint.
- ``ablation``: NGC_Euc.
- ``budget``: L2 cost and NGC with CEM sizes 0.2x to 4x (1x is the
  ladder), at the longest offset.
- ``graph``: NGC with k in 5, 10, 30, 60 and with 1/8, 1/4, 1/2 of the
  nodes, the study pair seed, offsets 25 and longest.
- ``rule``: NGC with the union rule, the study pair seed, offsets 25 and
  longest.
- ``redraw``: NGC on graph nodes redrawn with the node seeds of
  ``REDRAW_NODE_SEEDS`` from the same eligible episodes, the study pair seed,
  offsets 25 and longest.
- ``oracle``: L2 cost and NGC with the oracle predictor, Cube, TwoRoom and
  PushT, the study pair seed, planner seed 42, one pair per evaluation.
- ``official``: L2 cost and NGC on the released checkpoint.
- ``comparability``: the LeWM evaluation, project and released checkpoints.
"""

import json
from dataclasses import dataclass
from pathlib import Path

TASKS = ('cube', 'tworoom', 'pusht', 'reacher')
OFFSETS = {
    'cube': (25, 50, 100),
    'tworoom': (25, 50, 75),
    'pusht': (25, 50, 100),
    'reacher': (25, 50, 100),
}
PAIR_SEEDS = (0, 1, 2)
PAIR_COUNT = 100
# The pair seed of the studies that run on one pair seed.
STUDY_PAIR_SEED = 0
PLANNER_SEEDS = (42, 43, 44)
NODE_COUNT = {'cube': 5000, 'tworoom': 5000, 'pusht': 5000, 'reacher': 5000}
TRAINING_SPLIT = (
    '--train-fraction', '0.9', '--frameskip', '5', '--num-steps', '4',
)  # fmt: skip
PROJECT_TRAINING_SEED = 3072
REPLICATE_TRAINING_SEEDS = (3073, 3074)
# Tasks whose project checkpoint the ``train`` stage trains.
RETRAINED_TASKS = ('tworoom',)
TRAINING_EPOCHS = 10
SMOKE_TASKS = ('tworoom', 'pusht')
SMOKE_BATCHES = 8
NEIGHBORS, ANCHORS = 15, 4
RULE = 'mutual'
# The neighbor rule of the rule study.
RULE_VARIANT = 'union'
# Node seeds of the redraw study; the graph of a pair seed uses the pair seed.
REDRAW_NODE_SEEDS = (100, 101)
CEM = (300, 30)
CEM_MULTIPLIERS = {
    '0.2': (60, 6),
    '0.5': (150, 15),
    '0.8': (240, 24),
    '1.2': (360, 36),
    '2': (600, 60),
    '4': (1200, 120),
}
GRAPH_NEIGHBORS = (5, 10, 30, 60)
GRAPH_FRACTIONS = (0.125, 0.25, 0.5)
NESTING_SEED = 0
ORACLE_TASKS = ('cube', 'tworoom', 'pusht')
COMPARABILITY_OFFSET = 25
COMPARABILITY_PAIRS = 50
COMPARABILITY_NODE_SEED = 0
ENCODE_BATCH = 32


@dataclass
class Command:
    out: Path
    argv: list


def longest(task):
    return max(OFFSETS[task])


class Plan:
    def __init__(self, config, repo, root):
        self.config, self.repo, self.root = config, Path(repo), Path(root)

    def python(self, *argv):
        return [self.config['python'], *map(str, argv)]

    def dataset(self, task):
        return self.config['datasets'][task]

    def checkpoint(self, kind, task):
        if kind == 'project' and task in RETRAINED_TASKS:
            return str(self.trained(task, PROJECT_TRAINING_SEED))
        return self.config['checkpoints'][kind].format(task=task)

    def pairs(self, task, offset, seed):
        return self.repo / 'pairs' / task / f'offset{offset}-seed{seed}.json'

    def lewm_pairs(self, task, planner_seed):
        return self.root / 'pairs-lewm' / task / f'planner{planner_seed}.json'

    def nodes(self, kind, task, seed):
        return self.root / 'nodes' / kind / f'{task}-seed{seed}.npz'

    def redrawn_nodes(self, task, node_seed):
        return (
            self.root / 'nodes' / 'redraw'
            / f'{task}-seed{STUDY_PAIR_SEED}-nodes{node_seed}.npz'
        )  # fmt: skip

    def lewm_nodes(self, kind, task):
        return self.root / 'nodes-lewm' / kind / f'{task}.npz'

    def subset(self, task, count):
        stem = self.nodes('project', task, STUDY_PAIR_SEED).stem
        return self.root / 'subsets' / f'{stem}-{count}.npz'

    def tworoom_grid(self):
        return self.root / 'tworoom' / 'grid.npz'

    def result(self, study, task, name):
        return self.root / 'results' / study / task / f'{name}.json'

    def trained(self, task, seed):
        run = self.config['training_root'] + f'/checkpoints/{task}-seed{seed}'
        return Path(run) / f'weights_epoch_{TRAINING_EPOCHS}.pt'

    def replicate_nodes_path(self, training_seed, task, seed):
        return (
            self.root / 'nodes' / f'training{training_seed}'
            / f'{task}-seed{seed}.npz'
        )  # fmt: skip

    # -- stages

    def train(self):
        return [
            self.train_command(task, PROJECT_TRAINING_SEED)
            for task in RETRAINED_TASKS
        ]

    def replicate_train(self):
        return [
            self.train_command(task, seed)
            for task in TASKS
            for seed in REPLICATE_TRAINING_SEEDS
        ]

    def train_command(self, task, seed):
        return Command(
            self.trained(task, seed),
            self.train_argv(
                task, seed, self.config['training_root'], TRAINING_EPOCHS
            ),
        )

    def train_smoke(self):
        root = self.root / 'smoke'
        return [
            Command(
                root / 'checkpoints' / f'{task}-seed{PROJECT_TRAINING_SEED}'
                / 'weights_epoch_1.pt',
                self.train_argv(task, PROJECT_TRAINING_SEED, root, 1)
                + [
                    f'+trainer.limit_train_batches={SMOKE_BATCHES}',
                    f'+trainer.limit_val_batches={SMOKE_BATCHES}',
                ],
            )
            for task in SMOKE_TASKS
        ]  # fmt: skip

    def train_argv(self, task, seed, checkpoint_root, epochs):
        return self.python(
            '-m', 'training.lewm',
            f'data={task}',
            f'data.dataset.name={self.config["training_datasets"][task]}',
            f'seed={seed}',
            f'run_name={task}-seed{seed}',
            f'checkpoint_root={checkpoint_root}',
            f'trainer.max_epochs={epochs}',
        )  # fmt: skip

    def replicate_nodes(self):
        return [
            self.build_nodes(
                self.trained(task, training), training, task,
                [self.pairs(task, o, seed) for o in OFFSETS[task]], seed,
                self.replicate_nodes_path(training, task, seed),
            )
            for training in REPLICATE_TRAINING_SEEDS
            for task in TASKS
            for seed in PAIR_SEEDS
        ]  # fmt: skip

    def replicate_cells(self):
        commands = []
        for training in REPLICATE_TRAINING_SEEDS:
            for task in TASKS:
                for seed in PAIR_SEEDS:
                    nodes = self.replicate_nodes_path(training, task, seed)
                    arms = {
                        'l2': self.cem('l2', CEM, 'learned'),
                        'ngc': self.cem('ngc', CEM, 'learned')
                        + self.graph(nodes, NEIGHBORS, RULE),
                    }
                    for offset in OFFSETS[task]:
                        for planner in PLANNER_SEEDS:
                            for cost, arm in arms.items():
                                commands.append(
                                    self.evaluate(
                                        'replicate', task,
                                        f'{cost}-training{training}'
                                        f'-offset{offset}-seed{seed}'
                                        f'-planner{planner}',
                                        self.pairs(task, offset, seed),
                                        self.trained(task, training),
                                        offset, planner, arm,
                                        f'configs/{task}.yaml', 'main',
                                    )
                                )  # fmt: skip
        return commands

    def pairs_stage(self):
        return [
            Command(
                self.pairs(task, offset, seed),
                self.python(
                    'scripts/sample_pairs.py',
                    '--dataset', self.dataset(task),
                    '--task', task,
                    '--rule', 'protocol',
                    '--goal-offset', offset,
                    '--count', PAIR_COUNT,
                    '--seed', seed,
                    '--out', self.pairs(task, offset, seed),
                ),
            )
            for task in TASKS
            for offset in OFFSETS[task]
            for seed in PAIR_SEEDS
        ]  # fmt: skip

    def prepare(self):
        commands = []
        for task in TASKS:
            for planner_seed in PLANNER_SEEDS:
                out = self.lewm_pairs(task, planner_seed)
                commands.append(
                    Command(
                        out,
                        self.python(
                            'scripts/sample_pairs.py',
                            '--dataset', self.dataset(task),
                            '--task', task,
                            '--rule', 'lewm',
                            '--goal-offset', COMPARABILITY_OFFSET,
                            '--count', COMPARABILITY_PAIRS,
                            '--seed', planner_seed,
                            '--out', out,
                        ),
                    )
                )  # fmt: skip
        return commands

    def build_nodes(self, checkpoint, training_seed, task, exclude, seed,
                    out):  # fmt: skip
        return Command(
            out,
            self.python(
                'scripts/build_nodes.py',
                '--dataset', self.dataset(task),
                '--checkpoint', checkpoint,
                *TRAINING_SPLIT,
                '--train-seed', training_seed,
                '--exclude', *exclude,
                '--count', NODE_COUNT[task],
                '--seed', seed,
                '--image-size', 224,
                '--batch-size', ENCODE_BATCH,
                '--out', out,
            ),
        )  # fmt: skip

    def nodes_stage(self):
        commands = []
        for kind in ('project', 'official'):
            for task in TASKS:
                for seed in PAIR_SEEDS:
                    exclude = [
                        self.pairs(task, o, seed) for o in OFFSETS[task]
                    ]
                    commands.append(
                        self.build_nodes(
                            self.checkpoint(kind, task),
                            PROJECT_TRAINING_SEED, task, exclude, seed,
                            self.nodes(kind, task, seed),
                        )
                    )  # fmt: skip
        for kind in ('project', 'official'):
            for task in TASKS:
                exclude = [self.lewm_pairs(task, p) for p in PLANNER_SEEDS]
                commands.append(
                    self.build_nodes(
                        self.checkpoint(kind, task), PROJECT_TRAINING_SEED,
                        task, exclude, COMPARABILITY_NODE_SEED,
                        self.lewm_nodes(kind, task),
                    )
                )  # fmt: skip
        for task in TASKS:
            exclude = [
                self.pairs(task, o, STUDY_PAIR_SEED) for o in OFFSETS[task]
            ]
            for node_seed in REDRAW_NODE_SEEDS:
                commands.append(
                    self.build_nodes(
                        self.checkpoint('project', task),
                        PROJECT_TRAINING_SEED, task, exclude, node_seed,
                        self.redrawn_nodes(task, node_seed),
                    )
                )  # fmt: skip
        commands.append(
            Command(
                self.tworoom_grid(),
                self.python(
                    '-m', 'analysis.tworoom_maps', 'grid',
                    '--checkpoint', self.checkpoint('project', 'tworoom'),
                    '--out', self.tworoom_grid(),
                ),
            )
        )  # fmt: skip
        return commands

    def subsets(self):
        commands = []
        for task in TASKS:
            counts = [int(NODE_COUNT[task] * f) for f in GRAPH_FRACTIONS]
            commands.append(
                Command(
                    self.subset(task, counts[-1]),
                    self.python(
                        'scripts/subset_nodes.py',
                        '--nodes', self.nodes('project', task, STUDY_PAIR_SEED),
                        '--counts', *counts,
                        '--seed', NESTING_SEED,
                        '--out-dir', self.root / 'subsets',
                    ),
                )
            )  # fmt: skip
        return commands

    def evaluate(self, study, task, name, pairs, checkpoint, offset, seed,
                 arm, config, protocol):  # fmt: skip
        out = self.result(study, task, name)
        return Command(
            out,
            self.python(
                'scripts/evaluate.py',
                '--config', config,
                '--pairs', pairs,
                '--dataset', self.dataset(task),
                '--checkpoint', checkpoint,
                '--protocol', protocol,
                *arm,
                '--eval-budget', 2 * offset,
                '--planner-seed', seed,
                '--out', out,
            ),
        )  # fmt: skip

    @staticmethod
    def cem(cost, size, dynamics):
        return [
            '--dynamics', dynamics, '--cost', cost,
            '--num-samples', size[0], '--topk', size[1],
        ]  # fmt: skip

    @staticmethod
    def graph(nodes, neighbors, rule):
        """Graph arguments; ``rule`` is None for NGC_Euc, which has no
        edges."""
        rule_argv = [] if rule is None else ['--rule', rule]
        return [
            '--nodes', nodes, '--neighbors', neighbors, '--anchors', ANCHORS,
            *rule_argv,
        ]  # fmt: skip

    def arms(self, task, seed, kind):
        """L2 cost and NGC on the nodes of one checkpoint kind."""
        return {
            'l2': self.cem('l2', CEM, 'learned'),
            'ngc': self.cem('ngc', CEM, 'learned')
            + self.graph(self.nodes(kind, task, seed), NEIGHBORS, RULE),
        }

    def ladder_cells(self, study, kind, arms):
        commands = []
        for task in TASKS:
            for offset in OFFSETS[task]:
                for seed in PAIR_SEEDS:
                    for planner in PLANNER_SEEDS:
                        for cost, arm in arms(task, seed).items():
                            commands.append(
                                self.evaluate(
                                    study, task,
                                    f'{cost}-offset{offset}-seed{seed}-planner{planner}',
                                    self.pairs(task, offset, seed),
                                    self.checkpoint(kind, task),
                                    offset, planner, arm,
                                    f'configs/{task}.yaml', 'main',
                                )
                            )  # fmt: skip
        return commands

    def single_pair(self, task, offset, seed, pair):
        out = (
            self.root
            / 'pairs-single'
            / task
            / (f'offset{offset}-seed{seed}-pair{pair["pair_id"]:03d}.json')
        )
        return out

    def cells(self):
        commands = self.ladder_cells(
            'ladder', 'project', lambda t, s: self.arms(t, s, 'project')
        )
        commands += self.ladder_cells(
            'ablation',
            'project',
            lambda t, s: {
                'ngc-euc': self.cem('ngc-euc', CEM, 'learned')
                + self.graph(self.nodes('project', t, s), NEIGHBORS, None),
            },
        )
        commands += self.ladder_cells(
            'official', 'official', lambda t, s: self.arms(t, s, 'official')
        )
        for task in TASKS:
            offset = longest(task)
            for label, size in CEM_MULTIPLIERS.items():
                for seed in PAIR_SEEDS:
                    for planner in PLANNER_SEEDS:
                        for cost, arm in (
                            ('l2', self.cem('l2', size, 'learned')),
                            (
                                'ngc',
                                self.cem('ngc', size, 'learned')
                                + self.graph(
                                    self.nodes('project', task, seed),
                                    NEIGHBORS,
                                    RULE,
                                ),
                            ),
                        ):
                            commands.append(
                                self.evaluate(
                                    'budget', task,
                                    f'{cost}-cem{label}-seed{seed}-planner{planner}',
                                    self.pairs(task, offset, seed),
                                    self.checkpoint('project', task),
                                    offset, planner, arm,
                                    f'configs/{task}.yaml', 'main',
                                )
                            )  # fmt: skip
        for task in TASKS:
            seed = STUDY_PAIR_SEED
            nodes = self.nodes('project', task, seed)
            graph_arms = {
                f'k{k}': self.graph(nodes, k, RULE) for k in GRAPH_NEIGHBORS
            }
            for fraction in GRAPH_FRACTIONS:
                count = int(NODE_COUNT[task] * fraction)
                graph_arms[f'n{count}'] = self.graph(
                    self.subset(task, count), NEIGHBORS, RULE
                )
            for offset in (min(OFFSETS[task]), longest(task)):
                for planner in PLANNER_SEEDS:
                    for name, graph in graph_arms.items():
                        commands.append(
                            self.evaluate(
                                'graph', task,
                                f'{name}-offset{offset}-planner{planner}',
                                self.pairs(task, offset, seed),
                                self.checkpoint('project', task),
                                offset, planner, self.cem('ngc', CEM, 'learned') + graph,
                                f'configs/{task}.yaml', 'main',
                            )
                        )  # fmt: skip
        for task in TASKS:
            seed = STUDY_PAIR_SEED
            variants = {
                'rule': {
                    RULE_VARIANT: self.graph(
                        self.nodes('project', task, seed), NEIGHBORS,
                        RULE_VARIANT,
                    ),
                },
                'redraw': {
                    f'nodes{node_seed}': self.graph(
                        self.redrawn_nodes(task, node_seed), NEIGHBORS, RULE
                    )
                    for node_seed in REDRAW_NODE_SEEDS
                },
            }  # fmt: skip
            for study, graphs in variants.items():
                for offset in (min(OFFSETS[task]), longest(task)):
                    for planner in PLANNER_SEEDS:
                        for name, graph in graphs.items():
                            commands.append(
                                self.evaluate(
                                    study, task,
                                    f'{name}-offset{offset}-planner{planner}',
                                    self.pairs(task, offset, seed),
                                    self.checkpoint('project', task),
                                    offset, planner,
                                    self.cem('ngc', CEM, 'learned') + graph,
                                    f'configs/{task}.yaml', 'main',
                                )
                            )  # fmt: skip
        for task in ORACLE_TASKS:
            seed = STUDY_PAIR_SEED
            for offset in OFFSETS[task]:
                pair_set = json.loads(
                    self.pairs(task, offset, seed).read_text()
                )
                for pair in pair_set['pairs']:
                    single = self.single_pair(task, offset, seed, pair)
                    for cost, arm in (
                        ('l2', self.cem('l2', CEM, 'oracle')),
                        (
                            'ngc',
                            self.cem('ngc', CEM, 'oracle')
                            + self.graph(
                                self.nodes('project', task, seed),
                                NEIGHBORS,
                                RULE,
                            ),
                        ),
                    ):
                        commands.append(
                            self.evaluate(
                                'oracle', task,
                                f'{cost}-offset{offset}-pair{pair["pair_id"]:03d}',
                                single, self.checkpoint('project', task),
                                offset, 42, arm, f'configs/{task}.yaml', 'main',
                            )
                        )  # fmt: skip
        for task in TASKS:
            config = (
                'configs/reacher-native.yaml'
                if task == 'reacher'
                else f'configs/{task}.yaml'
            )
            for kind in ('project', 'official'):
                for planner in PLANNER_SEEDS:
                    for cost, arm in (
                        ('l2', self.cem('l2', CEM, 'learned')),
                        (
                            'ngc',
                            self.cem('ngc', CEM, 'learned')
                            + self.graph(
                                self.lewm_nodes(kind, task), NEIGHBORS, RULE
                            ),
                        ),
                    ):
                        commands.append(
                            self.evaluate(
                                'comparability', task,
                                f'{kind}-{cost}-planner{planner}',
                                self.lewm_pairs(task, planner),
                                self.checkpoint(kind, task),
                                COMPARABILITY_OFFSET, planner, arm,
                                config=config, protocol='comparability',
                            )
                        )  # fmt: skip
        return commands

    def analysis(self):
        commands = []
        geometry = [(RULE, 'project', seed) for seed in PAIR_SEEDS]
        geometry.append(
            (RULE_VARIANT, f'project-{RULE_VARIANT}', STUDY_PAIR_SEED)
        )
        for task in TASKS:
            for rule, folder, seed in geometry:
                out = (
                    self.root / 'analysis' / 'geometry' / folder
                    / f'{task}-seed{seed}.json'
                )  # fmt: skip
                commands.append(
                    Command(
                        out,
                        self.python(
                            '-m', 'analysis.geometry',
                            '--task', task,
                            '--nodes', self.nodes('project', task, seed),
                            '--dataset', self.dataset(task),
                            '--checkpoint', self.checkpoint('project', task),
                            '--image-size', 224,
                            '--batch-size', ENCODE_BATCH,
                            '--neighbors', NEIGHBORS,
                            '--anchors', ANCHORS,
                            '--rule', rule,
                            '--out', out,
                        ),
                    )
                )  # fmt: skip
            ladder = self.root / 'results' / 'ladder' / task
            out = self.root / 'analysis' / 'strata' / f'{task}.json'
            commands.append(
                Command(
                    out,
                    self.python(
                        '-m', 'analysis.strata',
                        '--task', task,
                        '--dataset', self.dataset(task),
                        '--l2', *sorted(ladder.glob('l2-*.json')),
                        '--ngc', *sorted(ladder.glob('ngc-*.json')),
                        '--out', out,
                    ),
                )
            )  # fmt: skip
            for seed in PAIR_SEEDS:
                out = (
                    self.root
                    / 'analysis'
                    / 'timing'
                    / f'{task}-seed{seed}.json'
                )
                commands.append(
                    Command(
                        out,
                        self.python(
                            '-m', 'analysis.timing',
                            '--nodes', self.nodes('project', task, seed),
                            '--neighbors', NEIGHBORS,
                            '--anchors', ANCHORS,
                            '--samples', CEM[0],
                            '--out', out,
                        ),
                    )
                )  # fmt: skip
        offset, planner = min(OFFSETS['tworoom']), PLANNER_SEEDS[0]
        run = f'offset{offset}-seed{STUDY_PAIR_SEED}-planner{planner}'
        out = self.root / 'tworoom' / 'maps.npz'
        commands.append(
            Command(
                out,
                self.python(
                    '-m', 'analysis.tworoom_maps', 'maps',
                    '--grid', self.tworoom_grid(),
                    '--checkpoint', self.checkpoint('project', 'tworoom'),
                    '--dataset', self.dataset('tworoom'),
                    '--pairs',
                    self.pairs('tworoom', offset, STUDY_PAIR_SEED),
                    '--nodes',
                    self.nodes('project', 'tworoom', STUDY_PAIR_SEED),
                    '--neighbors', NEIGHBORS,
                    '--anchors', ANCHORS,
                    '--l2', self.result('ladder', 'tworoom', f'l2-{run}'),
                    '--ngc', self.result('ladder', 'tworoom', f'ngc-{run}'),
                    '--out', out,
                ),
            )
        )  # fmt: skip
        return commands

    def stage(self, name):
        return {
            'train-smoke': self.train_smoke,
            'train': self.train,
            'replicate-train': self.replicate_train,
            'replicate-nodes': self.replicate_nodes,
            'replicate-cells': self.replicate_cells,
            'pairs': self.pairs_stage,
            'prepare': self.prepare,
            'nodes': self.nodes_stage,
            'subsets': self.subsets,
            'cells': self.cells,
            'analysis': self.analysis,
        }[name]()
