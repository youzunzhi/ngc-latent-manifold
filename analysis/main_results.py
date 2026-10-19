"""Main-table aggregation shared by the CLI and paper formatting scripts.

Inputs are result directories, with no dependency on the paper package.
Bootstrap draw order is preserved across both entry points.
"""

from pathlib import Path

import numpy as np

from analysis import success
from experiments.plan import OFFSETS, PAIR_SEEDS, PLANNER_SEEDS

TASKS = [
    ('cube', 'Cube'),
    ('tworoom', 'TwoRoom'),
    ('pusht', 'PushT'),
    ('reacher', 'Reacher'),
]
# The order in which the paper presents the tasks. The bootstrap draws follow
# TASKS, so this order leaves every interval unchanged.
PAPER_TASKS = [
    (task, dict(TASKS)[task])
    for task in ('tworoom', 'reacher', 'cube', 'pusht')
]


def cells(results, study, task, pattern):
    """Result files of one arm; ``pattern`` is a glob with ``{a,b}``
    alternatives expanded."""
    results = Path(results)
    patterns = ['']
    for piece in pattern.replace('}', '{').split('{'):
        options = piece.split(',') if ',' in piece else [piece]
        patterns = [p + o for p in patterns for o in options]
    paths = sorted(
        path
        for p in patterns
        for path in (results / study / task).glob(f'{p}.json')
    )
    assert paths, f'no {study}/{task}/{pattern}.json'
    return paths


def arm(results, study, task, pattern):
    """Per-pair success of one arm, the planner seeds averaged within a pair."""
    return success.pair_success(cells(results, study, task, pattern))


def contrast(first, second, rng):
    """``second`` minus ``first`` pair by pair, with the pair arrays."""
    keys, a, b = success.paired(first, second)
    record = success.difference(a, b, np.arange(len(keys)), rng)
    record.update(keys=keys, first_pairs=a, second_pairs=b)
    return record


def ladder(results, study, l2_prefix, ngc_prefix, pair_seeds, planner_seeds):
    """L2 cost against NGC on every task and offset, pooled over the pair
    seeds: {task: {offset: contrast record with ``by_seed`` rates}}."""
    rng = np.random.default_rng(success.BOOTSTRAP_SEED)
    seeds = '{' + ','.join(map(str, pair_seeds)) + '}'
    planners = '{' + ','.join(map(str, planner_seeds)) + '}'
    out = {}
    for task, _ in TASKS:
        out[task] = {}
        for offset in OFFSETS[task]:
            l2 = arm(
                results,
                study,
                task,
                f'{l2_prefix}-offset{offset}-seed{seeds}-planner{planners}',
            )
            ngc = arm(
                results,
                study,
                task,
                f'{ngc_prefix}-offset{offset}-seed{seeds}-planner{planners}',
            )
            record = contrast(l2, ngc, rng)
            assert record['pairs'] == 100 * len(pair_seeds)
            record['by_seed'] = {
                name: [
                    100
                    * np.mean(
                        [
                            r['success']
                            for k, r in records.items()
                            if f'seed{s}.json' in k[0]
                        ]
                    )
                    for s in pair_seeds
                ]
                for name, records in (('l2', l2), ('ngc', ngc))
            }
            out[task][offset] = record
    return out


def ladder_rate_intervals(table):
    """The per-arm intervals of a ladder table, drawn in figure order."""
    rng = np.random.default_rng(success.BOOTSTRAP_SEED)
    intervals = {}
    for task, _ in TASKS:
        for name in ('first', 'second'):
            for offset, record in table[task].items():
                pairs = record[f'{name}_pairs']
                intervals[task, name, offset] = success.interval(
                    pairs, np.arange(len(pairs)), rng
                )
    return intervals


def main_ladder(results):
    return ladder(results, 'ladder', 'l2', 'ngc', PAIR_SEEDS, PLANNER_SEEDS)


def summarize(root):
    """Return the 12 main-table rows for an experiment output root."""
    table = main_ladder(Path(root) / 'results')
    rates = ladder_rate_intervals(table)
    records = []
    for task, _ in PAPER_TASKS:
        for offset, record in table[task].items():
            records.append(
                {
                    'task': task,
                    'goal_offset': offset,
                    **{
                        k: v
                        for k, v in record.items()
                        if k not in ('keys', 'first_pairs', 'second_pairs')
                    },
                    'l2_ci95': rates[task, 'first', offset],
                    'ngc_ci95': rates[task, 'second', offset],
                }
            )
    return records
