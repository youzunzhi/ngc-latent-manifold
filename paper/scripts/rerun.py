"""The rerun results behind the paper, loaded for the tables and figures.

Every number of the paper comes from the run root ``RERUN`` (the ``cells``
and ``analysis`` stages of ``experiments/plan.py``), read
through ``analysis.success``: a pair's success is the mean over its planner
seeds, a difference between two arms is taken pair by pair, and its 95%
interval is a bootstrap over pairs, absent when every pair has the same
value. The per-arm interval of a rate is the same bootstrap, drawn
in the order of ``fig_success`` (task, arm, offset) so that the whiskers of
the figure and the intervals of ``tab:success`` are the same numbers.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from analysis import success  # noqa: E402
from experiments.plan import (  # noqa: E402
    CEM_MULTIPLIERS,
    GRAPH_FRACTIONS,
    GRAPH_NEIGHBORS,
    NODE_COUNT,
    OFFSETS,
    ORACLE_TASKS,
    PAIR_SEEDS,
    PLANNER_SEEDS,
    REDRAW_NODE_SEEDS,
    RULE_VARIANT,
    STUDY_PAIR_SEED,
    longest,
)

RERUN = Path(os.environ.get('NGC_RESULTS_ROOT', REPO / 'outputs/paper'))
RESULTS = RERUN / 'results'
# The oracle study has its own run root; every other study reads RESULTS.
ORACLE_RESULTS = Path(os.environ.get('NGC_ORACLE_ROOT', RERUN)) / 'results'
# Physical-distance bins use intervals over episodes.
STRATA = Path(os.environ.get('NGC_STRATA_ROOT', RERUN / 'analysis/strata'))
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
ORACLE_PLANNER_SEED = 42
CHECKPOINTS = ('project', 'official')


def cells(study, task, pattern):
    """Result files of one arm; ``pattern`` is a glob with ``{a,b}``
    alternatives expanded."""
    results = ORACLE_RESULTS if study == 'oracle' else RESULTS
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


def arm(study, task, pattern):
    """Per-pair success of one arm, the planner seeds averaged within a pair."""
    return success.pair_success(cells(study, task, pattern))


def contrast(first, second, rng):
    """``second`` minus ``first`` pair by pair, with the pair arrays."""
    keys, a, b = success.paired(first, second)
    record = success.difference(a, b, np.arange(len(keys)), rng)
    record.update(keys=keys, first_pairs=a, second_pairs=b)
    return record


def ladder(study, l2_prefix, ngc_prefix, pair_seeds, planner_seeds):
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
                study,
                task,
                f'{l2_prefix}-offset{offset}-seed{seeds}-planner{planners}',
            )
            ngc = arm(
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


def main_ladder():
    return ladder('ladder', 'l2', 'ngc', PAIR_SEEDS, PLANNER_SEEDS)


def official_ladder():
    return ladder('official', 'l2', 'ngc', PAIR_SEEDS, PLANNER_SEEDS)


def replicate_ladder(training_seed):
    return ladder(
        'replicate',
        f'l2-training{training_seed}',
        f'ngc-training{training_seed}',
        PAIR_SEEDS,
        PLANNER_SEEDS,
    )


def ablation():
    """L2, NGC_Euc and NGC with the two contrasts of ``tab:ablation``."""
    rng = np.random.default_rng(success.BOOTSTRAP_SEED)
    out = {}
    for task, _ in TASKS:
        out[task] = {}
        for offset in OFFSETS[task]:
            l2 = arm('ladder', task, f'l2-offset{offset}-seed*-planner*')
            euc = arm(
                'ablation', task, f'ngc-euc-offset{offset}-seed*-planner*'
            )
            ngc = arm('ladder', task, f'ngc-offset{offset}-seed*-planner*')
            out[task][offset] = {
                'euc_vs_l2': contrast(l2, euc, rng),
                'ngc_vs_euc': contrast(euc, ngc, rng),
            }
    return out


def budget():
    """Every CEM multiplier at the longest offset: NGC minus L2 at each
    multiplier, and each arm against its own 1x cells."""
    rng = np.random.default_rng(success.BOOTSTRAP_SEED)
    out = {}
    for task, _ in TASKS:
        offset = longest(task)
        arms = {
            '1': {
                'l2': arm('ladder', task, f'l2-offset{offset}-seed*-planner*'),
                'ngc': arm(
                    'ladder', task, f'ngc-offset{offset}-seed*-planner*'
                ),
            }
        }
        for label in CEM_MULTIPLIERS:
            arms[label] = {
                'l2': arm('budget', task, f'l2-cem{label}-seed*-planner*'),
                'ngc': arm('budget', task, f'ngc-cem{label}-seed*-planner*'),
            }
        labels = sorted(arms, key=float)
        out[task] = {
            'offset': offset,
            'multipliers': labels,
            'ngc_vs_l2': {
                m: contrast(arms[m]['l2'], arms[m]['ngc'], rng) for m in labels
            },
            'vs_default': {
                m: {
                    name: contrast(arms['1'][name], arms[m][name], rng)
                    for name in ('l2', 'ngc')
                }
                for m in labels
            },
            'ngc_default_vs_l2': {
                m: contrast(arms[m]['l2'], arms['1']['ngc'], rng)
                for m in labels
            },
        }
    return out


def graph_study():
    """k and node subsets on the study pair seed, against the reference."""
    rng = np.random.default_rng(success.BOOTSTRAP_SEED)
    out = {}
    for task, _ in TASKS:
        out[task] = {}
        for offset in (min(OFFSETS[task]), longest(task)):
            reference = arm(
                'ladder',
                task,
                f'ngc-offset{offset}-seed{STUDY_PAIR_SEED}-planner*',
            )
            l2 = arm(
                'ladder',
                task,
                f'l2-offset{offset}-seed{STUDY_PAIR_SEED}-planner*',
            )
            variants = {}
            for fraction in GRAPH_FRACTIONS:
                count = int(NODE_COUNT[task] * fraction)
                variants[f'n{count}'] = arm(
                    'graph', task, f'n{count}-offset{offset}-planner*'
                )
            for k in GRAPH_NEIGHBORS:
                variants[f'k{k}'] = arm(
                    'graph', task, f'k{k}-offset{offset}-planner*'
                )
            out[task][offset] = {
                'l2': 100 * np.mean([r['success'] for r in l2.values()]),
                'reference': 100
                * np.mean([r['success'] for r in reference.values()]),
                'variants': {
                    name: contrast(reference, records, rng)
                    for name, records in variants.items()
                },
                'bridges': {
                    name: bridges(
                        'graph',
                        task,
                        f'{name}-offset{offset}-planner{PLANNER_SEEDS[0]}',
                    )
                    for name in variants
                },
            }
    return out


def bridges(study, task, name):
    """Number of inter-component edges added to the graph of a cell."""
    result = json.loads((RESULTS / study / task / f'{name}.json').read_text())
    return result['graph']['bridges']


def rule_study():
    """Union rule and the two node redraws on the study pair seed."""
    rng = np.random.default_rng(success.BOOTSTRAP_SEED)
    out = {}
    for task, _ in TASKS:
        out[task] = {}
        for offset in (min(OFFSETS[task]), longest(task)):
            reference = arm(
                'ladder',
                task,
                f'ngc-offset{offset}-seed{STUDY_PAIR_SEED}-planner*',
            )
            l2 = arm(
                'ladder',
                task,
                f'l2-offset{offset}-seed{STUDY_PAIR_SEED}-planner*',
            )
            variants = {
                RULE_VARIANT: arm(
                    'rule', task, f'{RULE_VARIANT}-offset{offset}-planner*'
                )
            }
            for node_seed in REDRAW_NODE_SEEDS:
                variants[f'nodes{node_seed}'] = arm(
                    'redraw', task, f'nodes{node_seed}-offset{offset}-planner*'
                )
            out[task][offset] = {
                'l2': 100 * np.mean([r['success'] for r in l2.values()]),
                'reference': 100
                * np.mean([r['success'] for r in reference.values()]),
                'variants': {
                    name: contrast(reference, records, rng)
                    for name, records in variants.items()
                },
            }
    return out


def oracle():
    """Learned and oracle predictor on the study pair seed, one planner seed:
    the success rates, NGC minus L2 with the oracle predictor, and NGC with
    the oracle predictor minus NGC with the learned one."""
    rng = np.random.default_rng(success.BOOTSTRAP_SEED)
    out = {}
    for task in ORACLE_TASKS:
        out[task] = {}
        for offset in OFFSETS[task]:
            learned = {
                name: arm(
                    'ladder',
                    task,
                    f'{name}-offset{offset}-seed{STUDY_PAIR_SEED}-planner{ORACLE_PLANNER_SEED}',
                )
                for name in ('l2', 'ngc')
            }
            sim = {
                name: arm('oracle', task, f'{name}-offset{offset}-pair*')
                for name in ('l2', 'ngc')
            }
            for name, records in sim.items():
                assert len(records) == 100
                for (pair_set, pair_id), r in records.items():
                    assert r['planner_seeds'] == [ORACLE_PLANNER_SEED], (
                        name,
                        pair_id,
                    )
            learned_contrast = contrast(learned['l2'], learned['ngc'], rng)
            oracle_contrast = contrast(sim['l2'], sim['ngc'], rng)
            assert [k[1] for k in learned_contrast['keys']] == [
                k[1] for k in oracle_contrast['keys']
            ]
            out[task][offset] = {
                'learned': learned_contrast,
                'oracle': oracle_contrast,
                'ngc_change': success.difference(
                    learned_contrast['second_pairs'],
                    oracle_contrast['second_pairs'],
                    np.arange(len(learned_contrast['keys'])),
                    rng,
                ),
            }
    return out


def comparability():
    """The LeWM protocol: mean and SD over the three runs, NGC minus L2 over
    the 150 pairs, for both checkpoints."""
    rng = np.random.default_rng(success.BOOTSTRAP_SEED)
    out = {}
    for task, _ in TASKS:
        out[task] = {}
        for kind in CHECKPOINTS:
            runs = {}
            for name in ('l2', 'ngc'):
                rates = []
                for planner in PLANNER_SEEDS:
                    result = json.loads(
                        (
                            RESULTS
                            / 'comparability'
                            / task
                            / f'{kind}-{name}-planner{planner}.json'
                        ).read_text()
                    )
                    assert (
                        result['protocol'] == 'comparability'
                        and len(result['pairs']) == 50
                    )
                    rates.append(
                        100 * np.mean([p['success'] for p in result['pairs']])
                    )
                runs[name] = np.array(rates)
            l2 = arm('comparability', task, f'{kind}-l2-planner*')
            ngc = arm('comparability', task, f'{kind}-ngc-planner*')
            record = contrast(l2, ngc, rng)
            assert record['pairs'] == 150
            record['runs'] = runs
            out[task][kind] = record
    return out


def timing():
    """Objective, replan and evaluation times at offset 25, per task."""
    out = {}
    for task, _ in TASKS:
        pools = [
            json.loads(
                (
                    RERUN / 'analysis' / 'timing' / f'{task}-seed{s}.json'
                ).read_text()
            )
            for s in PAIR_SEEDS
        ]
        solves, replans, wall = {}, {}, {}
        for name in ('l2', 'ngc'):
            seconds, count = [], 0
            for path in cells(
                'ladder', task, f'{name}-offset25-seed*-planner*'
            ):
                result = json.loads(path.read_text())
                seconds.extend(result['solve_seconds'])
                count += len(result['solve_seconds'])
            solves[name] = float(np.median(seconds))
            replans[name] = count
            wall[name] = float(np.sum(seconds))
        out[task] = {
            'objective_ms': {
                name: float(np.mean([p[f'{name}_ms_per_pool'] for p in pools]))
                for name in ('l2', 'ngc')
            },
            'device': {p['device'] for p in pools},
            'replan_s': solves,
            'replan_ratio': solves['ngc'] / solves['l2'],
            'replans': replans,
            'solve_time_ratio': wall['ngc'] / wall['l2'],
        }
    return out


def strata(task):
    return json.loads((STRATA / f'{task}.json').read_text())


def geometry(task, folder, seed):
    return json.loads(
        (
            RERUN
            / 'analysis'
            / 'geometry'
            / folder
            / f'{task}-seed{seed}.json'
        ).read_text()
    )
