"""Per-pair success from evaluation results, and paired differences.

A result of ``scripts/evaluate.py`` holds one planner seed on one pair set.
A pair is identified by its pair set and ``pair_id``; its success is the
mean over the planner seeds evaluated on it, so planner seeds are repeated
measurements within a pair, never independent samples. A difference
between two methods is taken pair by pair, and its 95% interval is a
bootstrap over resampling units: the pairs, or their episodes when pairs of
one episode are pooled across offsets.
"""

import json
from pathlib import Path

import numpy as np

BOOTSTRAP_RESAMPLES = 50000
BOOTSTRAP_SEED = 20260920


def pair_success(paths):
    """{(pair_set, pair_id): record} with the success averaged over seeds.

    A record holds ``goal_offset``, ``episode_id``, ``start_step``,
    ``planner_seeds`` and ``success`` (the mean, in [0, 1]).
    """
    seen, records = set(), {}
    for path in paths:
        result = json.loads(Path(path).read_text())
        run = (result['pair_set'], result['planner_seed'])
        assert run not in seen, f'planner seed evaluated twice: {path}'
        seen.add(run)
        for pair in result['pairs']:
            key = (result['pair_set'], pair['pair_id'])
            record = records.setdefault(
                key,
                {
                    'goal_offset': result['goal_offset'],
                    'episode_id': pair['episode_id'],
                    'start_step': pair['start_step'],
                    'planner_seeds': [],
                    'successes': [],
                },
            )
            assert record['episode_id'] == pair['episode_id']
            assert record['start_step'] == pair['start_step']
            record['planner_seeds'].append(result['planner_seed'])
            record['successes'].append(float(pair['success']))
    for record in records.values():
        record['success'] = float(np.mean(record.pop('successes')))
    return records


def paired(first, second):
    """Keys and success arrays of two methods on the same pairs and seeds."""
    assert first.keys() == second.keys(), 'the methods cover different pairs'
    keys = sorted(first)
    for key in keys:
        assert sorted(first[key]['planner_seeds']) == sorted(
            second[key]['planner_seeds']
        ), f'{key}: different planner seeds'
    return (
        keys,
        np.array([first[k]['success'] for k in keys]),
        np.array([second[k]['success'] for k in keys]),
    )


def interval(values, units, rng):
    """Bootstrap 95% interval of the mean of ``values`` in percent, resampling
    ``units`` with all their values; None when every value is equal, where
    every resample has the same mean."""
    _, unit = np.unique(units, return_inverse=True)
    sums = np.bincount(unit, weights=values)
    counts = np.bincount(unit)
    draws = rng.integers(0, len(sums), size=(BOOTSTRAP_RESAMPLES, len(sums)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    if (values == values[0]).all():
        return None
    return [float(v) for v in 100 * np.quantile(means, [0.025, 0.975])]


def difference(first, second, units, rng):
    """Mean of ``second - first`` in percent and its bootstrap interval."""
    delta = second - first
    return {
        'pairs': int(len(delta)),
        'first': float(100 * first.mean()),
        'second': float(100 * second.mean()),
        'delta': float(100 * delta.mean()),
        'ci95': interval(delta, units, rng),
        'pairs_gained': int((delta > 0).sum()),
        'pairs_lost': int((delta < 0).sum()),
    }
