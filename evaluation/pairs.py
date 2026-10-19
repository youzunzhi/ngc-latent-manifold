"""Start-goal pairs: the draw of the main protocol and of the LeWM evaluation.

A pair is a start step of a dataset episode and the step ``goal_offset``
later in the same episode. A pair set is written as ``{goal_offset, pairs}``
with ``{pair_id, episode_id, start_step}`` per pair.

The main protocol (``protocol_pairs``) draws each pair from a different
episode. A candidate pair is kept when its start does not already satisfy
the task's success predicate toward its goal (``solved_at_start``) and, on
TwoRoom, when start and goal lie in different rooms. The draw picks
``count`` episodes uniformly among the episodes with a kept pair, then one
kept start uniformly within each.

The LeWM evaluation (``lewm_pairs``) is the draw of ``scripts/plan/eval_wm.py``
of stable-worldmodel at 4506b1a4: starts are drawn without replacement among
all dataset rows whose goal lies in the same episode, from the first
``len - 1`` of them, so an episode can hold several pairs.
"""

import numpy as np

from analysis.physical import PUSHT_ANGLE_TOL, WALL_X, wrap
from evaluation.reacher import JOINT_THRESHOLD, joint_error

# State columns that the predicates read, and the success thresholds of the
# environments (Cube: ``_compute_successes`` of the OGBench cube
# environment; PushT: ``eval_state`` of the PushT environment; TwoRoom: the
# termination of the TwoRoom environment).
PREDICATE_COLUMN = {
    'cube': 'privileged_block_0_pos',
    'pusht': 'state',
    'reacher': 'qpos',
    'tworoom': 'proprio',
}
CUBE_SUCCESS_M = 0.04
PUSHT_POSITION_TOL = 20.0
TWOROOM_SUCCESS_PX = 16.0


def solved_at_start(task, start, goal):
    """Boolean [P]: the start state already satisfies the success predicate
    toward the goal state. ``start`` and ``goal`` are [P, D] rows of
    ``PREDICATE_COLUMN[task]``."""
    if task == 'cube':
        return np.linalg.norm(goal - start, axis=1) <= CUBE_SUCCESS_M
    if task == 'pusht':
        position = np.linalg.norm(goal[:, :4] - start[:, :4], axis=1)
        angle = np.abs(wrap(goal[:, 4] - start[:, 4]))
        return (position < PUSHT_POSITION_TOL) & (angle < PUSHT_ANGLE_TOL)
    if task == 'reacher':
        return joint_error(start, goal).max(axis=1) < JOINT_THRESHOLD
    assert task == 'tworoom', task
    return np.linalg.norm(goal - start, axis=1) < TWOROOM_SUCCESS_PX


def kept_starts(task, states, lengths, offsets, goal_offset):
    """{episode: start steps of its kept pairs}, over the episodes that keep
    at least one. ``states`` is the ``PREDICATE_COLUMN[task]`` column of
    every dataset row."""
    episodes = np.flatnonzero(np.asarray(lengths) > goal_offset)
    steps = [np.arange(lengths[e] - goal_offset) for e in episodes]
    rows = np.concatenate([offsets[e] + s for e, s in zip(episodes, steps)])
    start, goal = states[rows], states[rows + goal_offset]
    keep = ~solved_at_start(task, start, goal)
    if task == 'tworoom':
        keep &= (start[:, 0] > WALL_X) != (goal[:, 0] > WALL_X)
    kept, first = {}, 0
    for episode, episode_steps in zip(episodes, steps):
        mask = keep[first : first + len(episode_steps)]
        first += len(episode_steps)
        if mask.any():
            kept[int(episode)] = episode_steps[mask]
    return kept


def protocol_pairs(kept, count, seed):
    """``count`` pairs ``{episode_id, start_step}`` from distinct episodes of
    ``kept`` (from ``kept_starts``), in episode order."""
    episodes = sorted(kept)
    assert 1 <= count <= len(episodes), (
        f'{count} pairs requested from {len(episodes)} eligible episodes'
    )
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(episodes), count, replace=False))
    return [
        {
            'episode_id': episodes[index],
            'start_step': int(rng.choice(kept[episodes[index]])),
        }
        for index in chosen
    ]


def lewm_pairs(episodes, steps, goal_offset, count, seed):
    """``count`` pairs ``{episode_id, start_step}`` in row order, from the
    episode and ``step_idx`` columns of every dataset row."""
    episodes, steps = np.asarray(episodes), np.asarray(steps)
    ids, index = np.unique(episodes, return_inverse=True)
    length = np.zeros(len(ids), dtype=steps.dtype)
    np.maximum.at(length, index, steps + 1)
    valid = np.nonzero(steps <= (length - goal_offset - 1)[index])[0]
    rng = np.random.default_rng(seed)
    rows = np.sort(
        valid[rng.choice(len(valid) - 1, size=count, replace=False)]
    )
    return [
        {'episode_id': int(episodes[row]), 'start_step': int(steps[row])}
        for row in rows
    ]
