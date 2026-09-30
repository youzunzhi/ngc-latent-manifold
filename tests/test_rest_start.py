"""Rest start on a dataset that, like the HDF5 reader, returns cache views.

The start and goal states are extracted with the evaluator's own
``_extract_init_goal``, so the tests see exactly what ``World.evaluate``
restores.
"""

import numpy as np
import pytest
import torch
from stable_worldmodel.data.dataset import Dataset
from stable_worldmodel.world.world import _extract_init_goal

from evaluation.rest_start import RestStartDataset

EPISODES = [0, 1, 0]
STARTS = [2, 1, 5]
GOAL_OFFSET = 3


class Cached(Dataset):
    """Episodes of random columns held in memory.

    ``_load_slice`` returns tensors that share memory with the columns, as
    the HDF5 reader does for the columns it caches.
    """

    def __init__(self, lengths):
        lengths = np.asarray(lengths)
        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        super().__init__(lengths, offsets)
        rng = np.random.default_rng(0)
        rows = lengths.sum()
        self.cache = {
            'pixels': rng.integers(0, 256, (rows, 4, 4, 3), dtype=np.uint8),
            'action': rng.normal(size=(rows, 2)).astype(np.float32),
            'qpos': rng.normal(size=(rows, 3)).astype(np.float32),
            'qvel': rng.normal(size=(rows, 3)).astype(np.float32),
            'state': rng.normal(size=(rows, 7)).astype(np.float32),
        }

    @property
    def column_names(self):
        return list(self.cache)

    def _load_slice(self, ep_idx, start, end):
        rows = slice(self.offsets[ep_idx] + start, self.offsets[ep_idx] + end)
        steps = {
            col: torch.from_numpy(v[rows]) for col, v in self.cache.items()
        }
        steps['pixels'] = steps['pixels'].permute(0, 3, 1, 2)
        return steps


def extract(dataset):
    init, goal, _ = _extract_init_goal(dataset, EPISODES, STARTS, GOAL_OFFSET)
    return init, goal


@pytest.mark.parametrize(
    'task, column, entries',
    [
        ('cube', 'qvel', [0, 1, 2]),
        ('reacher', 'qvel', [0, 1, 2]),
        # PushT's state ends with the agent velocity.
        ('pusht', 'state', [5, 6]),
        ('tworoom', None, None),
    ],
)
def test_start_state_loses_its_velocity_and_nothing_else(
    task, column, entries
):
    dataset = Cached([12, 9])
    recorded = {col: v.copy() for col, v in dataset.cache.items()}
    init, goal = extract(RestStartDataset(dataset, task))
    expected_init, expected_goal = extract(dataset)
    if column is not None:
        assert (expected_init[column][:, entries] != 0).all()
        expected_init[column][:, entries] = 0
    assert init.keys() == expected_init.keys()
    for key in init:
        np.testing.assert_array_equal(init[key], expected_init[key])
    assert goal.keys() == expected_goal.keys()
    for key in goal:
        np.testing.assert_array_equal(goal[key], expected_goal[key])
    # The recorded columns keep their velocities.
    for col, values in recorded.items():
        np.testing.assert_array_equal(dataset.cache[col], values)


def test_view_forwards_the_dataset_and_rejects_bad_input():
    dataset = Cached([12, 9])
    view = RestStartDataset(dataset, 'cube')
    assert view.column_names == dataset.column_names
    assert view.lengths is dataset.lengths
    # A chunk of one row has its goal on the start row.
    with pytest.raises(AssertionError):
        view.load_chunk(np.array([0]), np.array([2]), np.array([3]))
    with pytest.raises(AssertionError, match='unknown task'):
        RestStartDataset(dataset, 'cube-single')
