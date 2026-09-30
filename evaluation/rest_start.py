"""Rest start: each pair starts from its recorded pose at zero velocity.

The evaluator restores the recorded state of a pair's start row, which
includes the velocity the demonstration had at that row. The planner sees
only the start frame, which does not show that velocity. Rest start sets
the velocity part of the start state to zero, so the planner does not act
from a motion it cannot see. The goal state is unchanged.

``VELOCITY`` names, per task, the dataset column and the entries of that
column that hold velocity. TwoRoom restores the agent position alone and
has no velocity to zero.
"""

import torch

VELOCITY = {
    'cube': ('qvel', slice(None)),
    'pusht': ('state', slice(-2, None)),
    'reacher': ('qvel', slice(None)),
    'tworoom': None,
}


class RestStartDataset:
    """Dataset view whose loaded chunks start at zero velocity.

    ``load_chunk`` zeroes the velocity entries of the first row of every
    chunk, the row ``World.evaluate`` restores as the start state; every
    other row, the goal row included, is left as recorded. All other
    attributes are those of ``dataset``.
    """

    def __init__(self, dataset, task):
        assert task in VELOCITY, f'unknown task {task!r}'
        self.dataset = dataset
        self.velocity = VELOCITY[task]

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def load_chunk(self, episodes_idx, start, end):
        chunks = self.dataset.load_chunk(episodes_idx, start, end)
        if self.velocity is None:
            return chunks
        column, entries = self.velocity
        for chunk in chunks:
            values = chunk[column]
            assert torch.is_tensor(values) and len(values) > 1
            # The loader may return a view of its column cache.
            values = values.clone()
            values[0, entries] = 0
            chunk[column] = values
        return chunks
