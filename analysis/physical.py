"""Physical distance from one simulator state to a goal state.

Each task's distance is a proxy of the work left to reach the goal, ordered
from the query state to the goal state:

- TwoRoom: the length of the shortest path of the agent center to the goal
  position, through the door when the two lie in different rooms (px).
- Reacher: the joint-space distance (rad), the shoulder angle compared on
  the circle; the wrist is a bounded hinge and is compared directly.
- Cube: the distance from the gripper to the cube plus the distance from
  the cube to the goal cube (m).
- PushT: the distance from the pusher to the block's centroid plus the
  root-mean-square displacement of the block's points from its pose to the
  goal pose (px), which counts both the translation and the rotation of the
  block.

States are read from the dataset rows by ``read_states``.
"""

import numpy as np
import stable_worldmodel as swm
from scipy.spatial.distance import cdist

TASKS = ('tworoom', 'reacher', 'cube', 'pusht')
STATE_COLUMNS = {
    'tworoom': ('proprio',),
    'reacher': ('qpos',),
    'cube': ('privileged_block_0_pos', 'proprio_effector_pos'),
    'pusht': ('state',),
}
UNIT = {'tworoom': 'px', 'reacher': 'rad', 'cube': 'm', 'pusht': 'px'}
# TwoRoom: the wall's center line and the stretch of it that the agent
# center can cross (the door, narrowed by the agent radius), in pixels.
WALL_X = 112.0
DOOR_Y = (42.0, 56.0)
DOOR_SAMPLES = 141
PUSHT_ANGLE_TOL = np.pi / 9
QUERY_CHUNK = 64
# PushT: the T block of stable-worldmodel's ``PushT.add_tee``, a bar of
# TEE_LENGTH x 1 and a stem of 1 x (TEE_LENGTH - 1) squares of TEE_SCALE px
# in body coordinates, as (x_min, x_max, y_min, y_max).
TEE_SCALE = 30.0
TEE_LENGTH = 4.0
TEE_RECTANGLES = (
    (-TEE_LENGTH * TEE_SCALE / 2, TEE_LENGTH * TEE_SCALE / 2, 0, TEE_SCALE),
    (-TEE_SCALE / 2, TEE_SCALE / 2, TEE_SCALE, TEE_LENGTH * TEE_SCALE),
)


def tee_geometry():
    """Area centroid [2] in body coordinates, and the mean squared distance
    of the block's points from it.

    The recorded block position is the body origin, so a point ``p`` in body
    coordinates lies at ``position + R(angle) p``.
    """
    rectangles = np.asarray(TEE_RECTANGLES)
    width = rectangles[:, 1] - rectangles[:, 0]
    height = rectangles[:, 3] - rectangles[:, 2]
    area = width * height
    centers = np.stack(
        [rectangles[:, :2].mean(axis=1), rectangles[:, 2:].mean(axis=1)],
        axis=1,
    )
    centroid = (area[:, None] * centers).sum(axis=0) / area.sum()
    spread = (width**2 + height**2) / 12 + ((centers - centroid) ** 2).sum(1)
    second_moment = (area * spread).sum() / area.sum()
    return centroid, second_moment


TEE_CENTROID, TEE_SECOND_MOMENT = tee_geometry()


def episode_rows(dataset_path, episodes, steps):
    """Dataset rows of the given steps of the given episodes."""
    dataset = swm.data.load_dataset(dataset_path, keys_to_load=['action'])
    lengths = np.asarray(dataset.lengths)[np.asarray(episodes)]
    assert np.all((0 <= np.asarray(steps)) & (np.asarray(steps) < lengths))
    return np.asarray(dataset.offsets)[np.asarray(episodes)] + np.asarray(
        steps
    )


def read_states(dataset_path, task, rows):
    """State columns of ``task`` at the dataset ``rows``, as float64, in
    the order of ``rows``; HDF5 reads the rows sorted and without
    repeats."""
    dataset = swm.data.load_dataset(
        dataset_path, keys_to_load=list(STATE_COLUMNS[task])
    )
    unique, inverse = np.unique(np.asarray(rows), return_inverse=True)
    data = dataset.get_row_data(unique.tolist())
    return {
        column: np.asarray(data[column], dtype=np.float64)[inverse]
        for column in STATE_COLUMNS[task]
    }


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def door_path(query, goal):
    """TwoRoom path length [Q, G] between agent positions."""
    door = np.stack(
        [np.full(DOOR_SAMPLES, WALL_X), np.linspace(*DOOR_Y, DOOR_SAMPLES)],
        axis=1,
    )
    path = cdist(query, goal)
    cross = (query[:, 0] > WALL_X)[:, None] != (goal[:, 0] > WALL_X)[None, :]
    to_door = cdist(goal, door)
    for start in range(0, len(query), QUERY_CHUNK):
        rows = slice(start, start + QUERY_CHUNK)
        via = (cdist(query[rows], door)[:, None, :] + to_door[None, :, :]).min(
            axis=-1
        )
        path[rows] = np.where(cross[rows], via, path[rows])
    return path


def tee_centroids(state):
    """World position [N, 2] of the block's area centroid."""
    angle = state[:, 4]
    cos, sin = np.cos(angle), np.sin(angle)
    x, y = TEE_CENTROID
    offset = np.stack([cos * x - sin * y, sin * x + cos * y], axis=1)
    return state[:, 2:4] + offset


def block_displacement(query, goal):
    """PushT root-mean-square displacement [Q, G] of the block's points.

    Splitting each point into the area centroid plus its offset from the
    centroid, the cross term averages to zero, and a rotation by ``a`` moves
    an offset ``v`` by ``2 (1 - cos a) |v|^2`` in squared length.
    """
    shift = cdist(tee_centroids(query), tee_centroids(goal)) ** 2
    turn = query[:, None, 4] - goal[None, :, 4]
    return np.sqrt(shift + 2 * (1 - np.cos(turn)) * TEE_SECOND_MOMENT)


def distance(task, query, goal):
    """Physical distance [Q, G] from every query state to every goal state."""
    if task == 'tworoom':
        return door_path(query['proprio'], goal['proprio'])
    if task == 'reacher':
        q, g = query['qpos'], goal['qpos']
        shoulder = wrap(q[:, None, 0] - g[None, :, 0])
        wrist = q[:, None, 1] - g[None, :, 1]
        return np.sqrt(shoulder**2 + wrist**2)
    if task == 'cube':
        cube = query['privileged_block_0_pos']
        reach = np.linalg.norm(query['proprio_effector_pos'] - cube, axis=1)
        return reach[:, None] + cdist(cube, goal['privileged_block_0_pos'])
    assert task == 'pusht', task
    q, g = query['state'], goal['state']
    reach = np.linalg.norm(q[:, 0:2] - tee_centroids(q), axis=1)
    return reach[:, None] + block_displacement(q, g)
