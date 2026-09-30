"""Reacher success: every joint within 0.05 rad of the goal pose.

The released ``qpos_match`` task compares raw joint angles after every
physics step, and a match ends the dm_control episode. The shoulder is an
unlimited hinge, so two poses one full turn apart look the same but differ
by 2 pi in raw angle. Here the shoulder error is taken on the circle and
the wrist error, limited to +-160 degrees, stays raw. Success is checked
once per environment step, after its action repeat, and the dm_control
episode never ends on a match.

Importing this module registers ``ngc/ReacherJointMatch-v0``.
"""

import gymnasium as gym
import numpy as np
from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper

JOINT_THRESHOLD = 0.05


def joint_error(qpos, goal_qpos):
    """Absolute [shoulder, wrist] error; the shoulder's is at most pi."""
    delta = np.asarray(qpos, dtype=np.float64) - np.asarray(
        goal_qpos, dtype=np.float64
    )
    assert delta.shape == (2,)
    error = np.abs(delta)
    error[0] = np.abs(np.arctan2(np.sin(delta[0]), np.cos(delta[0])))
    return error


class ReacherJointMatch(ReacherDMControlWrapper):
    """Reacher whose step terminates exactly when ``joint_error`` of every
    joint is below ``JOINT_THRESHOLD``.

    The goal pose set by ``set_target_qpos`` is kept here, and the
    dm_control task's own target stays unset, so the task never ends the
    episode.
    """

    def __init__(self, **kwargs):
        assert kwargs['task'] == 'qpos_match'
        super().__init__(**kwargs)
        self.goal_qpos = None

    def set_target_qpos(self, target_qpos):
        self.goal_qpos = np.asarray(target_qpos, dtype=np.float64)

    def reset(self, seed=None, options=None):
        assert options is None or 'target_qpos' not in options, (
            'set the goal pose with set_target_qpos'
        )
        return super().reset(seed=seed, options=options)

    def step(self, action):
        assert self.goal_qpos is not None, 'goal pose not set'
        assert self.env.task.target_qpos is None
        observation, reward, terminated, truncated, info = super().step(action)
        assert not terminated
        error = joint_error(self.env.physics.data.qpos, self.goal_qpos)
        success = bool(error.max() < JOINT_THRESHOLD)
        return observation, reward, success, truncated, info


gym.register(
    id='ngc/ReacherJointMatch-v0',
    entry_point='evaluation.reacher:ReacherJointMatch',
)
