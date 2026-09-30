"""Reacher joint-match success on the MuJoCo environment.

Reacher moves in a horizontal plane, so under zero action an arm at rest
keeps its pose exactly, and each test sets the pose directly.
"""

import gymnasium as gym
import numpy as np
import pytest

from evaluation.reacher import joint_error

STILL = np.zeros(2, dtype=np.float32)


def make_env(qpos, goal_qpos):
    env = gym.make('ngc/ReacherJointMatch-v0', task='qpos_match').unwrapped
    env.reset(seed=0)
    env.set_state(np.asarray(qpos), np.zeros(2))
    env.set_target_qpos(goal_qpos)
    return env


def test_joint_error_wraps_the_shoulder_alone():
    np.testing.assert_allclose(
        joint_error([0.3 + 2 * np.pi, -0.4], [0.3, -0.4]), [0, 0], atol=1e-12
    )
    np.testing.assert_allclose(
        joint_error([0.3, -0.4 + 2 * np.pi], [0.3, -0.4]), [0, 2 * np.pi]
    )
    np.testing.assert_allclose(
        joint_error([np.pi - 0.1, 0], [-np.pi + 0.1, 0]), [0.2, 0]
    )
    with pytest.raises(AssertionError):
        joint_error([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    'goal_qpos, success',
    [
        # A full shoulder turn is the same pose.
        ([0.3 - 2 * np.pi, -0.4], True),
        # The wrist is compared raw, so a full wrist turn is 2 pi away.
        ([0.3, -0.4 + 2 * np.pi], False),
        ([0.3, -0.4 + 0.049], True),
        ([0.3, -0.4 - 0.051], False),
    ],
)
def test_step_terminates_on_a_joint_match(goal_qpos, success):
    env = make_env([0.3, -0.4], goal_qpos)
    _, _, terminated, truncated, _ = env.step(STILL)
    assert terminated is success and not truncated
    np.testing.assert_array_equal(env.env.physics.data.qpos, [0.3, -0.4])


def test_threshold_is_strict():
    env = make_env([0.0, 0.0], [0.0, 0.05])
    assert env.step(STILL)[2] is False
    env = make_env([0.0, 0.0], [0.0, np.nextafter(0.05, 0)])
    assert env.step(STILL)[2] is True


def test_match_does_not_end_the_dm_control_episode():
    # The released task would end its episode on this raw match, and
    # dm_control would reset the arm to a random pose on the next step.
    env = make_env([0.3, -0.4], [0.3, -0.4])
    for _ in range(3):
        assert env.step(STILL)[2] is True
    np.testing.assert_array_equal(env.env.physics.data.qpos, [0.3, -0.4])
    assert env.env.task.target_qpos is None


def test_env_rejects_a_missing_goal_and_the_released_target_option():
    env = gym.make('ngc/ReacherJointMatch-v0', task='qpos_match').unwrapped
    env.reset(seed=0)
    with pytest.raises(AssertionError, match='goal pose not set'):
        env.step(STILL)
    with pytest.raises(AssertionError, match='set_target_qpos'):
        env.reset(seed=0, options={'target_qpos': np.zeros(2)})
    with pytest.raises(AssertionError):
        gym.make('ngc/ReacherJointMatch-v0', task='hard')
