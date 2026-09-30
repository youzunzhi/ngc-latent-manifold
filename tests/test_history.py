"""Executed-rollout history on a scripted solver and environment.

Every frame records its step in channel 0 and its environment in channel
1, so the frames a replan sees tell which steps they came from. Actions are
normalized by ``x -> (x - 1) / 2`` and the action space is [-1, 1], so part
of every plan is clipped.
"""

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_worldmodel.policy import PlanConfig
from torch import nn

from evaluation.history import HistoryCostEvaluator, HistoryPolicy

BLOCK = 5
ACTION_DIM = 2


class Affine:
    """Action normalization ``x -> (x - 1) / 2``."""

    def transform(self, x):
        return (x - 1) / 2

    def inverse_transform(self, x):
        return x * 2 + 1


class Envs:
    """Vector-environment stand-in with a [-1, 1] action box."""

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.single_action_space = spaces.Box(-1, 1, (ACTION_DIM,))
        self.action_space = spaces.Box(-1, 1, (num_envs, ACTION_DIM))


class ScriptedSolver:
    """Returns a fixed ramp per call and records what each call received."""

    def __init__(self):
        self.calls = []

    def configure(self, *, action_space, n_envs, config):
        self.config = config

    @property
    def action_dim(self):
        return ACTION_DIM * BLOCK

    @property
    def n_envs(self):
        return 1

    @property
    def horizon(self):
        return self.config.horizon

    def solve(self, info_dict, init_action=None):
        count = self.config.horizon * self.action_dim
        ramp = torch.linspace(-1.5, 0.5, count) + 0.01 * len(self.calls)
        actions = ramp.reshape(1, self.config.horizon, self.action_dim)
        self.calls.append(
            {
                'steps': info_dict['pixels'][0, :, 0, 0, 0].tolist(),
                'env': int(info_dict['pixels'][0, -1, 0, 0, 1]),
                'history_action': info_dict.get('history_action'),
                'init_action': init_action,
                'actions': actions,
            }
        )
        return {'actions': actions}

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)


def make_policy(num_envs, horizon):
    config = PlanConfig(
        horizon=horizon,
        receding_horizon=5,
        history_len=3,
        action_block=BLOCK,
        warm_start=True,
    )
    solver = ScriptedSolver()
    policy = HistoryPolicy(
        solver=solver, config=config, process={'action': Affine()}
    )
    policy.set_env(Envs(num_envs))
    return policy, solver


def observation(step, num_envs, terminated):
    pixels = np.zeros((num_envs, 1, 2, 2, 3), dtype=np.uint8)
    pixels[..., 0] = step
    pixels[..., 1] = np.arange(num_envs)[:, None, None, None]
    return {
        'pixels': pixels,
        'terminated': np.asarray(terminated)[:, None],
    }


def test_replans_see_block_endpoint_frames_and_clipped_blocks():
    policy, solver = make_policy(num_envs=1, horizon=5)
    returned = [
        policy.get_action(observation(step, 1, [False]))[0]
        for step in range(26)
    ]
    first, second = solver.calls
    # The first replan sees the start frame alone; after five blocks it
    # sees the last three block endpoints and the two blocks between them.
    assert first['steps'] == [0] and first['history_action'] is None
    assert second['steps'] == [15, 20, 25]
    executed = np.stack(returned[15:25])
    np.testing.assert_array_equal(
        second['history_action'].numpy(),
        ((executed - 1) / 2).reshape(1, 2, BLOCK * ACTION_DIM),
    )
    # The plan leaves the box, so the recorded blocks are the clipped ones.
    planned = first['actions'].reshape(-1, ACTION_DIM).numpy() * 2 + 1
    assert np.abs(planned).max() > 1
    np.testing.assert_array_equal(
        np.stack(returned[:25]), np.clip(planned, -1, 1)
    )


def test_each_live_environment_is_planned_alone():
    # Environment 1 terminates when its buffer runs out, so it is due for
    # a replan but gets none. The plan is one block longer than the
    # receding horizon, so that block warm-starts the next replan of the
    # same environment.
    policy, solver = make_policy(num_envs=2, horizon=6)
    returned = [
        policy.get_action(observation(step, 2, [False, step >= 25]))
        for step in range(26)
    ]
    assert [call['env'] for call in solver.calls] == [0, 1, 0]
    assert solver.calls[2]['steps'] == [15, 20, 25]
    # The first replan allocates a zero warm start for every environment.
    assert solver.calls[0]['init_action'] is None
    assert not solver.calls[1]['init_action'].any()
    torch.testing.assert_close(
        solver.calls[2]['init_action'], solver.calls[0]['actions'][:, 5:]
    )
    returned = np.stack(returned)
    assert np.isfinite(returned[:, 0]).all()
    assert np.isfinite(returned[:25, 1]).all()
    assert np.isnan(returned[25:, 1]).all()


def test_policy_rejects_resets_and_plans_without_history():
    policy, _ = make_policy(num_envs=1, horizon=5)
    info = observation(0, 1, [False])
    info['_needs_flush'] = np.array([True])
    with pytest.raises(AssertionError, match='reset during the evaluation'):
        policy.get_action(info)
    config = PlanConfig(horizon=5, receding_horizon=5, history_len=1)
    policy = HistoryPolicy(solver=ScriptedSolver(), config=config)
    with pytest.raises(AssertionError, match='use WorldModelPolicy'):
        policy.set_env(Envs(1))


class RecordingModel(nn.Module):
    """World-model stand-in that records the actions of each rollout."""

    def rollout(self, info, actions):
        self.actions = actions
        return info


def test_evaluator_replays_history_before_candidates():
    model = RecordingModel()
    evaluator = HistoryCostEvaluator(
        model,
        lambda info: info['action_candidates'].sum(dim=(-2, -1)),
        encode_goal=None,
    )
    history = torch.randn(1, 4, 2, BLOCK * ACTION_DIM)
    candidates = torch.randn(1, 4, 5, BLOCK * ACTION_DIM)
    cost = evaluator.get_cost({'history_action': history}, candidates)
    torch.testing.assert_close(
        model.actions, torch.cat((history, candidates), dim=2)
    )
    torch.testing.assert_close(cost, candidates.sum(dim=(-2, -1)))
    evaluator.get_cost({}, candidates)
    assert model.actions is candidates
    with pytest.raises(AssertionError):
        evaluator.get_cost({'history_action': history[:, :3]}, candidates)
