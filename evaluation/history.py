"""Planning conditioned on the executed rollout.

At each replan the world model sees up to ``history_len`` real frames, the
frames at the ends of the most recent executed action blocks, together with
the blocks executed between them. The first replan sees the start frame
alone. The rollout starts from the latents of these frames and first
replays the executed blocks, then the candidate blocks.

``HistoryPolicy`` records every action it returns and, once a block is
complete, the frame the next call receives. ``HistoryCostEvaluator`` puts
the executed blocks in front of the candidates before the rollout.
"""

from collections import deque

import numpy as np
import torch
from stable_worldmodel.planning.evaluator import ShootingCostEvaluator
from stable_worldmodel.policy import WorldModelPolicy


class HistoryPolicy(WorldModelPolicy):
    """``WorldModelPolicy`` whose replans see the executed rollout.

    Each environment is planned in its own solver call, since environments
    may hold histories of different lengths. Returned actions are clipped
    to the action space, and the history records the clipped actions that
    the environments execute.

    The history starts empty at ``set_env``, so the policy serves one
    dataset evaluation per ``World.set_policy``. Planning environments
    are never reset during that evaluation (``reset_mode='wait'``).
    """

    def set_env(self, env):
        super().set_env(env)
        assert self.cfg.history_len > 1, (
            'history_len 1 plans without history: use WorldModelPolicy'
        )
        self._pixel_history = [
            deque(maxlen=self.cfg.history_len) for _ in range(env.num_envs)
        ]
        self._action_history = [
            deque(maxlen=self.cfg.history_len - 1) for _ in range(env.num_envs)
        ]
        self._pending_actions = [[] for _ in range(env.num_envs)]
        self._action_low = env.single_action_space.low
        self._action_high = env.single_action_space.high

    def get_action(self, info_dict, **kwargs):
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        assert '_needs_flush' not in info_dict, (
            'an environment was reset during the evaluation'
        )
        n_envs = self.env.num_envs
        dead = (
            np.asarray(info_dict['terminated'], dtype=bool)
            .reshape(n_envs, -1)
            .any(axis=1)
        )

        # The frame reached by the last action of a block ends the block.
        for i in range(n_envs):
            pending = self._pending_actions[i]
            assert len(pending) <= self.cfg.action_block
            if len(pending) == self.cfg.action_block:
                self._action_history[i].append(np.stack(pending))
                self._pixel_history[i].append(
                    info_dict['pixels'][i, -1].copy()
                )
                pending.clear()

        for i in range(n_envs):
            if dead[i] or len(self._action_buffer[i]) > 0:
                continue
            self._replan(info_dict, i)

        single_shape = self.env.single_action_space.shape
        action = torch.full(
            (n_envs, *single_shape),
            fill_value=float('nan'),
            dtype=torch.float32,
        )
        for i in range(n_envs):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()
        action = action.reshape(*self.env.action_space.shape).numpy()
        if 'action' in self.process:
            action = self.process['action'].inverse_transform(action)
        action = np.clip(action, self._action_low, self._action_high)

        executed = action.reshape(n_envs, *single_shape)
        for i in range(n_envs):
            if not dead[i]:
                self._pending_actions[i].append(executed[i].copy())
        return action

    def _replan(self, info_dict, i):
        """Solve for environment ``i`` and fill its action buffer."""
        sliced = {}
        for key, value in info_dict.items():
            if torch.is_tensor(value):
                sliced[key] = value[torch.as_tensor([i])]
            elif isinstance(value, np.ndarray):
                sliced[key] = value[[i]]
            elif isinstance(value, list):
                sliced[key] = [value[i]]
            else:
                sliced[key] = value
        if not self._pixel_history[i]:
            self._pixel_history[i].append(info_dict['pixels'][i, -1].copy())
        sliced['pixels'] = np.stack(self._pixel_history[i])[None]
        sliced = self._prepare_info(sliced)
        if self._action_history[i]:
            sliced['history_action'] = self._history_action(i)

        idx_tensor = torch.as_tensor([i])
        init_action = (
            self._next_init[idx_tensor]
            if self._next_init is not None
            else None
        )
        actions = self.solver(sliced, init_action=init_action)['actions']
        plan = actions[:, : self.cfg.receding_horizon]
        rest = actions[:, self.cfg.receding_horizon :]
        if self.cfg.warm_start and rest.shape[1] > 0:
            if self._next_init is None:
                self._next_init = torch.zeros(
                    self.env.num_envs,
                    rest.shape[1],
                    rest.shape[2],
                    dtype=rest.dtype,
                )
            self._next_init[idx_tensor] = rest
        elif not self.cfg.warm_start:
            self._next_init = None
        plan = plan.reshape(self.flatten_receding_horizon, -1)
        self._action_buffer[i].extend(plan)

    def _history_action(self, i):
        """Executed blocks of environment ``i`` in planner coordinates.

        Shape [1, blocks, action_block * action_dim], matching the
        candidate blocks of the solver.
        """
        blocks = np.stack(self._action_history[i])
        flat = blocks.reshape(-1, *blocks.shape[2:])
        if 'action' in self.process:
            flat = self.process['action'].transform(flat)
        return torch.from_numpy(
            np.asarray(flat).reshape(1, blocks.shape[0], -1)
        )


class HistoryCostEvaluator(ShootingCostEvaluator):
    """``ShootingCostEvaluator`` that replays ``history_action`` first.

    ``history_action`` [B, S, blocks, A] is prepended to the candidates
    [B, S, horizon, A] before the rollout. ``action_candidates`` stays the
    candidates alone.
    """

    def _rollout(self, info_dict, action_candidates):
        if self.encode_goal is not None and 'goal_emb' not in info_dict:
            info_dict['goal_emb'] = self.encode_goal(self.model, info_dict)
        actions = action_candidates
        if 'history_action' in info_dict:
            history_action = info_dict['history_action']
            assert history_action.shape[:-2] == action_candidates.shape[:-2]
            assert history_action.shape[-1] == action_candidates.shape[-1]
            actions = torch.cat((history_action, action_candidates), dim=-2)
        info_dict = self.model.rollout(info_dict, actions)
        info_dict['action_candidates'] = action_candidates
        return info_dict
