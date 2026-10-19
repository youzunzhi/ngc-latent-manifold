"""Oracle predictor: candidate blocks executed in the simulator.

``SimulatorOracle`` takes the place of the world model in the cost
evaluator. Its rollout executes every candidate's blocks in a scratch copy
of the environment being planned, from that environment's current state,
renders the state after the last block, and encodes it with the world
model's encoder; the terminal cost is unchanged. The executed blocks that
``HistoryCostEvaluator`` puts in front of the candidates are already part of
the current state, so they are checked and dropped.

A scratch environment copies the current state as follows:

- OGBench Cube (MuJoCo): the complete ``MjData``, derived quantities
  included, since the arm controller reads ``site_xpos`` between steps, and
  the model arrays that a reset randomizes for rendering.
- PushT (pymunk): the episode's inputs replayed through ``ReplayPushT``,
  then checked against the public state of the original.
- TwoRoom (torch): the agent and target positions, the wall and door
  parameters the reset caches, and the values of the variation space
  (speed, radius, colors), which the step and the renderer read.

Candidates are stepped on the unwrapped environment and stop at
termination; only the terminal frame is rendered.

The oracle reads which environment a solve plans from ``env_index``
(``IndexedHistoryPolicy``).
"""

from copy import deepcopy

import mujoco
import numpy as np
import torch
from stable_worldmodel.envs.two_room.env import TwoRoomEnv
from stable_worldmodel.wrapper.default import AddPixelsWrapper
from torch import nn
from torchvision import tv_tensors

from evaluation.pusht_replay import ReplayPushT

# Python attributes of the environments that their step and success rule
# read; copied with the physical state.
PYTHON_STATE = (
    '_prev_qpos',
    '_prev_qvel',
    '_prev_ob_info',
    '_success',
    '_reset_next_step',
    '_target_effector_pose',
    '_cur_goal_ob',
    '_cur_goal_rendered',
    '_render_goal',
    '_target_block',
    '_target_task',
    'cur_task_id',
    'cur_task_info',
    'latest_action',
    'n_contact_points',
    'coverage_arr',
    'goal_state',
    'goal_pose',
    '_goal',
    'agent_position',
    'target_position',
    '_target_img',
    'wall_axis',
    'wall_thickness',
    'num_doors',
    'door_positions',
    'door_sizes',
    'wall_pos',
)
MUJOCO_MODEL_ARRAYS = (
    'geom_rgba',
    'geom_size',
    'mat_rgba',
    'light_ambient',
    'light_diffuse',
    'light_specular',
    'cam_pos',
    'cam_quat',
    'cam_mat0',
    'tex_data',
)
MUJOCO_SIZES = (
    'nq',
    'nv',
    'nu',
    'na',
    'nbody',
    'ngeom',
    'nsite',
    'nmocap',
    'nuserdata',
    'nsensordata',
    'narena',
)
PYMUNK_BODY = (
    'center_of_gravity',
    'angle',
    'angular_velocity',
    'velocity',
    'force',
    'torque',
    'position',
)
PYMUNK_SPACE = (
    'gravity',
    'damping',
    'iterations',
    'idle_speed_threshold',
    'sleep_time_threshold',
    'collision_slop',
    'collision_bias',
    'collision_persistence',
)


def python_state(env):
    state = {
        name: deepcopy(getattr(env, name))
        for name in PYTHON_STATE
        if hasattr(env, name)
    }
    rngs = {
        name: deepcopy(getattr(env, name).bit_generator.state)
        for name in ('np_random', 'rng')
        if hasattr(getattr(env, name, None), 'bit_generator')
    }
    return state, rngs


def restore_python_state(env, state):
    values, rngs = state
    for name, value in values.items():
        setattr(env, name, deepcopy(value))
    for name, value in rngs.items():
        getattr(env, name).bit_generator.state = deepcopy(value)


def is_mujoco(env):
    return hasattr(env, '_model') and hasattr(env, '_data')


def variation_values(space, prefix=''):
    """Current values of a variation space, keyed by dotted path."""
    values = {}
    for name, value in space.value.items():
        path = f'{prefix}{name}'
        if isinstance(value, dict):
            values.update(variation_values(space[name], f'{path}.'))
        else:
            values[path] = deepcopy(value)
    return values


class ScratchEnv:
    """A scratch environment that copies, saves and restores states."""

    def __init__(self, env):
        self.env = env
        self.raw = env.unwrapped
        self.pixels = env
        while not isinstance(self.pixels, AddPixelsWrapper):
            self.pixels = self.pixels.env
        self.ready = False

    def copy_from(self, source):
        """Take the current state of the evaluation environment ``source``."""
        if not self.ready:
            self.env.reset()
            self.ready = True
        if is_mujoco(source):
            assert is_mujoco(self.raw)
            for name in MUJOCO_MODEL_ARRAYS:
                getattr(self.raw._model, name)[...] = getattr(
                    source._model, name
                )
        self.restore(self.save(source))

    @staticmethod
    def save(env):
        if is_mujoco(env):
            data = mujoco.MjData(env._model)
            mujoco.mj_copyData(data, env._model, env._data)
            sizes = tuple(getattr(env._model, n) for n in MUJOCO_SIZES)
            return {'data': data, 'sizes': sizes, 'python': python_state(env)}
        if isinstance(env, TwoRoomEnv):
            return {
                'variation': variation_values(env.variation_space),
                'python': python_state(env),
            }
        assert isinstance(env, ReplayPushT), type(env)
        return {
            'replay': env.replay_state(),
            'bodies': {
                name: {
                    field: deepcopy(getattr(getattr(env, name), field))
                    for field in PYMUNK_BODY
                }
                for name in ('agent', 'block')
            },
            'space': {
                name: deepcopy(getattr(env.space, name))
                for name in PYMUNK_SPACE
            },
            'variation_space': deepcopy(env.variation_space),
            'python': python_state(env),
        }

    def restore(self, state):
        raw = self.raw
        if is_mujoco(raw):
            sizes = tuple(getattr(raw._model, n) for n in MUJOCO_SIZES)
            assert sizes == state['sizes'], 'MuJoCo models differ'
            mujoco.mj_copyData(raw._data, raw._model, state['data'])
        elif isinstance(raw, TwoRoomEnv):
            raw.variation_space.set_value(deepcopy(state['variation']))
        else:
            raw.restore_replay(state['replay'])
            for name, fields in state['bodies'].items():
                body = getattr(raw, name)
                for field, value in fields.items():
                    assert np.array_equal(getattr(body, field), value), (
                        f'replayed {name}.{field} differs'
                    )
            for name, value in state['space'].items():
                assert np.array_equal(getattr(raw.space, name), value), (
                    f'replayed space.{name} differs'
                )
            raw.variation_space = deepcopy(state['variation_space'])
        restore_python_state(raw, state['python'])

    def step(self, action):
        """Step the unwrapped environment.

        The episode flags are ignored: a candidate is executed to its end
        even after the agent reaches the goal, as the learned predictor
        rolls it out, so the terminal frame is the one the candidate
        commands.
        """
        self.raw.step(action)

    def render(self):
        pixels, _ = self.pixels._get_pixels()
        return np.asarray(pixels['pixels']).copy()


class SimulatorOracle(nn.Module):
    """World model whose rollout is the simulator.

    ``source_envs`` are the evaluation environments, ``scratch_envs`` at
    least one scratch environment per solver batch row. ``action_process``
    maps planner actions back to environment actions, and
    ``pixel_transform`` is the policy's image transform.
    ``predicted_emb`` holds the terminal latent alone: [B, S, 1, D].
    """

    def __init__(
        self,
        model,
        source_envs,
        scratch_envs,
        action_block,
        action_process,
        pixel_transform,
        encoder_batch_size,
    ):
        super().__init__()
        self.model = model
        self.source_envs = [env.unwrapped for env in source_envs]
        self.scratch = [ScratchEnv(env) for env in scratch_envs]
        self.action_block = action_block
        self.action_process = action_process
        self.pixel_transform = pixel_transform
        self.encoder_batch_size = encoder_batch_size

    def encode(self, info):
        return self.model.encode(info)

    def environment_actions(self, candidates):
        """Planner blocks [B, S, H, A * block] -> clipped env actions."""
        batch, samples, horizon, packed = candidates.shape
        space = self.scratch[0].env.action_space
        dim = int(np.prod(space.shape))
        assert packed == self.action_block * dim
        flat = (
            candidates.detach()
            .to(dtype=torch.float32, device='cpu')
            .numpy()
            .reshape(-1, dim)
        )
        flat = self.action_process.inverse_transform(flat)
        flat = np.clip(
            flat, space.low.reshape(1, dim), space.high.reshape(1, dim)
        )
        return flat.reshape(
            batch, samples, horizon, self.action_block, *space.shape
        )

    def encode_frames(self, frames, device):
        images = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
        images = self.pixel_transform(tv_tensors.Image(images.contiguous()))
        latents = []
        for start in range(0, len(images), self.encoder_batch_size):
            batch = images[start : start + self.encoder_batch_size]
            encoded = self.model.encode({'pixels': batch[:, None].to(device)})
            latents.append(encoded['emb'][:, 0].detach())
        return torch.cat(latents)

    @torch.inference_mode()
    def rollout(self, info, action_sequence):
        candidates = action_sequence
        if 'history_action' in info:
            history = info['history_action']
            blocks = history.shape[-2]
            assert torch.equal(
                action_sequence[..., :blocks, :],
                history.to(action_sequence.device, action_sequence.dtype),
            ), 'the rollout does not start with the executed blocks'
            candidates = action_sequence[..., blocks:, :]
        batch, samples, horizon, _ = candidates.shape
        assert horizon >= 1 and batch <= len(self.scratch)
        # The policy's info preprocessing may move the index to the device.
        env_index = torch.as_tensor(info['env_index']).cpu().numpy()
        env_index = env_index.reshape(batch, -1)[:, 0]
        actions = self.environment_actions(candidates)
        frames = []
        for row, index in enumerate(env_index):
            scratch = self.scratch[row]
            scratch.copy_from(self.source_envs[int(index)])
            start = scratch.save(scratch.raw)
            for sample in range(samples):
                scratch.restore(start)
                for action in actions[row, sample].reshape(
                    -1, *actions.shape[4:]
                ):
                    scratch.step(action)
                frames.append(scratch.render())
            scratch.restore(start)
        device = next(self.model.parameters()).device
        predicted = self.encode_frames(frames, device)
        info['predicted_emb'] = predicted.reshape(batch, samples, 1, -1)
        return info
