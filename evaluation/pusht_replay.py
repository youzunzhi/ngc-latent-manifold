"""PushT that can rebuild its exact physical state by replay.

Copying the public pose and velocity of the bodies loses the contact
solver's internal state (body bias, collision history), so a copy of a
PushT in contact steps differently from the original. ``ReplayPushT``
records the values its reset drew, the dataset setters and every executed
action; ``restore_replay`` runs the same inputs through a fresh reset, which
rebuilds the solver state exactly. Recording does not change the physics.

A reset without a seed uses ``reset_seed``. The evaluation resets every
environment without a seed and then sets the pair's start state, a step of
physics that begins from whatever state the reset drew; an unseeded reset
would let that state, and through contact the start, vary between runs.

Registered as ``ngc/PushTReplay-v1``.
"""

from copy import deepcopy

import gymnasium as gym
from stable_worldmodel.envs.pusht.env import PushT

# Attributes the setup, controller and renderer read; they must not change
# between a reset and a replay of it.
REPLAY_SETTINGS = (
    'block_cog',
    'damping',
    'relative',
    'action_scale',
    'k_p',
    'k_v',
    'dt',
    'control_hz',
    'window_size',
    'render_size',
    'with_target',
    'render_action',
)
REPLAYED_METHODS = ('step', '_set_state', '_set_goal_state')


class ReplayPushT(PushT):
    """``PushT`` that records the inputs of the current episode."""

    def __init__(self, reset_seed, **kwargs):
        self.replaying = False
        self.reset_seed = reset_seed
        super().__init__(**kwargs)

    def reset(self, seed=None, options=None):
        if seed is None:
            seed = self.reset_seed
        self.reset_events = []
        self.replay_events = None
        observation, info = super().reset(seed=seed, options=options)
        if not self.replaying:
            # A reset sets the goal state on the bodies, records it as the
            # goal, then sets the start state.
            (goal, _), (goal_again, _), (start, _) = self.reset_events
            assert [name for _, name in self.reset_events] == [
                '_set_state',
                '_set_goal_state',
                '_set_state',
            ]
            assert goal is goal_again
            self.replay_reset = {
                'variation': [],
                'variation_values': deepcopy(self.variation_space.value),
                'goal_state': deepcopy(goal),
                'state': deepcopy(start),
            }
            self.replay_settings = {
                name: deepcopy(getattr(self, name)) for name in REPLAY_SETTINGS
            }
            self.replay_events = []
        return observation, info

    def record(self, method, value):
        if self.replaying:
            return
        if self.replay_events is None:
            self.reset_events.append((value, method))
        else:
            self.replay_events.append((method, deepcopy(value)))

    def step(self, action):
        self.record('step', action)
        return super().step(action)

    def _set_state(self, state):
        self.record('_set_state', state)
        super()._set_state(state)

    def _set_goal_state(self, goal_state):
        self.record('_set_goal_state', goal_state)
        super()._set_goal_state(goal_state)

    def replay_state(self):
        """The inputs of the episode so far: reset values, setters, actions."""
        assert self.replay_events is not None, 'no completed reset'
        for name in REPLAY_SETTINGS:
            assert repr(getattr(self, name)) == repr(
                self.replay_settings[name]
            ), f'{name} changed after the reset'
        return deepcopy(
            {
                'settings': self.replay_settings,
                'reset': self.replay_reset,
                'events': self.replay_events,
            }
        )

    def restore_replay(self, state):
        """Reset with the recorded values and rerun every recorded input."""
        self.replaying = True
        for name, value in state['settings'].items():
            setattr(self, name, deepcopy(value))
        # Every sampled value is given explicitly, so the seed draws nothing
        # that reaches the physics.
        self.reset(seed=0, options=deepcopy(state['reset']))
        for method, value in state['events']:
            assert method in REPLAYED_METHODS, method
            getattr(self, method)(deepcopy(value))
        self.replaying = False
        self.replay_settings = deepcopy(state['settings'])
        self.replay_reset = deepcopy(state['reset'])
        self.replay_events = deepcopy(state['events'])


gym.register(
    id='ngc/PushTReplay-v1',
    entry_point='evaluation.pusht_replay:ReplayPushT',
)
