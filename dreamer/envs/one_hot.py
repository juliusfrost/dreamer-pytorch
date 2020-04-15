import gym
import numpy as np

from dreamer.envs.wrapper import EnvWrapper
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.float_box import FloatBox


class OneHotAction(EnvWrapper):

    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete) or isinstance(env.action_space, IntBox)
        super().__init__(env)
        self._dtype = np.float32

    @property
    def action_space(self):
        shape = (self.env.action_space.n,)
        space = FloatBox(low=0, high=1, shape=shape, dtype=self._dtype)
        space.sample = self._sample_action
        return space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action, atol=1e6):
            raise ValueError(f'Invalid one-hot action:\n{action}')
        return self.env.step(index)

    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.n
        index = self.random.randint(0, actions)
        reference = np.zeros(actions, dtype=self._dtype)
        reference[index] = 1.0
        return reference
