import gym
import numpy as np
from dm_control import suite
from rlpyt.envs.base import Env, EnvStep
from rlpyt.utils.collections import namedarraytuple
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.float_box import FloatBox

from dreamer.envs.env import EnvInfo


# DMCInfo = namedarraytuple("DMCInfo", ["discount", 'traj_done'])


class DeepMindControl(Env):

    def __init__(self, name, size=(64, 64), camera=None):
        domain, task = name.split('_', 1)
        if domain == 'cup':  # Only domain with multiple words.
            domain = 'ball_in_cup'
        if isinstance(domain, str):
            self._env = suite.load(domain, task)
        else:
            assert task is None
            self._env = domain()
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera

    @property
    def observation_space(self):
        return IntBox(low=0, high=255, shape=(3,) + self._size,
                      dtype="uint8")

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return FloatBox(low=spec.minimum, high=spec.maximum)

    def step(self, action):
        time_step = self._env.step(action)
        _ = dict(time_step.observation)
        obs = self.render()
        reward = time_step.reward or 0
        done = time_step.last()

        info = EnvInfo(np.array(time_step.discount, np.float32), None, done)
        return EnvStep(obs, reward, done, info)

    def reset(self):
        time_step = self._env.reset()
        _ = dict(time_step.observation)
        obs = self.render()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera).transpose(2, 0, 1).copy()

    @property
    def horizon(self):
        raise NotImplementedError
