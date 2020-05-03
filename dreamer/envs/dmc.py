import gym
import numpy as np
from dm_control import suite
from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.float_box import FloatBox
from dreamer.envs.env import StateObs

from dreamer.envs.env import EnvInfo


class DeepMindControl(Env):

    def __init__(self, name, size=(64, 64), camera=None, use_state=None):
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
        self.use_state = use_state

    @property
    def observation_space(self):
        img_obs = IntBox(low=0, high=255, shape=(3,) + self._size,
                      dtype="uint8")
        if self.use_state:
            state_obs = self._env.observation_spec()
            total_shape = np.sum([v.shape[0] for k, v in state_obs.items()])
            state_obs_box = FloatBox(low=-float('inf'), high=float('inf'), shape=total_shape)
            return StateObs(img_obs, state_obs_box)
        else:
            return img_obs


    @property
    def action_space(self):
        spec = self._env.action_spec()
        return FloatBox(low=spec.minimum, high=spec.maximum)

    def step(self, action):
        time_step = self._env.step(action)
        obs = dict(time_step.observation)
        state_obs = np.concatenate([value for key, value in obs.items()])
        img_obs = self.render()
        reward = time_step.reward or 0
        done = time_step.last()

        info = EnvInfo(np.array(time_step.discount, np.float32), None, done)
        obs = StateObs(img_obs, state_obs) if self.use_state else img_obs
        return EnvStep(obs, reward, done, info)

    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        state_obs = np.concatenate([value for key, value in obs.items()])
        img_obs = self.render()
        if self.use_state:
            return StateObs(img_obs, state_obs)
        return img_obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera).transpose(2, 0, 1).copy()

    @property
    def horizon(self):
        raise NotImplementedError
