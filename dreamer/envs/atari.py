import threading

import numpy as np
from PIL import Image
from rlpyt.envs.atari.atari_env import AtariTrajInfo
from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox

from dreamer.envs.env import EnvInfo

AtariTrajInfo = AtariTrajInfo


class AtariEnv(Env):
    LOCK = threading.Lock()

    def __init__(
            self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
            life_done=False, sticky_actions=True):
        import gym
        version = 0 if sticky_actions else 4
        name = ''.join(word.title() for word in name.split('_'))
        with self.LOCK:
            self._env = gym.make('{}NoFrameskip-v{}'.format(name, version))
        self._action_repeat = action_repeat
        self._size = size
        self._grayscale = grayscale
        self._noops = noops
        self._life_done = life_done
        self._lives = None
        shape = self._env.observation_space.shape[:2] + (() if grayscale else (3,))
        self._buffers = [np.empty(shape, dtype=np.uint8) for _ in range(2)]
        self.random = np.random.RandomState(seed=None)  # expose for one_hot wrapper

    @property
    def observation_space(self):
        shape = (1 if self._grayscale else 3,) + self._size
        space = IntBox(low=0, high=255, shape=shape, dtype=np.uint8)
        return space

    @property
    def action_space(self):
        return self._env.action_space

    def close(self):
        return self._env.close()

    def reset(self):
        with self.LOCK:
            self._env.reset()
        noops = self.random.randint(1, self._noops + 1)
        for _ in range(noops):
            done = self._env.step(0)[2]
            if done:
                with self.LOCK:
                    self._env.reset()
        self._lives = self._env.ale.lives()
        if self._grayscale:
            self._env.ale.getScreenGrayscale(self._buffers[0])
        else:
            self._env.ale.getScreenRGB2(self._buffers[0])
        self._buffers[1].fill(0)
        self._step_counter = 0
        return self._get_obs()

    def step(self, action):
        total_reward = 0.0
        for step in range(self._action_repeat):
            _, reward, done, info = self._env.step(action)
            total_reward += reward
            if self._life_done:
                lives = self._env.ale.lives()
                done = done or lives < self._lives
                self._lives = lives
            if done:
                break
            elif step >= self._action_repeat - 2:
                index = step - (self._action_repeat - 2)
                if self._grayscale:
                    self._env.ale.getScreenGrayscale(self._buffers[index])
                else:
                    self._env.ale.getScreenRGB2(self._buffers[index])
        obs = self._get_obs()
        env_info = EnvInfo(None, total_reward, done)
        return EnvStep(obs, total_reward, done, env_info)

    def render(self, mode):
        return self._env.render(mode)

    def _get_obs(self):
        if self._action_repeat > 1:
            np.maximum(self._buffers[0], self._buffers[1], out=self._buffers[0])
        image = np.array(Image.fromarray(self._buffers[0]).resize(
            self._size, Image.BILINEAR))
        image = np.clip(image, 0, 255).astype(np.uint8)
        image = image[:, :, None] if self._grayscale else image
        image = np.transpose(image, (2, 0, 1))
        return image

    @property
    def horizon(self):
        raise NotImplementedError
