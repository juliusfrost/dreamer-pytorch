import babyai
from gym_minigrid.minigrid import MiniGridEnv, OBJECT_TO_IDX, COLOR_TO_IDX
from gym_minigrid.wrappers import RGBImgObsWrapper
import gym
import numpy as np
import random
from dreamer.envs.wrapper import EnvWrapper
from rlpyt.spaces.int_box import IntBox
from dreamer.envs.env import StateObs
from dreamer.envs.env import EnvInfo
from rlpyt.envs.base import EnvStep, Env

class Minigrid(Env):
    def __init__(self, level, slipperiness=0.0, one_hot_obs=True):
        self._env = gym.make(level)
        if one_hot_obs:
            self._env = OneHotObs(self._env)
        self.slipperiness = slipperiness
        self.one_hot_obs = one_hot_obs

    def step(self, action):
        step_results = None
        if action == MiniGridEnv.Actions.forward: # Go forward action; no slippage for other actions
            if np.random.uniform() < self.slipperiness:
                action = random.choice(["left", "right"])
                # By default, the agent can only move in a direction if it's facing that way.
                # We can model slippage by turning a direction, moving that way, then turning back
                # and only counting it as a single action.
                self._env.step_count -= 2
                if action == "left":
                    self._env.step(MiniGridEnv.Actions.left)
                    self._env.step(MiniGridEnv.Actions.forward)
                    step_results = self._env.step(MiniGridEnv.Actions.right)
                else:
                    self._env.step(MiniGridEnv.Actions.right)
                    self._env.step(MiniGridEnv.Actions.forward)
                    step_results = self._env.step(MiniGridEnv.Actions.left)
        if step_results is None:
            step_results = self._env.step(action)

        obs, reward, done, info = step_results
        obs = StateObs(obs['image'], np.concatenate([obs['mission'], obs['direction']]))
        info = EnvInfo(None, None, done)
        return EnvStep(obs, reward, done, info)

    def reset(self):
        obs = self._env.reset()
        obs = StateObs(obs['image'], np.concatenate([obs['mission'], obs['direction']]))
        return obs

    @property
    def observation_space(self):
        obs = self._env.observation_space
        mission_shape = obs['mission'].shape[0]
        direction_shape = obs['direction'].shape[0]
        return StateObs(obs['image'], IntBox(0, 1, (mission_shape + direction_shape)))

    @property
    def action_space(self):
        return self._env.action_space

    def render(self):
        image =self._env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=8
        )
        return image

class OneHotObs(EnvWrapper):

    def __init__(self, env):
        super().__init__(env)
        example_obs = self.reset()
        obs_space = {}
        for k, v in example_obs.items():
            obs_space[k] = IntBox(0, 1, shape=v.shape)
        self.obs_space = obs_space

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.one_hot(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self.one_hot(obs)

    def one_hot(self, obs):

        one_hot_obs = {}

        # One-hotify direction
        NUM_DIRECTIONS = 4
        dir = np.zeros(NUM_DIRECTIONS)
        dir[obs["direction"]] = 1
        one_hot_obs["direction"] = dir

        # One-hotify mission
        # TODO: This is tricky!  Missions can be arbitrary lengths, so we could either
        #       (a) One-hot encode each token sequentially, return a variable-length array
        #       (b) Give a fix-sized mission vector where smaller missions are zero-padded
        #       (c) Assume we'll only use levels with a particular mission structure
        #       The code currently does (c).  We assume there's at most one object mentioned.
        #       But at some point we should probably switch to the more generalizable version using the babyai
        #       repo's preprocessor InstructionsPreprocessor in file utils/format.py
        obj_id = np.zeros((len(OBJECT_TO_IDX.keys())))
        colors_id = np.zeros((len(COLOR_TO_IDX.keys()) + 1))
        found_color = False
        for key, index in COLOR_TO_IDX.items():
            if key in obs["mission"]:
                colors_id[index] = 1
                found_color = True
        if not found_color:
            colors_id[-1] = 1
        for key, index in OBJECT_TO_IDX.items():
            if key in obs["mission"]:
                obj_id[index] = 1
        one_hot_obs["mission"] = np.concatenate([obj_id, colors_id], axis=0)

        # One-hotify grid observation
        num_objects = len(OBJECT_TO_IDX.keys())
        num_colors = len(COLOR_TO_IDX.keys())
        num_states = 3 # open, closed or locked.  Looks like they don't have an enum for it.
        # Observation space is [height, width, 3]
        image = obs["image"]
        height, width, _ = image.shape
        height_index = np.repeat(np.arange(height), width).reshape(height, width)
        width_index = np.tile(np.arange(width), height).reshape(height, width)

        # First layer has object IDs
        obj_ids = np.zeros((height, width, num_objects))
        obj_ids[height_index, width_index, image[:,:,0]] = 1

        # Second layer has color IDs
        color_ids = np.zeros((height, width, num_colors))
        color_ids[height_index, width_index, image[:,:,1]] = 1

        # Third layer has state IDs
        state_ids = np.zeros((height, width, num_states))
        state_ids[height_index, width_index, image[:,:,2]] = 1

        image = np.concatenate([obj_ids, color_ids, state_ids], axis=2).transpose((2, 0, 1))

        # Zero-pad observations to the nearest multiple of 2.
        channels, height, width = image.shape
        pow_2_height = int(2 ** np.ceil(np.log2(height)))
        pow_2_width = int(2 ** np.ceil(np.log2(width)))
        pow_2_image = np.zeros((channels, pow_2_height, pow_2_width))
        pow_2_image[:, :height, :width] = image

        one_hot_obs["image"] = pow_2_image

        return one_hot_obs

    @property
    def observation_space(self):
        return self.obs_space

