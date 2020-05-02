import babyai
import pickle
from gym_minigrid.minigrid import MiniGridEnv, OBJECT_TO_IDX, COLOR_TO_IDX
import gym
import numpy as np
import random
from dreamer.envs.wrapper import EnvWrapper
from dreamer.envs.one_hot import OneHotAction
from gym.spaces import Box

class SlipperinessWrapper(EnvWrapper):
    def __init__(self, env, slip_prob):
        super().__init__(env)
        self.slip_prob = slip_prob

    def step(self, action):
        if action == MiniGridEnv.Actions.forward: # Go forward action; no slippage for other actions
            if np.random.uniform() < self.slip_prob:
                action = random.choice(["left", "right"])
                # By default, the agent can only move in a direction if it's facing that way.
                # We can model slippage by turning a direction, moving that way, then turning back
                # and only counting it as a single action.
                self.step_count -= 2
                if action == "left":
                    super().step(MiniGridEnv.Actions.left)
                    super().step(MiniGridEnv.Actions.forward)
                    obs = super().step(MiniGridEnv.Actions.right)
                else:
                    super().step(MiniGridEnv.Actions.right)
                    super().step(MiniGridEnv.Actions.forward)
                    obs = super().step(MiniGridEnv.Actions.left)
                return obs
        return super().step(action)


class OneHotObs(EnvWrapper):

    def __init__(self, env):
        super().__init__(env)
        example_obs = self.reset()
        obs_space = {}
        for k, v in example_obs.items():
            v_shape = v * 0
            obs_space[k] = Box(v_shape, v_shape + 1, dtype=np.int32)
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

        image = np.concatenate([obj_ids, color_ids, state_ids], axis=2)
        one_hot_obs["image"] = image
        return one_hot_obs

    @property
    def observation_space(self):
        return self.obs_space

def save_env(env, save_location):
    # TODO: we could save just the important info, but might as well save the entire object
    pickle.dump(env, open(save_location, "wb"))

def load_env(save_location):
    return pickle.load(open(save_location, "rb"))


def make_env(level=None, slipperiness=0.0, one_hot_obs=True, one_hot_actions=True):
    env = gym.make(level)
    env = SlipperinessWrapper(env, slipperiness)
    if one_hot_actions:
        env = OneHotAction(env)
    if one_hot_obs:
        env = OneHotObs(env)
    return env
