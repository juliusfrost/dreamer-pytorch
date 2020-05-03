import babyai
import pickle
from dreamer.envs.minigrid import Minigrid
from dreamer.envs.one_hot import OneHotAction

def save_env(env, save_location):
    # TODO: we could save just the important info, but might as well save the entire object
    pickle.dump(env, open(save_location, "wb"))

def load_env(save_location):
    return pickle.load(open(save_location, "rb"))


def make_env(level=None, slipperiness=0.0, one_hot_obs=True, one_hot_actions=True):
    env = Minigrid(level, slipperiness, one_hot_obs)
    if one_hot_actions:
        env = OneHotAction(env)
    return env
