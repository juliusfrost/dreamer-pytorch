from rlpyt.envs.base import Env


class EnvWrapper(Env):

    def __init__(self, env: Env):
        self.env = env

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def horizon(self):
        return self.env.horizon

    def close(self):
        self.env.close()
