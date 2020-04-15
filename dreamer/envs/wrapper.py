from typing import Sequence, Dict

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


def make_wapper(base_class, wrapper_classes: Sequence = None, wrapper_kwargs: Sequence[Dict] = None):
    """
    Creates the correct factory method with wrapper support.
    This would get passed as the EnvCls argument in the sampler.

    Examples:
    The following code would make a factory method for atari with action repeat 2
    ``factory_method = make(AtariEnv, (ActionRepeat, ), (dict(amount=2),))``

    :param base_class: the base environment class (eg. AtariEnv)
    :param wrapper_classes: list of wrapper classes in order inner-first, outer-last
    :param wrapper_kwargs: list of kwargs dictionaries passed to the wrapper classes
    :return: factory method
    """
    if wrapper_classes is None:
        def make_env(**env_kwargs):
            """:return only the base environment instance"""
            return base_class(**env_kwargs)

        return make_env
    else:
        assert len(wrapper_classes) == len(wrapper_kwargs)

        def make_env(**env_kwargs):
            """:return the wrapped environment instance"""
            env = base_class(**env_kwargs)
            for i, wrapper_cls in enumerate(wrapper_classes):
                w_kwargs = wrapper_kwargs[i]
                if w_kwargs is None:
                    w_kwargs = dict()
                env = wrapper_cls(env, **w_kwargs)
            return env

        return make_env
