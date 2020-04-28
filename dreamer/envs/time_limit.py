from rlpyt.envs.base import EnvStep
from dreamer.envs.wrapper import EnvWrapper
from dreamer.envs.env import EnvInfo


class TimeLimit(EnvWrapper):
    def __init__(self, env, duration):
        super().__init__(env)
        self._duration = duration
        self._step = None

    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            # done = True
            # if 'discount' not in info:
            #     info['discount'] = np.array(1.0).astype(np.float32)
            if isinstance(info, EnvInfo):
                # The last attribute in EnvInfo indicates termination of the trajectory
                # we do not set done = True because it should only be controlled by the environment
                info = EnvInfo(info.discount, info.game_score, True)
            self._step = None
        return EnvStep(obs, reward, done, info)

    def reset(self):
        self._step = 0
        return self.env.reset()
