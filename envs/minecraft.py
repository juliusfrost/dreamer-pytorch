"""
initial minecraft environment code.
largely taken from https://github.com/minerllabs/baselines/blob/master/general/chainerrl/baselines/ppo.py
"""

import os
import copy
import time
from collections import OrderedDict, deque
from logging import getLogger

import cv2
import gym
import numpy as np
from gym import spaces
from gym.wrappers import Monitor
from gym.wrappers.monitoring.stats_recorder import StatsRecorder
from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.float_box import FloatBox
from rlpyt.spaces.int_box import IntBox

from envs.env import EnvInfo

cv2.ocl.setUseOpenCL(False)
logger = getLogger(__name__)

ENVIRONMENTS = [
    'MineRLTreechop-v0',
    'MineRLNavigate-v0', 'MineRLNavigateDense-v0',
    'MineRLNavigateExtreme-v0', 'MineRLNavigateExtremeDense-v0',
    'MineRLObtainIronPickaxe-v0', 'MineRLObtainIronPickaxeDense-v0',
    'MineRLObtainDiamond-v0', 'MineRLObtainDiamondDense-v0',
]

os.environ['MALMO_MINECRAFT_OUTPUT_LOGDIR'] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../data/malmo'))


def wrap_env(env, env_name, test=False, monitor=False, frame_skip=None, gray_scale=False, frame_stack=None,
             disable_action_prior=False, always_keys=None, reverse_keys=None, exclude_keys=None, exclude_noop=False):
    # wrap env: time limit...
    if isinstance(env, gym.wrappers.TimeLimit):
        logger.info('Detected `gym.wrappers.TimeLimit`! Unwrap it and re-wrap our own time limit.')
        env = env.env
        max_episode_steps = env.spec.max_episode_steps
        env = ContinuingTimeLimit(env, max_episode_steps=max_episode_steps)

    # wrap env: observation...
    # NOTE: wrapping order matters!

    if test and monitor:
        raise NotImplementedError
        # env = ContinuingTimeLimitMonitor(
        #     env, os.path.join(args.outdir, 'monitor'),
        #     mode='evaluation' if test else 'training', video_callable=lambda episode_id: True)
    if frame_skip is not None:
        env = FrameSkip(env, skip=frame_skip)
    if gray_scale:
        env = GrayScaleWrapper(env, dict_space_key='pov')
    if env_name.startswith('MineRLNavigate'):
        env = PoVWithCompassAngleWrapper(env)
    else:
        env = ObtainPoVWrapper(env)
    env = MoveAxisWrapper(env, source=-1, destination=0)  # convert hwc -> chw as Chainer requires.
    env = ScaledFloatFrame(env)
    if frame_stack is not None and frame_stack > 0:
        env = FrameStack(env, frame_stack, channel_order='chw')

    # wrap env: action...
    if not disable_action_prior:
        env = SerialDiscreteActionWrapper(
            env,
            always_keys=always_keys, reverse_keys=reverse_keys, exclude_keys=exclude_keys,
            exclude_noop=exclude_noop)
    else:
        env = CombineActionWrapper(env)
        env = SerialDiscreteCombineActionWrapper(env)

    return env


class MineRL(Env):

    def __init__(
            self,
            name,
            frame_skip=None,
            gray_scale=False,
            frame_stack=None,
            disable_action_prior=False,
            always_keys=None,
            reverse_keys=None,
            exclude_keys=None,
            exclude_noop=False,
            seed=0,
    ):
        self.name = name
        env = gym.make(name)
        env = wrap_env(env, name, frame_skip=frame_skip, gray_scale=gray_scale, frame_stack=frame_stack,
                       disable_action_prior=disable_action_prior, always_keys=always_keys, reverse_keys=reverse_keys,
                       exclude_keys=exclude_keys, exclude_noop=exclude_noop)
        self.env = env
        self._action_space = IntBox(0, env.action_space.n)
        self._observation_space = FloatBox(env.observation_space.low, env.observation_space.high)
        self.random = np.random.RandomState(seed)

    def seed(self, seed):
        self.random = np.random.RandomState(seed)

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        info = EnvInfo(0, 0, done)
        return EnvStep(obs, reward, done, info)

    def reset(self):
        return self.env.reset()

    @property
    def horizon(self):
        raise NotImplementedError


class ContinuingTimeLimit(gym.Wrapper):
    """TimeLimit wrapper for continuing environments.
    This is similar gym.wrappers.TimeLimit, which sets a time limit for
    each episode, except that done=False is returned and that
    info['needs_reset'] is set to True when past the limit.
    Code that calls env.step is responsible for checking the info dict, the
    fourth returned value, and resetting the env if it has the 'needs_reset'
    key and its value is True.
    Args:
        env (gym.Env): Env to wrap.
        max_episode_steps (int): Maximum number of timesteps during an episode,
            after which the env needs a reset.
    """

    def __init__(self, env, max_episode_steps):
        super(ContinuingTimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps

        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, \
            "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._max_episode_steps <= self._elapsed_steps:
            info['needs_reset'] = True

        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()


class ResetTrimInfoWrapper(gym.Wrapper):
    """Take first return value.
    minerl's `env.reset()` returns tuple of `(obs, info)`
    but existing agent implementations expect `reset()` returns `obs` only.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs


class ContinuingTimeLimitMonitor(Monitor):
    """`Monitor` with ChainerRL's `ContinuingTimeLimit` support.
    Because of the original implementation's design,
    explicit `close()` is needed to save the last episode.
    Do not forget to call `close()` at the last line of your script.
    For details, see
    https://github.com/openai/gym/blob/master/gym/wrappers/monitor.py
    """

    def _start(self, directory, video_callable=None, force=False, resume=False,
               write_upon_reset=False, uid=None, mode=None):
        if self.env_semantics_autoreset:
            raise gym.error.Error(
                "Detect 'semantics.autoreset=True' in `env.metadata`, "
                "which means the env comes from deprecated OpenAI Universe.")
        ret = super()._start(directory=directory,
                             video_callable=video_callable, force=force,
                             resume=resume, write_upon_reset=write_upon_reset,
                             uid=uid, mode=mode)
        if self.env.spec is None:
            env_id = '(unknown)'
        else:
            env_id = self.env.spec.id
        self.stats_recorder = _ContinuingTimeLimitStatsRecorder(
            directory,
            '{}.episode_batch.{}'.format(self.file_prefix, self.file_infix),
            autoreset=False, env_id=env_id)
        return ret


class _ContinuingTimeLimitStatsRecorder(StatsRecorder):
    """`StatsRecorder` with ChainerRL's `ContinuingTimeLimit` support.
    For details, see
    https://github.com/openai/gym/blob/master/gym/wrappers/monitoring/stats_recorder.py
    """

    def __init__(self, directory, file_prefix, autoreset=False, env_id=None):
        super().__init__(directory, file_prefix,
                         autoreset=autoreset, env_id=env_id)
        self._save_completed = True

    def before_reset(self):
        assert not self.closed

        if self.done is not None and not self.done and self.steps > 0:
            logger.debug('Tried to reset env which is not done. '
                         'StatsRecorder completes the last episode.')
            self.save_complete()

        self.done = False
        if self.initial_reset_timestamp is None:
            self.initial_reset_timestamp = time.time()

    def after_step(self, observation, reward, done, info):
        self._save_completed = False
        return super().after_step(observation, reward, done, info)

    def save_complete(self):
        if not self._save_completed:
            super().save_complete()
            self._save_completed = True

    def close(self):
        self.save_complete()
        super().close()


class FrameSkip(gym.Wrapper):
    """Return every `skip`-th frame and repeat given action during skip.
    Note that this wrapper does not "maximize" over the skipped frames.
    """

    def __init__(self, env, skip=4):
        super().__init__(env)

        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class ObtainPoVWrapper(gym.ObservationWrapper):
    """Obtain 'pov' value (current game display) of the original observation."""

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = self.env.observation_space.spaces['pov']

    def observation(self, observation):
        return observation['pov']


class PoVWithCompassAngleWrapper(gym.ObservationWrapper):
    """Take 'pov' value (current game display) and concatenate compass angle information with it, as a new channel of image;
    resulting image has RGB+compass (or K+compass for gray-scaled image) channels.
    """

    def __init__(self, env):
        super().__init__(env)

        self._compass_angle_scale = 180 / 255  # NOTE: `ScaledFloatFrame` will scale the pixel values with 255.0 later

        pov_space = self.env.observation_space.spaces['pov']
        compass_angle_space = self.env.observation_space.spaces['compassAngle']

        low = self.observation({'pov': pov_space.low, 'compassAngle': compass_angle_space.low})
        high = self.observation({'pov': pov_space.high, 'compassAngle': compass_angle_space.high})

        self.observation_space = gym.spaces.Box(low=low, high=high)

    def observation(self, observation):
        pov = observation['pov']
        compass_scaled = observation['compassAngle'] / self._compass_angle_scale
        compass_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=pov.dtype) * compass_scaled
        return np.concatenate([pov, compass_channel], axis=-1)


class UnifiedObservationWrapper(gym.ObservationWrapper):
    """Take 'pov', 'compassAngle', 'inventory' and concatenate with scaling.
    Each element of 'inventory' is converted to a square whose side length is region_size.
    The color of each square is correlated to the reciprocal of (the number of the corresponding item + 1).
    """

    def __init__(self, env, region_size=8):
        super().__init__(env)

        self._compass_angle_scale = 180 / 255  # NOTE: `ScaledFloatFrame` will scale the pixel values with 255.0 later
        self.region_size = region_size

        pov_space = self.env.observation_space.spaces['pov']
        low_dict = {'pov': pov_space.low}
        high_dict = {'pov': pov_space.high}

        if 'compassAngle' in self.env.observation_space.spaces:
            compass_angle_space = self.env.observation_space.spaces['compassAngle']
            low_dict['compassAngle'] = compass_angle_space.low
            high_dict['compassAngle'] = compass_angle_space.high

        if 'inventory' in self.env.observation_space.spaces:
            inventory_space = self.env.observation_space.spaces['inventory']
            low_dict['inventory'] = {}
            high_dict['inventory'] = {}
            for key in inventory_space.spaces.keys():
                low_dict['inventory'][key] = inventory_space.spaces[key].low
                high_dict['inventory'][key] = inventory_space.spaces[key].high

        low = self.observation(low_dict)
        high = self.observation(high_dict)

        self.observation_space = gym.spaces.Box(low=low, high=high)

    def observation(self, observation):
        obs = observation['pov']
        pov_dtype = obs.dtype

        if 'compassAngle' in observation:
            compass_scaled = observation['compassAngle'] / self._compass_angle_scale
            compass_channel = np.ones(shape=list(obs.shape[:-1]) + [1], dtype=pov_dtype) * compass_scaled
            obs = np.concatenate([obs, compass_channel], axis=-1)
        if 'inventory' in observation:
            assert len(obs.shape[:-1]) == 2
            region_max_height = obs.shape[0]
            region_max_width = obs.shape[1]
            rs = self.region_size
            if min(region_max_height, region_max_width) < rs:
                raise ValueError("'region_size' is too large.")
            num_element_width = region_max_width // rs
            inventory_channel = np.zeros(shape=list(obs.shape[:-1]) + [1], dtype=pov_dtype)
            for idx, key in enumerate(observation['inventory']):
                item_scaled = np.clip(255 - 255 / (observation['inventory'][key] + 1),  # Inversed
                                      0, 255)
                item_channel = np.ones(shape=[rs, rs, 1], dtype=pov_dtype) * item_scaled
                width_low = (idx % num_element_width) * rs
                height_low = (idx // num_element_width) * rs
                if height_low + rs > region_max_height:
                    raise ValueError(
                        "Too many elements on 'inventory'. Please decrease 'region_size' of each component")
                inventory_channel[height_low:(height_low + rs), width_low:(width_low + rs), :] = item_channel
            obs = np.concatenate([obs, inventory_channel], axis=-1)
        return obs


class FullObservationSpaceWrapper(gym.ObservationWrapper):
    """Returns as observation a tuple with the frames and a list of
    compassAngle and inventory items.
    compassAngle is scaled to be in the interval [-1, 1] and inventory items
    are scaled to be in the interval [0, 1]
    """

    def __init__(self, env):
        super().__init__(env)

        pov_space = self.env.observation_space.spaces['pov']

        low_dict = {'pov': pov_space.low, 'inventory': {}}
        high_dict = {'pov': pov_space.high, 'inventory': {}}

        for obs_name in self.env.observation_space.spaces['inventory'].spaces.keys():
            obs_space = self.env.observation_space.spaces['inventory'].spaces[obs_name]
            low_dict['inventory'][obs_name] = obs_space.low
            high_dict['inventory'][obs_name] = obs_space.high

        if 'compassAngle' in self.env.observation_space.spaces:
            compass_angle_space = self.env.observation_space.spaces['compassAngle']
            low_dict['compassAngle'] = compass_angle_space.low
            high_dict['compassAngle'] = compass_angle_space.high

        low = self.observation(low_dict)
        high = self.observation(high_dict)

        pov_space = gym.spaces.Box(low=low[0], high=high[0])
        inventory_space = gym.spaces.Box(low=low[1], high=high[1])
        self.observation_space = gym.spaces.Tuple((pov_space, inventory_space))

    def observation(self, observation):
        frame = observation['pov']
        inventory = []

        if 'compassAngle' in observation:
            compass_scaled = observation['compassAngle'] / 180
            inventory.append(compass_scaled)

        for obs_name in observation['inventory'].keys():
            inventory.append(observation['inventory'][obs_name] / 2304)

        inventory = np.array(inventory)
        return (frame, inventory)


class MoveAxisWrapper(gym.ObservationWrapper):
    """Move axes of observation ndarrays."""

    def __init__(self, env, source, destination, use_tuple=False):
        if use_tuple:
            assert isinstance(env.observation_space[0], gym.spaces.Box)
        else:
            assert isinstance(env.observation_space, gym.spaces.Box)
        super().__init__(env)

        self.source = source
        self.destination = destination
        self.use_tuple = use_tuple

        if self.use_tuple:
            low = self.observation(
                tuple([space.low for space in self.observation_space]))
            high = self.observation(
                tuple([space.high for space in self.observation_space]))
            dtype = self.observation_space[0].dtype
            pov_space = gym.spaces.Box(low=low[0], high=high[0], dtype=dtype)
            inventory_space = self.observation_space[1]
            self.observation_space = gym.spaces.Tuple(
                (pov_space, inventory_space))
        else:
            low = self.observation(self.observation_space.low)
            high = self.observation(self.observation_space.high)
            dtype = self.observation_space.dtype
            self.observation_space = gym.spaces.Box(
                low=low, high=high, dtype=dtype)

    def observation(self, observation):
        if self.use_tuple:
            new_observation = list(observation)
            new_observation[0] = np.moveaxis(
                observation[0], self.source, self.destination)
            return tuple(new_observation)
        else:
            return np.moveaxis(observation, self.source, self.destination)


class GrayScaleWrapper(gym.ObservationWrapper):
    def __init__(self, env, dict_space_key=None):
        super().__init__(env)

        self._key = dict_space_key

        if self._key is None:
            original_space = self.observation_space
        else:
            original_space = self.observation_space.spaces[self._key]
        height, width = original_space.shape[0], original_space.shape[1]

        # sanity checks
        ideal_image_space = gym.spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)
        if original_space != ideal_image_space:
            raise ValueError('Image space should be {}, but given {}.'.format(ideal_image_space, original_space))
        if original_space.dtype != np.uint8:
            raise ValueError('Image should `np.uint8` typed, but given {}.'.format(original_space.dtype))

        height, width = original_space.shape[0], original_space.shape[1]
        new_space = gym.spaces.Box(low=0, high=255, shape=(height, width, 1), dtype=np.uint8)
        if self._key is None:
            self.observation_space = new_space
        else:
            new_space_dict = copy.deepcopy(self.observation_space)
            new_space_dict.spaces[self._key] = new_space
            self.observation_space = new_space_dict

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = np.expand_dims(frame, -1)
        if self._key is None:
            obs = frame
        else:
            obs[self._key] = frame
        return obs


class SerialDiscreteActionWrapper(gym.ActionWrapper):
    """Convert MineRL env's `Dict` action space as a serial discrete action space.
    The term "serial" means that this wrapper can only push one key at each step.
    "attack" action will be alwarys triggered.
    Parameters
    ----------
    env
        Wrapping gym environment.
    always_keys
        List of action keys, which should be always pressed throughout interaction with environment.
        If specified, the "noop" action is also affected.
    reverse_keys
        List of action keys, which should be always pressed but can be turn off via action.
        If specified, the "noop" action is also affected.
    exclude_keys
        List of action keys, which should be ignored for discretizing action space.
    exclude_noop
        The "noop" will be excluded from discrete action list.
    num_camera_discretize
        Number of discretization of yaw control (must be odd).
    allow_pitch
        If specified, this wrapper appends commands to control pitch.
    max_camera_range
        Maximum value of yaw control.
    """

    BINARY_KEYS = ['forward', 'back', 'left', 'right', 'jump', 'sneak', 'sprint', 'attack']

    def __init__(self, env, always_keys=None, reverse_keys=None, exclude_keys=None, exclude_noop=False,
                 num_camera_discretize=3, allow_pitch=False,
                 max_camera_range=10):
        super().__init__(env)

        self.always_keys = [] if always_keys is None else always_keys
        self.reverse_keys = [] if reverse_keys is None else reverse_keys
        self.exclude_keys = [] if exclude_keys is None else exclude_keys
        if len(set(self.always_keys) | set(self.reverse_keys) | set(self.exclude_keys)) != \
                len(self.always_keys) + len(self.reverse_keys) + len(self.exclude_keys):
            raise ValueError('always_keys ({}) or reverse_keys ({}) or exclude_keys ({}) intersect each other.'.format(
                self.always_keys, self.reverse_keys, self.exclude_keys))
        self.exclude_noop = exclude_noop

        self.wrapping_action_space = self.env.action_space
        self.num_camera_discretize = num_camera_discretize
        self._noop_template = OrderedDict([
            ('forward', 0),
            ('back', 0),
            ('left', 0),
            ('right', 0),
            ('jump', 0),
            ('sneak', 0),
            ('sprint', 0),
            ('attack', 0),
            ('camera', np.zeros((2,), dtype=np.float32)),
            # 'none', 'dirt' (Obtain*:)+ 'stone', 'cobblestone', 'crafting_table', 'furnace', 'torch'
            ('place', 0),
            # (Obtain* tasks only) 'none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe'
            ('equip', 0),
            # (Obtain* tasks only) 'none', 'torch', 'stick', 'planks', 'crafting_table'
            ('craft', 0),
            # (Obtain* tasks only) 'none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe', 'furnace'
            ('nearbyCraft', 0),
            # (Obtain* tasks only) 'none', 'iron_ingot', 'coal'
            ('nearbySmelt', 0),
        ])
        for key, space in self.wrapping_action_space.spaces.items():
            if key not in self._noop_template:
                raise ValueError('Unknown action name: {}'.format(key))

        # get noop
        self.noop = copy.deepcopy(self._noop_template)
        for key in self._noop_template:
            if key not in self.wrapping_action_space.spaces:
                del self.noop[key]

        # check&set always_keys
        for key in self.always_keys:
            if key not in self.BINARY_KEYS:
                raise ValueError('{} is not allowed for `always_keys`.'.format(key))
            self.noop[key] = 1
        logger.info('always pressing keys: {}'.format(self.always_keys))
        # check&set reverse_keys
        for key in self.reverse_keys:
            if key not in self.BINARY_KEYS:
                raise ValueError('{} is not allowed for `reverse_keys`.'.format(key))
            self.noop[key] = 1
        logger.info('reversed pressing keys: {}'.format(self.reverse_keys))
        # check exclude_keys
        for key in self.exclude_keys:
            if key not in self.noop:
                raise ValueError('unknown exclude_keys: {}'.format(key))
        logger.info('always ignored keys: {}'.format(self.exclude_keys))

        # get each discrete action
        self._actions = [self.noop]
        for key in self.noop:
            if key in self.always_keys or key in self.exclude_keys:
                continue
            if key in self.BINARY_KEYS:
                # action candidate : {1}  (0 is ignored because it is for noop), or {0} when `reverse_keys`.
                op = copy.deepcopy(self.noop)
                if key in self.reverse_keys:
                    op[key] = 0
                else:
                    op[key] = 1
                self._actions.append(op)
            elif key == 'camera':
                # action candidate : {[0, -max_camera_range], [0, -max_camera_range + delta_range], ..., [0, max_camera_range]}
                # ([0, 0] is excluded)
                delta_range = max_camera_range * 2 / (self.num_camera_discretize - 1)
                if self.num_camera_discretize % 2 == 0:
                    raise ValueError('Number of camera discretization must be odd.')
                for i in range(self.num_camera_discretize):
                    op = copy.deepcopy(self.noop)
                    if i < self.num_camera_discretize // 2:
                        op[key] = np.array([0, -max_camera_range + delta_range * i], dtype=np.float32)
                    elif i > self.num_camera_discretize // 2:
                        op[key] = np.array([0, -max_camera_range + delta_range * (i - 1)], dtype=np.float32)
                    else:
                        continue
                    self._actions.append(op)

                if allow_pitch:
                    for i in range(self.num_camera_discretize):
                        op = copy.deepcopy(self.noop)
                        if i < self.num_camera_discretize // 2:
                            op[key] = np.array([-max_camera_range + delta_range * i, 0], dtype=np.float32)
                        elif i > self.num_camera_discretize // 2:
                            op[key] = np.array([-max_camera_range + delta_range * (i - 1), 0], dtype=np.float32)
                        else:
                            continue
                        self._actions.append(op)

            elif key in {'place', 'equip', 'craft', 'nearbyCraft', 'nearbySmelt'}:
                # action candidate : {1, 2, ..., len(space)-1}  (0 is ignored because it is for noop)
                for a in range(1, self.wrapping_action_space.spaces[key].n):
                    op = copy.deepcopy(self.noop)
                    op[key] = a
                    self._actions.append(op)
        if self.exclude_noop:
            del self._actions[0]

        n = len(self._actions)
        self.action_space = gym.spaces.Discrete(n)
        logger.info('{} is converted to {}.'.format(self.wrapping_action_space, self.action_space))

    def action(self, action):
        if not self.action_space.contains(action):
            raise ValueError('action {} is invalid for {}'.format(action, self.action_space))

        original_space_action = self._actions[action]
        logger.debug('discrete action {} -> original action {}'.format(action, original_space_action))
        return original_space_action


class CombineActionWrapper(gym.ActionWrapper):
    """Combine MineRL env's "exclusive" actions.
    "exclusive" actions will be combined as:
        - "forward", "back" -> noop/forward/back (Discrete(3))
        - "left", "right" -> noop/left/right (Discrete(3))
        - "sneak", "sprint" -> noop/sneak/sprint (Discrete(3))
        - "attack", "place", "equip", "craft", "nearbyCraft", "nearbySmelt"
            -> noop/attack/place/equip/craft/nearbyCraft/nearbySmelt (Discrete(n))
    The combined action's names will be concatenation of originals, i.e.,
    "forward_back", "left_right", "snaek_sprint", "attack_place_equip_craft_nearbyCraft_nearbySmelt".
    """

    def __init__(self, env):
        super().__init__(env)

        self.wrapping_action_space = self.env.action_space

        def combine_exclusive_actions(keys):
            """
            Dict({'forward': Discrete(2), 'back': Discrete(2)})
            =>
            new_actions: [{'forward':0, 'back':0}, {'forward':1, 'back':0}, {'forward':0, 'back':1}]
            """
            new_key = '_'.join(keys)
            valid_action_keys = [k for k in keys if k in self.wrapping_action_space.spaces]
            noop = {a: 0 for a in valid_action_keys}
            new_actions = [noop]

            for key in valid_action_keys:
                space = self.wrapping_action_space.spaces[key]
                for i in range(1, space.n):
                    op = copy.deepcopy(noop)
                    op[key] = i
                    new_actions.append(op)
            return new_key, new_actions

        self._maps = {}
        for keys in (
                ('forward', 'back'), ('left', 'right'), ('sneak', 'sprint'),
                ('attack', 'place', 'equip', 'craft', 'nearbyCraft', 'nearbySmelt')):
            new_key, new_actions = combine_exclusive_actions(keys)
            self._maps[new_key] = new_actions

        self.noop = OrderedDict([
            ('forward_back', 0),
            ('left_right', 0),
            ('jump', 0),
            ('sneak_sprint', 0),
            ('camera', np.zeros((2,), dtype=np.float32)),
            ('attack_place_equip_craft_nearbyCraft_nearbySmelt', 0),
        ])

        self.action_space = gym.spaces.Dict({
            'forward_back':
                gym.spaces.Discrete(len(self._maps['forward_back'])),
            'left_right':
                gym.spaces.Discrete(len(self._maps['left_right'])),
            'jump':
                self.wrapping_action_space.spaces['jump'],
            'sneak_sprint':
                gym.spaces.Discrete(len(self._maps['sneak_sprint'])),
            'camera':
                self.wrapping_action_space.spaces['camera'],
            'attack_place_equip_craft_nearbyCraft_nearbySmelt':
                gym.spaces.Discrete(len(self._maps['attack_place_equip_craft_nearbyCraft_nearbySmelt']))
        })

        logger.info('{} is converted to {}.'.format(self.wrapping_action_space, self.action_space))
        for k, v in self._maps.items():
            logger.info('{} -> {}'.format(k, v))

    def action(self, action):
        if not self.action_space.contains(action):
            raise ValueError('action {} is invalid for {}'.format(action, self.action_space))

        original_space_action = OrderedDict()
        for k, v in action.items():
            if k in self._maps:
                a = self._maps[k][v]
                original_space_action.update(a)
            else:
                original_space_action[k] = v

        logger.debug('action {} -> original action {}'.format(action, original_space_action))
        return original_space_action


class SerialDiscreteCombineActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.wrapping_action_space = self.env.action_space

        self.noop = OrderedDict([
            ('forward_back', 0),
            ('left_right', 0),
            ('jump', 0),
            ('sneak_sprint', 0),
            ('camera', np.zeros((2,), dtype=np.float32)),
            ('attack_place_equip_craft_nearbyCraft_nearbySmelt', 0),
        ])

        # get each discrete action
        self._actions = [self.noop]
        for key in self.noop:
            if key == 'camera':
                # action candidate : {[0, -10], [0, 10]}
                op = copy.deepcopy(self.noop)
                op[key] = np.array([0, -10], dtype=np.float32)
                self._actions.append(op)
                op = copy.deepcopy(self.noop)
                op[key] = np.array([0, 10], dtype=np.float32)
                self._actions.append(op)
            else:
                for a in range(1, self.wrapping_action_space.spaces[key].n):
                    op = copy.deepcopy(self.noop)
                    op[key] = a
                    self._actions.append(op)

        n = len(self._actions)
        self.action_space = gym.spaces.Discrete(n)
        logger.info('{} is converted to {}.'.format(self.wrapping_action_space, self.action_space))

    def action(self, action):
        if not self.action_space.contains(action):
            raise ValueError('action {} is invalid for {}'.format(action, self.action_space))

        original_space_action = self._actions[action]
        logger.debug('discrete action {} -> original action {}'.format(action, original_space_action))
        return original_space_action


class NormalizedContinuousActionWrapper(gym.ActionWrapper):
    """Convert MineRL env's `Dict` action space as a continuous action space.
    Parameters
    ----------
    env
        Wrapping gym environment.
    """

    BINARY_KEYS = ['forward', 'back', 'left', 'right', 'jump', 'sneak', 'sprint', 'attack']

    def __init__(self, env, allow_pitch=False, max_camera_range=10):
        super().__init__(env)
        self.allow_pitch = allow_pitch
        self.wrapping_action_space = self.env.action_space
        self._noop_template = OrderedDict([
            ('forward', 0),
            ('back', 0),
            ('left', 0),
            ('right', 0),
            ('jump', 0),
            ('sneak', 0),
            ('sprint', 0),
            ('attack', 0),
            ('camera', np.zeros((2,), dtype=np.float32)),
            # 'none', 'dirt' (Obtain*:)+ 'stone', 'cobblestone', 'crafting_table', 'furnace', 'torch'
            ('place', 0),
            # (Obtain* tasks only) 'none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe'
            ('equip', 0),
            # (Obtain* tasks only) 'none', 'torch', 'stick', 'planks', 'crafting_table'
            ('craft', 0),
            # (Obtain* tasks only) 'none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe', 'furnace'
            ('nearbyCraft', 0),
            # (Obtain* tasks only) 'none', 'iron_ingot', 'coal'
            ('nearbySmelt', 0),
        ])
        for key, space in self.wrapping_action_space.spaces.items():
            if key not in self._noop_template:
                raise ValueError('Unknown action name: {}'.format(key))

        # get noop
        self.noop = copy.deepcopy(self._noop_template)
        for key in self._noop_template:
            if key not in self.wrapping_action_space.spaces:
                del self.noop[key]

        value_means = []
        value_ranges = []
        self.is_binary = []
        for key in self.noop:
            if key in self.BINARY_KEYS:
                value_means.append(0.5)
                value_ranges.append(0.5)
                self.is_binary.append(True)
            elif key == 'camera':
                value_means.append(0)
                value_means.append(0)
                self.is_binary.append(False)
                value_ranges.append(max_camera_range)
                value_ranges.append(max_camera_range)
                self.is_binary.append(False)
            elif key in {'place', 'craft', 'nearbyCraft', 'nearbySmelt'}:
                # TODO: implementation
                value_means.append(0)
                value_ranges.append(0)
        self.value_means = np.array(value_means, dtype=np.float32)
        self.value_ranges = np.array(value_ranges, dtype=np.float32)

        n = len(self.value_means)
        self.action_space = gym.spaces.Box(low=-np.ones(n), high=np.ones(n), dtype=np.float32)
        logger.info('{} is converted to {}.'.format(self.wrapping_action_space, self.action_space))

    def _action(self, action):
        original_action = copy.deepcopy(self.noop)
        idx = 0
        for key, is_binary in zip(self.noop, self.is_binary):
            if key == 'camera':
                orig_values = np.clip(action[idx:idx + 2], -1, 1)
                values = (orig_values * self.value_ranges[idx:idx + 2]
                          + self.value_means[idx:idx + 2])
                if not self.allow_pitch:
                    values[0] = 0

                original_action[key] = values
                idx += 2
            elif is_binary:
                value = (action[idx] * self.value_ranges[idx]
                         + self.value_means[idx])
                if np.random.rand() < value:
                    original_action[key] = 1
                else:
                    original_action[key] = 0
                idx += 1
            else:
                # noop
                # value = (action[idx] * self.value_ranges[idx]
                #          + self.value_means[idx])
                # original_action[key] = value
                idx += 1
        return original_action


class MultiDimensionalSoftmaxActionWrapper(gym.ActionWrapper):
    BINARY_KEYS = ['forward', 'back', 'left', 'right', 'jump', 'sneak', 'sprint', 'attack']

    def __init__(self, env, allow_pitch=False, max_camera_range=10,
                 num_camera_discretize=7):
        super().__init__(env)

        self.allow_pitch = allow_pitch
        self.max_camera_range = max_camera_range
        self.num_camera_discretize = num_camera_discretize
        self.wrapping_action_space = self.env.action_space
        self.noop = OrderedDict([
            ('forward', 0),
            ('back', 0),
            ('left', 0),
            ('right', 0),
            ('jump', 0),
            ('sneak', 0),
            ('sprint', 0),
            ('attack', 0),
            ('camera', np.zeros((2,), dtype=np.float32)),
            # 'none', 'dirt' (Obtain*:)+ 'stone', 'cobblestone', 'crafting_table', 'furnace', 'torch'
            ('place', 0),
            # (Obtain* tasks only) 'none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe'
            ('equip', 0),
            # (Obtain* tasks only) 'none', 'torch', 'stick', 'planks', 'crafting_table'
            ('craft', 0),
            # (Obtain* tasks only) 'none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe', 'furnace'
            ('nearbyCraft', 0),
            # (Obtain* tasks only) 'none', 'iron_ingot', 'coal'
            ('nearbySmelt', 0),
        ])
        for key, space in self.wrapping_action_space.spaces.items():
            if key not in self.noop:
                raise ValueError('Unknown action name: {}'.format(key))

        _noop_template = copy.deepcopy(self.noop)
        for key in _noop_template:
            if key not in self.wrapping_action_space.spaces:
                del self.noop[key]

        # get each discrete action
        self._actions = [self.noop]
        num_actions = []
        for key in self.noop:
            if self.num_camera_discretize % 2 == 0:
                raise ValueError('Number of camera discretization must be odd.')
            if key == 'camera':
                num_actions.append(self.num_camera_discretize)
                num_actions.append(self.num_camera_discretize)
            else:
                num_actions.append(self.wrapping_action_space.spaces[key].n)
        self.action_space = gym.spaces.Box(
            low=np.zeros_like(num_actions),
            high=(np.array(num_actions) - 1),
            dtype=np.int32)
        logger.info('{} is converted to {}.'.format(self.wrapping_action_space, self.action_space))

    def action(self, action):
        original_action = OrderedDict(copy.deepcopy(self.noop))
        half_scale = (self.num_camera_discretize - 1) / 2
        idx = 0
        for key in self.noop:
            if key == 'camera':
                original_action['camera'][0] = (
                        self.max_camera_range
                        * (action[idx] - half_scale)
                        / half_scale)
                original_action['camera'][1] = (
                        self.max_camera_range
                        * (action[idx + 1] - half_scale)
                        / half_scale)
                if not self.allow_pitch:
                    original_action['camera'][0] = 0
                idx += 2
            else:
                original_action[key] = int(action[idx])
                idx += 1
        logger.debug('discrete action {} -> original action {}'.format(action, original_action))
        return original_action


class BranchedRandomizedAction(gym.ActionWrapper):
    def __init__(self, env, branch_sizes, random_fraction):
        super().__init__(env)
        assert 0 <= random_fraction <= 1
        self._random_fraction = random_fraction
        self._np_random = np.random.RandomState()

    def action(self, action):
        if self._np_random.rand() < self._random_fraction:
            action = [self._np_random.randint(n) for n in self.branch_sizes]
            action = np.array(action)

        return action

    def seed(self, seed):
        super().seed(seed)
        self._np_random.seed(seed)


class BranchedActionWrapper(gym.ActionWrapper):
    def __init__(self, env, branch_sizes, camera_atomic_actions,
                 max_range_of_camera):
        super().__init__(env)
        self.env = env
        self.branch_sizes = branch_sizes
        self.camera_atomic_actions = camera_atomic_actions
        self.max_range_of_camera = max_range_of_camera

    def action(self, action):
        for i, branch_action in enumerate(action):
            assert (branch_action >= 0 and branch_action < self.branch_sizes[i])

        action_back_forward = action[0] // 3

        if action_back_forward == 1:
            action_back = 1
        else:
            action_back = 0

        if action_back_forward == 2:
            action_forward = 1
        else:
            action_forward = 0

        action_left_right = action[0] % 3

        if action_left_right == 1:
            action_left = 1
        else:
            action_left = 0

        if action_left_right == 2:
            action_right = 1
        else:
            action_right = 0

        action_jump = (action[1] & 1)
        action_sprint = (action[1] & 2) // 2
        action_sneak = (action[1] & 4) // 4
        action_attack = (action[1] & 8) // 8

        segment_size = 2 * self.max_range_of_camera / (self.camera_atomic_actions - 1)
        camera0 = action[2] * segment_size
        camera0 -= self.max_range_of_camera
        camera1 = action[3] * segment_size
        camera1 -= self.max_range_of_camera

        # assert abs(camera0) <= self.max_range_of_camera
        # assert abs(camera1) <= self.max_range_of_camera

        minerl_action = {
            'back': action_back,
            'forward': action_forward,
            'left': action_left,
            'right': action_right,
            'jump': action_jump,
            'sprint': action_sprint,
            'sneak': action_sneak,
            'attack': action_attack,
            'camera': np.array([camera0, camera1]),
        }

        if 'place' in self.env.action_space:
            assert (len(action) == 5)
            num_place_actions = len(self.env.action_space['place'])

            if num_place_actions == 2:  # Navigate envs
                minerl_action['place'] = action[4]
            elif num_place_actions == 7:  # Obtain envs
                craft = 0
                equip = 0
                nearbyCraft = 0
                nearbySmelt = 0
                place = 0

                if action[4] > 0 and action[4] <= 5:
                    craft = action[4] - 1
                elif action[4] <= 13:
                    equip = action[4] - 6
                elif action[4] <= 21:
                    nearbyCraft = action[4] - 14
                elif action[4] <= 24:
                    nearbySmelt = action[4] - 22
                elif action[4] <= 31:
                    place = action[4] - 25

                minerl_action['craft'] = craft
                minerl_action['equip'] = equip
                minerl_action['nearbyCraft'] = nearbyCraft
                minerl_action['nearbySmelt'] = nearbySmelt
                minerl_action['place'] = place
            else:
                raise Exception("Invalid number of place actions")

        return minerl_action


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, channel_order='hwc'):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.stack_axis = {'hwc': 2, 'chw': 0}[channel_order]
        orig_obs_space = env.observation_space
        low = np.repeat(orig_obs_space.low, k, axis=self.stack_axis)
        high = np.repeat(orig_obs_space.high, k, axis=self.stack_axis)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=orig_obs_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames), stack_axis=self.stack_axis)


class ScaledFloatFrame(gym.ObservationWrapper):
    """Divide frame values by 255.0 and return them as np.float32.
    Especially, when the original env.observation_space is np.uint8,
    this wrapper converts frame values into [0.0, 1.0] of dtype np.float32.
    """

    def __init__(self, env):
        assert isinstance(env.observation_space, spaces.Box)
        gym.ObservationWrapper.__init__(self, env)

        self.scale = 255.0

        orig_obs_space = env.observation_space
        self.observation_space = spaces.Box(
            low=self.observation(orig_obs_space.low),
            high=self.observation(orig_obs_space.high),
            dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / self.scale


class LazyFrames(object):
    """Array-like object that lazily concat multiple frames.
    This object ensures that common frames between the observations are only
    stored once.  It exists purely to optimize memory usage which can be huge
    for DQN's 1M frames replay buffers.
    This object should only be converted to numpy array before being passed to
    the model.
    You'd not believe how complex the previous solution was.
    """

    def __init__(self, frames, stack_axis=2):
        self.stack_axis = stack_axis
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=self.stack_axis)
        if dtype is not None:
            out = out.astype(dtype)
        return out
