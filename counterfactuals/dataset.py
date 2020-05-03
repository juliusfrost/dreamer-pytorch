import glob
import os
import pickle
from collections import namedtuple
from typing import List

import numpy as np
import torch
import torch.utils.data

# state can be either simulation state or model state.
TrajectoryStep = namedtuple('TrajectoryStep', ['state', 'observation', 'action', 'reward'])

Trajectory = namedtuple('Trajectory', ['episode', 'states', 'observations', 'actions', 'rewards', 'steps'])

NUM_DIGITS = 5
MAX_NUMBER = 10 ** (NUM_DIGITS + 1) - 1


def save_item(path, obj):
    """saves the object to file based on its type"""
    if isinstance(obj, torch.Tensor):
        torch.save(obj, path + '.pt')
    elif isinstance(obj, np.ndarray):
        np.save(path + '.npy', obj)
    else:
        pickle.dump(obj, open(path + '.pickle', "wb"))


def load_item(path):
    """loads an object from the path depending on file type"""
    basename = os.path.basename(path)
    if basename.split('.')[-1] == 'pt':
        obj = torch.load(path)
    elif basename.split('.')[-1] == 'npy':
        obj = np.load(path)
    else:
        obj = pickle.load(open(path, "rb"))
    return obj


def leading_zeros(num: int):
    """:returns a string of the number with leading zeros"""
    if num > MAX_NUMBER:
        raise ValueError(f'The maximum number is {MAX_NUMBER} but tried to use {num}')
    return f'{num:0{NUM_DIGITS}d}'


def downsize(obj):
    """
    Use this method to save memory where possible.
    Turns torch tensors into numpy arrays. If the size is 1, then it turns arrays into scalars.
    :param obj: any object
    :return: downsized object
    """
    if isinstance(obj, torch.Tensor):
        obj = obj.cpu().numpy()
    if isinstance(obj, np.ndarray):
        if obj.size == 1:
            obj = obj.item()
    return obj


def stack(objs: list):
    """stacks list of tensors/arrays into one big tensor/array"""
    if isinstance(objs[0], torch.Tensor):
        return torch.stack(objs)
    elif isinstance(objs[0], np.ndarray):
        return np.stack(objs)
    else:
        return objs


def unstack(objs):
    """unstacks tensor/array to a list of tensors/arrays"""
    if isinstance(objs, torch.Tensor):
        return list(objs)
    elif isinstance(objs, np.ndarray):
        return list(objs)
    else:
        assert isinstance(objs, list)
        return objs


class TrajectoryDataset:

    def __init__(self, save_location=None):
        """
        Creates a trajectory dataset that you can save and load from.
        :param save_location: path to dataset save location
        """
        self.save_location = save_location

        self.memory_states = []
        self.memory_observations = []
        self.memory_actions = []
        self.memory_rewards = []
        self.memory_episodes = []
        self.memory_steps = []

    def append(self, episode: int, step: int, trajectory: TrajectoryStep):
        """
        Append a single time step to an episode in the dataset.
        :param episode: which episode to append to
        :param step: the time step in the episode
        :param trajectory: TrajectoryStep(state, observation, action, reward)
        :return:
        """
        self.memory_episodes.append(episode)
        self.memory_steps.append(step)
        self.memory_states.append(downsize(trajectory.state))
        self.memory_observations.append(downsize(trajectory.observation))
        self.memory_actions.append(downsize(trajectory.action))
        self.memory_rewards.append(downsize(trajectory.reward))

    def add(self, episode, step, state, observation, action, reward):
        """
        Same as append, but has explicit arguments.
        :param episode: typically an integer. specifies which episode to append to
        :param step: typically an integer. specifies the time step in the episode
        :param state: Can be simulation state or model state. Will downsize torch tensors and numpy arrays.
            If a different type of object is given it will keep the reference so make sure pass a copy!
        :param observation: Can by any object but preferably a tensor or array.
        :param action: Can be any object, int, tensor, or array.
        :param reward: Can be any object, int, tensor, or array.
        :return:
        """
        self.append(episode, step, TrajectoryStep(state, observation, action, reward))

    def trajectory_slices(self):
        """:returns a generator to iterate of episode slices"""
        prev_episode = None
        prev_index = 0
        for i in range(len(self.memory_episodes)):
            episode = self.memory_episodes[i]
            if episode != prev_episode and prev_episode is not None:
                yield prev_episode, slice(prev_index, i)
                prev_index = i
            prev_episode = episode
        yield prev_episode, slice(prev_index, len(self.memory_episodes))

    def trajectory_generator(self):
        """:returns a generator to iterate over available trajectories"""
        if len(self.memory_episodes) == 0:
            raise ValueError('No trajectories in memory! Either call load() or append() new trajectories.')
        for episode, trajectory_slice in self.trajectory_slices():
            yield Trajectory(
                episode,
                stack(self.memory_states[trajectory_slice]),
                stack(self.memory_observations[trajectory_slice]),
                stack(self.memory_actions[trajectory_slice]),
                stack(self.memory_rewards[trajectory_slice]),
                stack(self.memory_steps[trajectory_slice]),
            )

    def save(self, save_location=None):
        """
        Saves the whole dataset to save_location. Saves each episode in its own folder. The file format saved depends
        on the downsized format.
        :param save_location: must be a path. if not specified, it will use the one specified in __init__.
            One should provide an empty directory and avoid overwriting if possible.
        :return:
        """
        if save_location is None:
            save_location = self.save_location
            if save_location is None:
                raise ValueError('Please provide save location!')

        if not os.path.exists(save_location):
            os.makedirs(save_location)

        while len(os.listdir(save_location)) > 0:
            print(f'Empty directory not provided: {save_location}\nProceeding may overwrite the contents.')
            answer = input('Continue? y / [n] / (or enter different path): ')
            if answer == 'y':
                break
            elif answer == 'n' or answer == '':
                return
            elif os.path.exists(answer):
                save_location = answer
            else:
                return

        for trajectory in self.trajectory_generator():
            episode_dir = os.path.join(save_location, leading_zeros(trajectory.episode))
            if not os.path.exists(episode_dir):
                os.mkdir(episode_dir)
            save_item(os.path.join(episode_dir, 'states'), trajectory.state)
            save_item(os.path.join(episode_dir, 'observations'), trajectory.observation)
            save_item(os.path.join(episode_dir, 'actions'), trajectory.action)
            save_item(os.path.join(episode_dir, 'rewards'), trajectory.reward)
            save_item(os.path.join(episode_dir, 'steps'), trajectory.steps)

    def load(self, save_location=None):
        """
        Load the whole dataset from a save_location.
        :param save_location: must be a path. if not specified, it will use the one specified in __init__.
        :return:
        """
        if save_location is None:
            save_location = self.save_location
            if save_location is None:
                raise ValueError('Please provide save location!')
        count = 0
        for f in os.listdir(save_location):
            episode_path = os.path.join(save_location, f)
            if os.path.isdir(episode_path) and f.isnumeric():
                episode = int(f)
                print(f'loading episode {episode}')
                state_path = glob.glob(os.path.join(episode_path, 'states') + '.*')[0]
                observation_path = glob.glob(os.path.join(episode_path, 'observations') + '.*')[0]
                action_path = glob.glob(os.path.join(episode_path, 'actions') + '.*')[0]
                reward_path = glob.glob(os.path.join(episode_path, 'rewards') + '.*')[0]
                step_path = glob.glob(os.path.join(episode_path, 'steps') + '.*')[0]
                states = unstack(load_item(state_path))
                observations = unstack(load_item(observation_path))
                actions = unstack(load_item(action_path))
                rewards = unstack(load_item(reward_path))
                steps = unstack(load_item(step_path))

                for i in range(len(steps)):
                    trajectory = TrajectoryStep(states[i], observations[i], actions[i], rewards[i])
                    self.append(episode, steps[i], trajectory)
                count += 1
        print(f'Loaded {count} episodes')

    def sort(self):
        """Sort the memory, if for some reason you decided to add steps asynchronously."""
        episode_step = [leading_zeros(e) + leading_zeros(s) for e, s in zip(self.memory_episodes, self.memory_steps)]
        episode_indicies = np.argsort(episode_step)
        self.memory_episodes = [self.memory_episodes[i] for i in episode_indicies]
        self.memory_steps = [self.memory_steps[i] for i in episode_indicies]
        self.memory_states = [self.memory_states[i] for i in episode_indicies]
        self.memory_observations = [self.memory_observations[i] for i in episode_indicies]
        self.memory_actions = [self.memory_actions[i] for i in episode_indicies]
        self.memory_rewards = [self.memory_rewards[i] for i in episode_indicies]


class TorchTrajectoryDataset(TrajectoryDataset, torch.utils.data.Dataset):
    def __init__(self, save_location=None):
        """Trajectory dataset with torch compatibility. Make sure you call reset() before iterating through.
            This is done after adding trajectories or loading to finalize the list of slices for iterating."""
        super().__init__(save_location)
        self.slices = None

    def reset(self):
        self.slices = list(self.trajectory_slices())

    def load(self):
        super().load()
        self.reset()

    def __len__(self):
        if self.slices is None:
            raise ValueError('Must call load() or reset() first!')
        return len(self.slices)

    def __getitem__(self, item):
        if self.slices is None:
            raise ValueError('Must call load() or reset() first!')
        episode, trajectory_slice = self.slices[item]
        return Trajectory(
            episode,
            stack(self.memory_states[trajectory_slice]),
            stack(self.memory_observations[trajectory_slice]),
            stack(self.memory_actions[trajectory_slice]),
            stack(self.memory_rewards[trajectory_slice]),
            stack(self.memory_steps[trajectory_slice]),
        )


def collate_fn(list_trajectory: List[Trajectory]):
    """Tensorfies the contents of the trajectories but keeps in a list. Use with torch.utils.data.DataLoader"""
    return [Trajectory(
        torch.tensor(trajectory.episode),
        torch.tensor(trajectory.states),
        torch.tensor(trajectory.observations),
        torch.tensor(trajectory.actions),
        torch.tensor(trajectory.rewards),
        torch.tensor(trajectory.steps),
    ) for trajectory in list_trajectory]
