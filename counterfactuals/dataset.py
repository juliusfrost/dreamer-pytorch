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

file_structure = {
    'episode': {
        'step': ['state', 'observation', 'action', 'reward']
    }
}

NUM_DIGITS = 5
MAX_NUMBER = 10 ** (NUM_DIGITS + 1) - 1


def save_item(path, obj):
    if isinstance(obj, torch.Tensor):
        torch.save(obj, path + '.pt')
    elif isinstance(obj, np.ndarray):
        np.save(path + '.npy', obj)
    else:
        pickle.dump(obj, open(path + '.pickle', "wb"))


def load_item(path):
    basename = os.path.basename(path)
    if basename.split('.')[-1] == 'pt':
        obj = torch.load(path)
    elif basename.split('.')[-1] == 'npy':
        obj = np.load(path)
    else:
        obj = pickle.load(open(path, "rb"))
    return obj


def leading_zeros(num: int):
    if num > MAX_NUMBER:
        raise ValueError(f'The maximum number is {MAX_NUMBER} but tried to use {num}')
    return f'{num:0{NUM_DIGITS}d}'


def downsize(obj):
    if isinstance(obj, torch.Tensor):
        obj = obj.cpu().numpy()
    if isinstance(obj, np.ndarray):
        if obj.size == 1:
            obj = obj.item()
    return obj


def stack(objs: list):
    if isinstance(objs[0], torch.Tensor):
        return torch.stack(objs)
    elif isinstance(objs[0], np.ndarray):
        return np.stack(objs)
    else:
        return objs


def unstack(objs):
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

        :param save_location: path to dataset save location
        """
        self.save_location = save_location

        self.memory_states = []
        self.memory_observations = []
        self.memory_actions = []
        self.memory_rewards = []
        self.memory_episodes = []
        self.memory_steps = []

    def append(self, episode, step, trajectory: TrajectoryStep):
        self.memory_episodes.append(episode)
        self.memory_steps.append(step)
        self.memory_states.append(downsize(trajectory.state))
        self.memory_observations.append(downsize(trajectory.observation))
        self.memory_actions.append(downsize(trajectory.action))
        self.memory_rewards.append(downsize(trajectory.reward))

    def add(self, episode, step, state, observation, action, reward):
        self.append(episode, step, TrajectoryStep(state, observation, action, reward))

    def trajectory_slices(self):
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

    def load(self):
        count = 0
        for f in os.listdir(self.save_location):
            episode_path = os.path.join(self.save_location, f)
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


class TorchTrajectoryDataset(TrajectoryDataset, torch.utils.data.Dataset):
    def __init__(self, save_location=None):
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
    return [Trajectory(
        torch.tensor(trajectory.episode),
        torch.tensor(trajectory.states),
        torch.tensor(trajectory.observations),
        torch.tensor(trajectory.actions),
        torch.tensor(trajectory.rewards),
        torch.tensor(trajectory.steps),
    ) for trajectory in list_trajectory]
