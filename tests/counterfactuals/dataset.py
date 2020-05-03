import numpy as np
import torch
from torch.utils.data import DataLoader

from counterfactuals.dataset import TrajectoryDataset, TorchTrajectoryDataset, TrajectoryStep, collate_fn


def append_data(dataset: TrajectoryDataset):
    dataset.append(0, 0, TrajectoryStep(torch.zeros(2), np.random.randn(2), 0, 0, False, None))
    dataset.append(0, 1, TrajectoryStep(torch.zeros(2), np.random.randn(2), 0, 0, True, None))
    dataset.append(1, 0, TrajectoryStep(torch.zeros(2), np.random.randn(2), 0, 0, False, None))
    dataset.append(1, 1, TrajectoryStep(torch.zeros(2), np.random.randn(2), 0, 0, True, None))
    return dataset


def add_data(dataset: TrajectoryDataset):
    dataset.add(0, 0, torch.zeros(2), np.random.randn(2), 0, 0, False, None)
    dataset.add(0, 1, torch.zeros(2), np.random.randn(2), 0, 0, True, None)
    dataset.add(1, 0, torch.zeros(2), np.random.randn(2), 0, 0, False, None)
    dataset.add(1, 1, torch.zeros(2), np.random.randn(2), 0, 0, True, None)
    return dataset


def add_unsorted(dataset: TrajectoryDataset):
    dataset.add(1, 1, torch.zeros(2), np.random.randn(2), 0, 0, True, None)
    dataset.add(0, 1, torch.zeros(2), np.random.randn(2), 0, 0, True, None)
    dataset.add(1, 0, torch.zeros(2), np.random.randn(2), 0, 0, False, None)
    dataset.add(0, 0, torch.zeros(2), np.random.randn(2), 0, 0, False, None)
    return dataset


def test_trajectory_dataset():
    dataset = TrajectoryDataset()
    append_data(dataset)

    for trajectory in dataset.trajectory_generator():
        assert isinstance(trajectory.state, np.ndarray)
        assert isinstance(trajectory.observation, np.ndarray)
        assert isinstance(trajectory.action, list)
        assert isinstance(trajectory.reward, list)


def test_torch_trajectory_dataset():
    dataset = TorchTrajectoryDataset()
    add_data(dataset)
    dataset.reset()

    for trajectory in DataLoader(dataset, batch_size=1, collate_fn=collate_fn):
        trajectory = trajectory[0]
        assert isinstance(trajectory.state, torch.Tensor)
        assert isinstance(trajectory.observation, torch.Tensor)
        assert isinstance(trajectory.action, torch.Tensor)
        assert isinstance(trajectory.reward, torch.Tensor)


def test_sort_trajectory_dataset():
    dataset = TrajectoryDataset()
    add_unsorted(dataset)
    dataset.sort()

    prev_episode = -1
    for trajectory in dataset.trajectory_generator():
        assert trajectory.step[0] < trajectory.step[1]
        assert prev_episode < trajectory.episode
        prev_episode = trajectory.episode


def test_save_load_trajectory_dataset():
    path = '../../data/tests'
    dataset = TrajectoryDataset()
    add_data(dataset)

    dataset.save(path)
    dataset2 = dataset.load(path)
