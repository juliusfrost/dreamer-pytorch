import numpy as np
import torch
from torch.utils.data import DataLoader

from counterfactuals.dataset import TrajectoryDataset, TorchTrajectoryDataset, TrajectoryStep, collate_fn


def append_data(dataset: TrajectoryDataset):
    dataset.append(0, 0, TrajectoryStep(torch.zeros(2), np.random.randn(2), 0, 0))
    dataset.append(0, 1, TrajectoryStep(torch.zeros(2), np.random.randn(2), 0, 0))
    dataset.append(1, 0, TrajectoryStep(torch.zeros(2), np.random.randn(2), 0, 0))
    dataset.append(1, 1, TrajectoryStep(torch.zeros(2), np.random.randn(2), 0, 0))
    return dataset


def add_data(dataset: TrajectoryDataset):
    dataset.add(0, 0, torch.zeros(2), np.random.randn(2), 0, 0)
    dataset.add(0, 1, torch.zeros(2), np.random.randn(2), 0, 0)
    dataset.add(1, 0, torch.zeros(2), np.random.randn(2), 0, 0)
    dataset.add(1, 1, torch.zeros(2), np.random.randn(2), 0, 0)
    return dataset


def add_unsorted(dataset: TrajectoryDataset):
    dataset.add(1, 1, torch.zeros(2), np.random.randn(2), 0, 0)
    dataset.add(0, 1, torch.zeros(2), np.random.randn(2), 0, 0)
    dataset.add(1, 0, torch.zeros(2), np.random.randn(2), 0, 0)
    dataset.add(0, 0, torch.zeros(2), np.random.randn(2), 0, 0)
    return dataset


def test_trajectory_dataset():
    dataset = TrajectoryDataset()
    append_data(dataset)

    for trajectory in dataset.trajectory_generator():
        assert isinstance(trajectory.states, np.ndarray)
        assert isinstance(trajectory.observations, np.ndarray)
        assert isinstance(trajectory.actions, list)
        assert isinstance(trajectory.rewards, list)


def test_torch_trajectory_dataset():
    dataset = TorchTrajectoryDataset()
    add_data(dataset)
    dataset.reset()

    for trajectory in DataLoader(dataset, batch_size=1, collate_fn=collate_fn):
        trajectory = trajectory[0]
        assert isinstance(trajectory.states, torch.Tensor)
        assert isinstance(trajectory.observations, torch.Tensor)
        assert isinstance(trajectory.actions, torch.Tensor)
        assert isinstance(trajectory.rewards, torch.Tensor)


def test_sort_trajectory_dataset():
    dataset = TrajectoryDataset()
    add_unsorted(dataset)
    dataset.sort()

    prev_episode = -1
    for trajectory in dataset.trajectory_generator():
        assert trajectory.steps[0] < trajectory.steps[1]
        assert prev_episode < trajectory.episode
        prev_episode = trajectory.episode
