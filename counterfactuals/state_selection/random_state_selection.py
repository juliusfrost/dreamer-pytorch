from random import sample
from counterfactuals.dataset import TorchTrajectoryDataset, TrajectoryDataset, Trajectory
from typing import List


class RandomStateSelection:
    """
    Select random frames from a dataset for counterfactual representation
    """
    def __init__(self, dataset: TrajectoryDataset):

        self.dataset = dataset
        

    def select_indices(self, num_samples: int):
        """
        Select num_samples Trajectories from
        NOTE: Returns step number, not index in list.
        """
        samples = []
        for trajectory in self.dataset.trajectory_generator():
            samples.append(sample(list(trajectory.step), num_samples))
        return samples
