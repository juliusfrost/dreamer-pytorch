from random import randint
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
        NOTE: Returns index, not step number
        """
        samples = []
        for trajectory in self.dataset.trajectory_generator():
            samples.append([randint(0, len(trajectory.step)) for _ in range(num_samples)])
        return samples
