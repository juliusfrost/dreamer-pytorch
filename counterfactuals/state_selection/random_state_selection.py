from random import randint
from counterfactuals.dataset import TorchTrajectoryDataset, TrajectoryDataset, Trajectory
from typing import List


class RandomStateSelection:
    """
    Select random frames from a dataset for counterfactual representation
    """
    def __init__(self, dataset: TrajectoryDataset):

        self.dataset = dataset
        

    def select_indices(self, num_samples: int) -> List[Trajectory]:
        """
        Select num_samples Trajectories from
        """
        
        return [self.dataset.get_frame(randint(0, self.dataset.num_frames - 1)) for _ in range(num_samples)]


