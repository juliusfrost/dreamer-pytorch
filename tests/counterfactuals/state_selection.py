from counterfactuals.dataset import TrajectoryDataset
from counterfactuals.state_selection.random_state_selection import RandomStateSelection
from tests.counterfactuals.dataset import add_data

def test_random_state_selection():
    dataset = TrajectoryDataset()
    dataset = add_data(dataset)
    rss = RandomStateSelection(dataset)
    results = rss.select_indices(3)
    assert len(results) == 2
    assert len(results[0]) == 3
