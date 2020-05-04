from counterfactuals.dataset import Trajectory
from counterfactuals.state_selection.random_state_selection import RandomStateSelection
from tests.counterfactuals.dataset import add_data

def test_random_state_selection(dataset):
    dataset = add_data(dataset)
    rss = RandomStateSelection(dataset)
    results = rss.select_indices(5)
    assert len(results == 5)
    for result in results:
        assert isinstance(result, Trajectory)

