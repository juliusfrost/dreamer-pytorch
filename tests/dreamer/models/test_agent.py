import pytest
import torch

from dreamer.models.agent import AgentModel


@pytest.mark.parametrize('dist', ['tanh_normal', 'one_hot', 'relaxed_one_hot'])
def test_agent(dist):
    batch_size = 1
    action_shape = (2,)
    deterministic_size = 200
    obs_shape = (3, 64, 64)
    agent_model = AgentModel(action_shape, deterministic_size=deterministic_size, image_shape=obs_shape,
                             action_dist=dist)

    observation = torch.randn(batch_size, *obs_shape)
    prev_action = None
    prev_state = None

    state = agent_model.get_state_representation(observation, prev_action, prev_state)
    assert state.deter.shape == (batch_size, deterministic_size)

    action, action_dist = agent_model.policy(state)
    assert action.shape == (batch_size, *action_shape)

    agent_model.eval()
    try:
        _ = agent_model(observation, prev_action, prev_state)
    except NotImplementedError:
        pass

    next_state = agent_model.get_state_transition(action, state)
    assert next_state.deter.shape == (batch_size, deterministic_size)
