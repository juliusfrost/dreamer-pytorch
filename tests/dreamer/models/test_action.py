import pytest
import torch

from dreamer.models.action import ActionDecoder


@pytest.mark.parametrize('dist', ['tanh_normal', 'one_hot', 'relaxed_one_hot', 'not_implemented_dist'])
def test_action_decoder(dist):
    batch_size = 4
    action_size = 10
    feature_size = 20
    hidden_size = 40
    layers = 5

    try:
        action_decoder = ActionDecoder(action_size, feature_size, hidden_size, layers, dist)
    except NotImplementedError:
        return

    features = torch.randn(batch_size, feature_size)

    action_dist = action_decoder(features)

    if dist == 'tanh_normal':
        action_mean = action_dist.mean()
        action_mode = action_dist.mode()
        action_ent = action_dist.entropy()

        assert isinstance(action_mean, torch.Tensor)
        assert action_mean.shape == (batch_size, action_size)

        assert isinstance(action_mode, torch.Tensor)
        assert action_mode.shape == (batch_size, action_size)

        assert isinstance(action_ent, torch.Tensor)
        assert action_ent.shape == (batch_size,)

        true_action = torch.randn(batch_size, action_size)

        # make sure gradients can propagate backwards
        loss = torch.sum((action_mean - true_action) ** 2)
        loss += torch.sum((action_mode - true_action) ** 2)
        loss += - torch.sum(action_ent)
        loss.backward()
    elif dist == 'one_hot':
        action_mean = action_dist.mean
        action_ent = action_dist.entropy()

        assert isinstance(action_mean, torch.Tensor)
        assert action_mean.shape == (batch_size, action_size)

        assert isinstance(action_ent, torch.Tensor)
        assert action_ent.shape == (batch_size,)
