import pytest
import torch

from dreamer.models.dense import DenseModel


@pytest.mark.parametrize('dist', ['normal', 'binary'])
def test_dense_model(dist):
    shape = (1,)
    units = 20
    feature_size = 20
    layers = 5
    batch_size = 2
    features = torch.randn((batch_size, feature_size))

    try:
        dense = DenseModel(feature_size, shape, layers, units, dist)
    except NotImplementedError:
        return

    output = dense(features)
    sample = output.sample()
    assert isinstance(sample, torch.Tensor)
    # print(sample.size())
    assert sample.size() == (batch_size, *shape)
