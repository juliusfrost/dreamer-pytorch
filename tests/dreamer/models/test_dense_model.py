import pytest
import torch
import torch.nn as nn
from dreamer.models.dense_model import DenseModel
import numpy as np
                                                
@pytest.mark.parametrize('dist', ['normal', 'binary'])
def test_DenseModel(dist):
    shape = ()
    units = 20
    feature_size = 20
    layers = 5
    batch_size = 2
    features = torch.randn((batch_size,feature_size))
    
    try:
        dense = DenseModel(feature_size , shape, layers, units, dist, nn.ELU())
    except NotImplementedError:
        return

    
    output = dense(features)
    sample = output.sample()
    assert isinstance(sample , torch.Tensor)
    # print(sample.size())
    assert sample.size() == (batch_size,)
