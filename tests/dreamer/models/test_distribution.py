import torch

from dreamer.models.distribution import SampleDist, TanhBijector


def test_dist():
    batch_size = 4
    dist_size = 3
    samples = 10
    mean = torch.randn(batch_size, dist_size)
    std = torch.rand(batch_size, dist_size)
    dist = torch.distributions.Normal(mean, std)
    transform = TanhBijector()
    sign = transform.sign
    dist = torch.distributions.TransformedDistribution(dist, transform)
    dist = torch.distributions.Independent(dist, 1)
    dist = SampleDist(dist, samples)
    name = dist.name
    assert dist.event_shape == (dist_size,)
    assert dist.batch_shape == (batch_size,)
