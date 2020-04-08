import torch
import torch.distributions

from dreamer.models.observation import ObservationEncoder, ObservationDecoder


def test_observation_encoder():
    encoder = ObservationEncoder()

    batch_size = 2
    channels = 3
    width = 64
    height = 64
    image_obs = torch.randn(batch_size, channels, width, height)
    with torch.no_grad():
        embedding: torch.Tensor = encoder(image_obs)
    assert embedding.size(0) == batch_size
    assert embedding.size(1) == 1024


def test_observation_decoder():
    decoder = ObservationDecoder()

    batch_size = 2
    channels = 3
    width = 64
    height = 64
    embedding = torch.randn(batch_size, 1024)
    with torch.no_grad():
        obs_dist: torch.distributions.Normal = decoder(embedding)
    obs_sample: torch.Tensor = obs_dist.sample()
    assert obs_sample.size(0) == batch_size
    assert obs_sample.size(1) == channels
    assert obs_sample.size(2) == width
    assert obs_sample.size(3) == height

    # Test a version where we have 2 batch dimensions
    horizon = 4
    embedding = torch.randn(batch_size, horizon, 1024)
    with torch.no_grad():
        obs_dist: torch.distributions.Normal = decoder(embedding)
    obs_sample: torch.Tensor = obs_dist.sample()
    assert obs_sample.size(0) == batch_size
    assert obs_sample.size(1) == horizon
    assert obs_sample.size(2) == channels
    assert obs_sample.size(3) == width
    assert obs_sample.size(4) == height



def test_observation():
    batch_size = 2
    channels = 3
    width = 64
    height = 64

    encoder = ObservationEncoder()
    decoder = ObservationDecoder()

    image_obs = torch.randn(batch_size, channels, width, height)

    with torch.no_grad():
        obs_dist: torch.distributions.Normal = decoder(encoder(image_obs))
    obs_sample: torch.Tensor = obs_dist.sample()
    assert obs_sample.size(0) == batch_size
    assert obs_sample.size(1) == channels
    assert obs_sample.size(2) == width
    assert obs_sample.size(3) == height

    embedding = torch.randn(batch_size, 1024)

    with torch.no_grad():
        embedding: torch.Tensor = encoder(decoder(embedding).sample())
    assert embedding.size(0) == batch_size
    assert embedding.size(1) == 1024
