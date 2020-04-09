import pytest
import torch
import torch.distributions

from dreamer.models.observation import ObservationEncoder, ObservationDecoder, conv_out, conv_out_shape, \
    output_padding, output_padding_shape


def test_observation_encoder(shape=(3, 64, 64)):
    encoder = ObservationEncoder()

    batch_size = 2
    c, h, w = shape
    image_obs = torch.randn(batch_size, c, h, w)
    with torch.no_grad():
        embedding: torch.Tensor = encoder(image_obs)
    assert embedding.size(0) == batch_size
    assert embedding.size(1) == 1024


def test_observation_decoder(shape=(3, 64, 64)):
    decoder = ObservationDecoder()

    batch_size = 2
    c, h, w = shape
    embedding = torch.randn(batch_size, 1024)
    with torch.no_grad():
        obs_dist: torch.distributions.Normal = decoder(embedding)
    obs_sample: torch.Tensor = obs_dist.sample()
    assert obs_sample.size(0) == batch_size
    assert obs_sample.size(1) == c
    assert obs_sample.size(2) == h
    assert obs_sample.size(3) == w

    # Test a version where we have 2 batch dimensions
    horizon = 4
    embedding = torch.randn(batch_size, horizon, 1024)
    with torch.no_grad():
        obs_dist: torch.distributions.Normal = decoder(embedding)
    obs_sample: torch.Tensor = obs_dist.sample()
    assert obs_sample.size(0) == batch_size
    assert obs_sample.size(1) == horizon
    assert obs_sample.size(2) == c
    assert obs_sample.size(3) == h
    assert obs_sample.size(4) == w

    # Test a version where we have 2 batch dimensions
    horizon = 4
    embedding = torch.randn(batch_size, horizon, 1024)
    with torch.no_grad():
        obs_dist: torch.distributions.Normal = decoder(embedding)
    obs_sample: torch.Tensor = obs_dist.sample()
    assert obs_sample.size(0) == batch_size
    assert obs_sample.size(1) == horizon
    assert obs_sample.size(2) == c
    assert obs_sample.size(3) == h
    assert obs_sample.size(4) == w


@pytest.mark.parametrize('shape', [(3, 64, 64), (4, 104, 64)])
def test_observation(shape):
    batch_size = 2
    c, h, w = shape

    encoder = ObservationEncoder(shape=shape)
    decoder = ObservationDecoder(embed_size=encoder.embed_size, shape=shape)

    image_obs = torch.randn(batch_size, c, h, w)

    with torch.no_grad():
        obs_dist: torch.distributions.Normal = decoder(encoder(image_obs))
    obs_sample: torch.Tensor = obs_dist.sample()
    assert obs_sample.size(0) == batch_size
    assert obs_sample.size(1) == c
    assert obs_sample.size(2) == h
    assert obs_sample.size(3) == w

    embedding = torch.randn(batch_size, encoder.embed_size)

    with torch.no_grad():
        embedding: torch.Tensor = encoder(decoder(embedding).sample())
    assert embedding.size(0) == batch_size
    assert embedding.size(1) == encoder.embed_size


def test_observation_reconstruction(shape=(4, 104, 64)):
    batch_size = 2
    c, h, w = shape
    depth = 32
    stride = 2
    activation = torch.nn.ReLU

    conv1 = torch.nn.Conv2d(c, 1 * depth, 6, stride)
    conv1_shape = conv_out_shape((h, w), 0, 6, 2)
    conv1_pad = output_padding_shape((h, w), conv1_shape, 0, 6, 2)
    conv2 = torch.nn.Conv2d(1 * depth, 2 * depth, 6, stride)
    conv2_shape = conv_out_shape(conv1_shape, 0, 6, 2)
    conv2_pad = output_padding_shape(conv1_shape, conv2_shape, 0, 6, 2)
    conv3 = torch.nn.Conv2d(2 * depth, 4 * depth, 5, stride)
    conv3_shape = conv_out_shape(conv2_shape, 0, 5, 2)
    conv3_pad = output_padding_shape(conv2_shape, conv3_shape, 0, 5, 2)
    conv4 = torch.nn.Conv2d(4 * depth, 32 * depth, 5, stride)
    conv4_shape = conv_out_shape(conv3_shape, 0, 5, 2)
    conv4_pad = output_padding_shape(conv3_shape, conv4_shape, 0, 5, 2)

    decoder = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(32 * depth, 4 * depth, 5, stride, output_padding=conv4_pad),
        activation(),
        torch.nn.ConvTranspose2d(4 * depth, 2 * depth, 5, stride, output_padding=conv3_pad),
        activation(),
        torch.nn.ConvTranspose2d(2 * depth, 1 * depth, 6, stride, output_padding=conv2_pad),
        activation(),
        torch.nn.ConvTranspose2d(1 * depth, shape[0], 6, stride, output_padding=conv1_pad),
    )

    image_obs = torch.randn(batch_size, c, h, w)

    x1 = conv1(image_obs)
    x2 = conv2(x1)
    x3 = conv3(x2)
    x4 = conv4(x3)

    assert x4.shape == (batch_size, 32 * depth, *conv4_shape)

    reconstructed = decoder(x4)

    assert reconstructed.shape == image_obs.shape
