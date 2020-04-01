import torch
import torch.nn.functional as F
import torch.distributions


class TanhBijector(torch.distributions.Transform):
    def __init__(self):
        super().__init__()
        self.bijective = True

    @property
    def sign(self):
        return 1.

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y: torch.Tensor):
        y = torch.where(
            (torch.abs(y) <= 1.),
            torch.clamp(y, -0.99999997, 0.99999997),
            y
        )

        y = atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        log2 = torch.log(torch.tensor(2.))
        return 2. * (log2 - x - F.softplus(-2. * x))


class SampleDist:

    def __init__(self, dist: torch.distributions.Distribution, samples=100):
        self._dist = dist.expand((samples, *dist.batch_shape))
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        sample = self._dist.rsample()
        return torch.mean(sample, 0)

    def mode(self):
        sample = self._dist.rsample()
        logprob = self._dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        sample = self._dist.rsample()
        logprob = self._dist.log_prob(sample)
        return -torch.mean(logprob, 0)


def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))
