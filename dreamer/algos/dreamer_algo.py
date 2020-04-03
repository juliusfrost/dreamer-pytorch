import torch

from dreamer.models.rnns import RSSMState

from rlpyt.algos.base import RlAlgorithm
from rlpyt.utils.quick_args import save__init__args


class Dreamer(RlAlgorithm):

    def __init__(
            self,  # Hyper-parameters
            batch_size=50,
            batch_length=50,
            train_every=1000,
            train_steps=100,
            pretrain=100,
            model_lr=6e-4,
            value_lr=8e-5,
            actor_lr=8e-5,
            grad_clip=100.0,
            dataset_balance=False,
            discount=0.99,
            disclam=0.95,
            horizon=15,
            action_dist='tanh_normal',
            action_init_std=5.0,
            expl='additive_gaussian',
            expl_amount=0.3,
            expl_decay=0.0,
            expl_min=0.0,
            optim_kwargs=None,
    ):
        super().__init__()

        self._batch_size = batch_size
        del batch_size  # Property.
        save__init__args(locals())
        self.update_counter = 0

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples, world_size=1, rank=0):
        pass

    def async_initialize(self, agent, sampler_n_itr, batch_spec, mid_batch_reset, examples, world_size=1):
        pass

    def optim_initialize(self, rank=0):
        pass

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        pass

    def loss(self):
        return

    def model_loss(self):
        return

    def agent_loss(self):
        return

    def value_loss(self):
        return
