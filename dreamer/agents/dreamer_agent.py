import torch

from rlpyt.agents.base import BaseAgent, RecurrentAgentMixin
from rlpyt.utils.buffer import buffer_to


# see classes BaseAgent and RecurrentAgentMixin for documentation
class DreamerAgent(BaseAgent, RecurrentAgentMixin):
    def __call__(self, *args):
        model_inputs = buffer_to(args, device=self.device)
        return self.model(*model_inputs)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        pass
