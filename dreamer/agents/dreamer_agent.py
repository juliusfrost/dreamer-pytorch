import numpy as np
import torch
from rlpyt.agents.base import BaseAgent, RecurrentAgentMixin, AgentStep
from rlpyt.utils.buffer import buffer_to, buffer_func
from rlpyt.utils.collections import namedarraytuple

from dreamer.models.agent import AgentModel

DreamerAgentInfo = namedarraytuple('DreamerAgentInfo', ['prev_state'])


# see classes BaseAgent and RecurrentAgentMixin for documentation
class DreamerAgent(RecurrentAgentMixin, BaseAgent):

    def __init__(self, ModelCls=AgentModel, train_noise=0.4, eval_noise=0,
                 expl_type="additive_gaussian", expl_min=0.1, expl_decay=7000,
                 model_kwargs=None, initial_model_state_dict=None):
        self.train_noise = train_noise
        self.eval_noise = eval_noise
        self.expl_type = expl_type
        self.expl_min = expl_min
        self.expl_decay = expl_decay
        super().__init__(ModelCls, model_kwargs, initial_model_state_dict)
        self._mode = 'train'
        self._itr = 0

    def make_env_to_model_kwargs(self, env_spaces):
        """Generate any keyword args to the model which depend on environment interfaces."""
        return dict(action_size=env_spaces.action.shape[0])

    def __call__(self, observation, prev_action, init_rnn_state):
        model_inputs = buffer_to((observation, prev_action, init_rnn_state), device=self.device)
        return self.model(*model_inputs)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """"
        Compute policy's action distribution from inputs, and sample an
        action. Calls the model to produce mean, log_std, value estimate, and
        next recurrent state.  Moves inputs to device and returns outputs back
        to CPU, for the sampler.  Advances the recurrent state of the agent.
        (no grad)
        """
        model_inputs = buffer_to((observation, prev_action), device=self.device)
        action, state = self.model(*model_inputs, self.prev_rnn_state)
        action = self.exploration(action)
        # Model handles None, but Buffer does not, make zeros if needed:
        prev_state = self.prev_rnn_state or buffer_func(state, torch.zeros_like)
        self.advance_rnn_state(state)
        agent_info = DreamerAgentInfo(prev_state=prev_state)
        agent_step = AgentStep(action=action, agent_info=agent_info)
        return buffer_to(agent_step, device='cpu')

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        """
        Compute the value estimate for the environment state using the
        currently held recurrent state, without advancing the recurrent state,
        e.g. for the bootstrap value V(s_{T+1}), in the sampler.  (no grad)
        """
        agent_inputs = buffer_to((observation, prev_action), device=self.device)
        action, action_dist, value, reward, state = self.model(*agent_inputs, self.prev_rnn_state)
        return value.to("cpu")

    def exploration(self, action: torch.Tensor) -> torch.Tensor:
        """
        :param action: action to take, shape (1,) (if categorical), or (action dim,) (if continuous)
        :return: action of the same shape passed in, augmented with some noise
        """
        if self._mode in ['train', 'sample']:
            expl_amount = self.train_noise
            if self.expl_decay:  # Linear decay
                expl_amount = expl_amount - self._itr / self.expl_decay
            if self.expl_min:
                expl_amount = max(self.expl_min, expl_amount)
        elif self._mode == 'eval':
            expl_amount = self.eval_noise
        else:
            raise NotImplementedError

        if self.expl_type == 'additive_gaussian':  # For continuous actions
            noise = torch.randn(*action.shape, device=action.device) * expl_amount
            return torch.clamp(action + noise, -1, 1)
        if self.expl_type == 'completely_random':  # For continuous actions
            if expl_amount == 0:
                return action
            else:
                return torch.rand(*action.shape, device=action.device) * 2 - 1  # scale to [-1, 1]
        if self.expl_type == 'epsilon_greedy':  # For discrete actions
            action_dim = self.env_model_kwargs["action_shape"][0]
            if np.random.uniform(0, 1) < expl_amount:
                index = torch.randint(0, action_dim, action.shape[:-1], device=action.device)
                action = torch.zeros_like(action)
                action[..., index] = 1
            return action
        raise NotImplementedError(self.expl_type)
