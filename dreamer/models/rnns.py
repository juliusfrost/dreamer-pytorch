import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as tf
from rlpyt.utils.collections import namedarraytuple

RSSMState = namedarraytuple('RSSMState', ['dist', 'stoch', 'deter'])


class TransitionBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prev_action, prev_state):
        """:return: next state"""
        raise NotImplementedError


class RepresentationBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, obs_embed, prev_action, prev_state):
        """:return: next state"""
        raise NotImplementedError


class RollOutModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, state, action, obs_embed):
        raise NotImplementedError


class RSSMTransition(TransitionBase):
    def __init__(self, action_size, stochastic_size=30, deterministic_size=200, hidden_size=200, activation=nn.ELU,
                 distribution=td.Normal):
        super().__init__()
        self._action_size = action_size
        self._stoch_size = stochastic_size
        self._deter_size = deterministic_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._cell = nn.GRUCell(hidden_size, deterministic_size)
        self._rnn_input_model = self._build_rnn_input_model()
        self._stochastic_prior_model = self._build_stochastic_model()
        self._dist = distribution

    def _build_rnn_input_model(self):
        rnn_input_model = [nn.Linear(self._action_size + self._stoch_size, self._hidden_size)]
        rnn_input_model += [self._activation()]
        return nn.Sequential(*rnn_input_model)

    def _build_stochastic_model(self):
        stochastic_model = [nn.Linear(self._hidden_size, self._hidden_size)]
        stochastic_model += [self._activation()]
        stochastic_model += [nn.Linear(self._hidden_size, 2 * self._stoch_size)]
        return nn.Sequential(*stochastic_model)

    def forward(self, prev_action: torch.Tensor, prev_state: RSSMState):
        rnn_input = self._rnn_input_model(torch.cat([prev_action, prev_state.stoch], dim=-1))
        deter_state = self._cell(rnn_input, prev_state.deter)
        mean, std = torch.chunk(self._stochastic_prior_model(deter_state), 2, dim=-1)
        std = tf.softplus(std) + 0.1
        dist = self._dist(mean, std)
        stoch_state = dist.rsample()
        return RSSMState(dist, stoch_state, deter_state)


class RSSMRepresentation(RepresentationBase):
    def __init__(self, transition_model: RSSMTransition, obs_embed_size, action_size, stochastic_size=30,
                 deterministic_size=200, hidden_size=200, activation=nn.ELU, distribution=td.Normal):
        super().__init__()
        self._transition_model = transition_model
        self._obs_embed_size = obs_embed_size
        self._action_size = action_size
        self._stoch_size = stochastic_size
        self._deter_size = deterministic_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._dist = distribution
        self._stochastic_posterior_model = self._build_stochastic_model()

    def _build_stochastic_model(self):
        stochastic_model = [nn.Linear(self._deter_size + self._obs_embed_size, self._hidden_size)]
        stochastic_model += [self._activation()]
        stochastic_model += [nn.Linear(self._hidden_size, 2 * self._stoch_size)]
        return nn.Sequential(*stochastic_model)

    def forward(self, obs_embed: torch.Tensor, prev_action: torch.Tensor, prev_state: RSSMState):
        prior_state = self._transition_model(prev_action, prev_state)
        x = torch.cat([prior_state.deter, obs_embed], -1)
        mean, std = torch.chunk(self._stochastic_posterior_model(x), 2, dim=-1)
        std = tf.softplus(std) + 0.1
        dist = self._dist(mean, std)
        stoch_state = dist.rsample()
        posterior_state = RSSMState(dist, stoch_state, prior_state.deter)
        return prior_state, posterior_state
