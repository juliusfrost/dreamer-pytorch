import torch
import torch.nn as nn

from dreamer.models.rnns import RSSMState, RSSMRepresentation, RSSMTransition, RSSMRollout, get_feat
from dreamer.models.observation import ObservationDecoder, ObservationEncoder
from dreamer.models.action import ActionDecoder


class AgentModel(nn.Module):
    def __init__(
            self,
            action_size,
            stochastic_size=30,
            deterministic_size=200,
            hidden_size=200,
            obs_embed_size=1024,
            obs_shape=(3, 64, 64),
            action_hidden_size=200,
            action_layers=3,
            action_dist='tanh_normal',
    ):
        super().__init__()
        self.transition = RSSMTransition(action_size, stochastic_size, deterministic_size, hidden_size)
        self.representation = RSSMRepresentation(self.transition, obs_embed_size, action_size, stochastic_size,
                                                 deterministic_size, hidden_size)
        self.rollout = RSSMRollout(self.representation, self.transition)
        self.observation_decoder = ObservationDecoder(embed_size=obs_embed_size, shape=obs_shape)
        self.observation_encoder = ObservationEncoder()
        feature_size = stochastic_size + deterministic_size
        self.action_dist = action_dist
        self.action_decoder = ActionDecoder(action_size, feature_size, action_hidden_size, action_layers, action_dist)

    def forward(self, observation: torch.Tensor, prev_action: torch.Tensor, prev_state: RSSMState = None):
        obs_embed = self.observation_encoder(observation)
        if prev_state is None:
            prev_state = self.representation.initial_state(prev_action.size(0), device=prev_action.device,
                                                           dtype=prev_action.dtype)
        _, state = self.representation(obs_embed, prev_action, prev_state)
        action = self.policy(state)
        return action

    def policy(self, state: RSSMState):
        feat = get_feat(state)
        action_dist = self.action_decoder(feat)
        if self.action_dist == 'tanh_normal':
            if self.training:  # use agent.train(bool) or agent.eval()
                action = action_dist.mean()
            else:
                action = action_dist.mode()
        else:
            action = action_dist.rsample()
        return action

    def get_state(self, observation, prev_action, prev_state):
        state = None
        return state
