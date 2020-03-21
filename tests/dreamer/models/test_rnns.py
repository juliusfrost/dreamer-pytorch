import torch

from dreamer.models.rnns import RSSMState, RSSMTransition, RSSMRepresentation


def test_rssm():
    action_size = 10
    obs_embed_size = 100
    stochastic_size = 30
    deterministic_size = 200
    batch_size = 4

    transition_model = RSSMTransition(action_size, stochastic_size, deterministic_size)
    representation_model = RSSMRepresentation(transition_model, obs_embed_size, action_size, stochastic_size,
                                              deterministic_size)

    obs_embed: torch.Tensor = torch.randn(batch_size, obs_embed_size)
    prev_action: torch.Tensor = torch.randn(batch_size, action_size)
    prev_state: RSSMState = RSSMState(None, torch.randn(batch_size, stochastic_size), None)
    prior, posterior = representation_model(obs_embed, prev_action, prev_state)
    assert prior.stoch.size(1) == stochastic_size
    assert prior.deter.size(1) == deterministic_size
    assert posterior.stoch.size(1) == stochastic_size
    assert posterior.deter.size(1) == deterministic_size
