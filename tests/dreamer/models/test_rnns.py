import torch

from dreamer.models.rnns import RSSMState, RSSMTransition, RSSMRepresentation, RSSMRollout
from dreamer.models.distribution import SampleDist


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
    prev_state = representation_model.initial_state(batch_size)
    prior, posterior = representation_model(obs_embed, prev_action, prev_state)
    assert prior.stoch.size(1) == stochastic_size
    assert prior.deter.size(1) == deterministic_size
    assert posterior.stoch.size(1) == stochastic_size
    assert posterior.deter.size(1) == deterministic_size


def test_rollouts():
    action_size = 10
    obs_embed_size = 100
    stochastic_size = 30
    deterministic_size = 200
    batch_size = 4
    time_steps = 10

    transition_model = RSSMTransition(action_size, stochastic_size, deterministic_size)
    representation_model = RSSMRepresentation(transition_model, obs_embed_size, action_size, stochastic_size,
                                              deterministic_size)

    rollout_module = RSSMRollout(representation_model, transition_model)

    obs_embed: torch.Tensor = torch.randn(time_steps, batch_size, obs_embed_size)
    action: torch.Tensor = torch.randn(time_steps, batch_size, action_size)
    prev_state: RSSMState = representation_model.initial_state(batch_size)

    prior, post = rollout_module(time_steps, obs_embed, action, prev_state)

    assert isinstance(prior, RSSMState)
    assert isinstance(post, RSSMState)
    assert prior.mean.shape == (time_steps, batch_size, stochastic_size)
    assert post.mean.shape == (time_steps, batch_size, stochastic_size)
    assert prior.std.shape == (time_steps, batch_size, stochastic_size)
    assert post.std.shape == (time_steps, batch_size, stochastic_size)
    assert prior.stoch.shape == (time_steps, batch_size, stochastic_size)
    assert post.stoch.shape == (time_steps, batch_size, stochastic_size)
    assert prior.deter.shape == (time_steps, batch_size, deterministic_size)
    assert post.deter.shape == (time_steps, batch_size, deterministic_size)

    prior = rollout_module.rollout_transition(time_steps, action, transition_model.initial_state(batch_size))
    assert isinstance(prior, RSSMState)
    assert prior.mean.shape == (time_steps, batch_size, stochastic_size)
    assert prior.std.shape == (time_steps, batch_size, stochastic_size)
    assert prior.stoch.shape == (time_steps, batch_size, stochastic_size)
    assert prior.deter.shape == (time_steps, batch_size, deterministic_size)

    def policy(state):
        action = torch.randn(state.stoch.size(0), action_size)
        mean = torch.randn(state.stoch.size(0), action_size)
        std = torch.randn(state.stoch.size(0), action_size)
        action_dist = SampleDist(torch.distributions.Normal(mean, std))
        return action, action_dist

    prior, actions = rollout_module.rollout_policy(time_steps, policy, post[-1])
    assert isinstance(prior, RSSMState)
    assert prior.mean.shape == (time_steps, batch_size, stochastic_size)
    assert prior.std.shape == (time_steps, batch_size, stochastic_size)
    assert prior.stoch.shape == (time_steps, batch_size, stochastic_size)
    assert prior.deter.shape == (time_steps, batch_size, deterministic_size)

    assert isinstance(actions, torch.Tensor)
    assert actions.shape == (time_steps, batch_size, action_size)
