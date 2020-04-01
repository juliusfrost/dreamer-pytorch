import torch

from dreamer.models.rnns import RSSMState, RSSMTransition, RSSMRepresentation, RSSMRollout


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

    obs_embed: torch.Tensor = torch.randn(batch_size, time_steps, obs_embed_size)
    action: torch.Tensor = torch.randn(batch_size, time_steps, action_size)
    prev_state: RSSMState = representation_model.initial_state(batch_size)

    prior, post = rollout_module(time_steps, obs_embed, action, prev_state)

    assert isinstance(prior, RSSMState)
    assert isinstance(post, RSSMState)
    assert prior.mean.shape == (batch_size, time_steps, stochastic_size)
    assert post.mean.shape == (batch_size, time_steps, stochastic_size)
    assert prior.std.shape == (batch_size, time_steps, stochastic_size)
    assert post.std.shape == (batch_size, time_steps, stochastic_size)
    assert prior.stoch.shape == (batch_size, time_steps, stochastic_size)
    assert post.stoch.shape == (batch_size, time_steps, stochastic_size)
    assert prior.deter.shape == (batch_size, time_steps, deterministic_size)
    assert post.deter.shape == (batch_size, time_steps, deterministic_size)

    prior = rollout_module.rollout_transition(time_steps, action, transition_model.initial_state(batch_size))
    assert isinstance(post, RSSMState)
    assert prior.mean.shape == (batch_size, time_steps, stochastic_size)
    assert prior.std.shape == (batch_size, time_steps, stochastic_size)
    assert prior.stoch.shape == (batch_size, time_steps, stochastic_size)
    assert prior.deter.shape == (batch_size, time_steps, deterministic_size)

    def policy(state):
        return torch.randn(state.stoch.size(0), action_size)

    prev_action = torch.randn(batch_size, action_size)
    prior, actions = rollout_module.rollout_policy(time_steps, policy, prev_action,
                                                   transition_model.initial_state(batch_size))
    assert isinstance(prior, RSSMState)
    assert prior.mean.shape == (batch_size, time_steps, stochastic_size)
    assert prior.std.shape == (batch_size, time_steps, stochastic_size)
    assert prior.stoch.shape == (batch_size, time_steps, stochastic_size)
    assert prior.deter.shape == (batch_size, time_steps, deterministic_size)

    assert isinstance(actions, torch.Tensor)
    assert actions.shape == (batch_size, time_steps, action_size)