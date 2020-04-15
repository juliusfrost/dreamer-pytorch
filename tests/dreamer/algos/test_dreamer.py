# import torch
# import numpy as np
# from dreamer.algos.dreamer_algo import Dreamer
# from dreamer.models.rnns import RSSMState
# from dreamer.agents.dreamer_agent import DreamerAgentInfo, DreamerAgent
# from dreamer.models.agent import AgentModel
# from rlpyt.samplers.collections import Samples, AgentSamples, EnvSamples
# from rlpyt.envs.base import EnvInfo
#
#
# def make_dreamer(action_dim):
#     dreamer = Dreamer()
#     agent = DreamerAgent()
#     agent_model = AgentModel(output_size=action_dim)
#     agent.model = agent_model
#     dreamer.initialize(agent, -1, None, False, None)
#     return dreamer
#
#
# def make_rssm_state(batch_1: int, batch_2: int, stoch_dim: int, deter_dim: int):
#     mean = torch.randn(batch_1, batch_2, stoch_dim)
#     std = torch.randn(batch_1, batch_2, stoch_dim)
#     stoch = torch.randn(batch_1, batch_2, stoch_dim)
#     deter = torch.randn(batch_1, batch_2, deter_dim)
#     state = RSSMState(mean, std, stoch, deter)
#     return state
#
#
# def test_loss():
#     batch_b = 2
#     batch_t = 4
#     stoch_state_dim = 3
#     deter_state_dim = 4
#     action_size = 3
#     img_size = (3, 64, 64)  # TODO: figure out why atari games have 4 channels.
#
#     dreamer = make_dreamer(action_size)
#
#     # categorical action tensor
#     action = torch.randint(action_size, (batch_t, batch_b))
#     prev_action = torch.randn(batch_t, batch_b, action_size)
#     observation = torch.randn(batch_t, batch_b, *img_size)
#     env_reward = torch.randn(batch_t, batch_b, 1)
#     prev_reward = torch.randn(batch_t, batch_b)
#     done = torch.zeros(batch_t, batch_b, dtype=torch.bool)
#     env_info = EnvInfo()
#     prev_state = make_rssm_state(batch_t, batch_b, stoch_state_dim, deter_state_dim)
#     agent_info = DreamerAgentInfo(prev_state=prev_state)
#     agent_samples = AgentSamples(action=action, prev_action=prev_action, agent_info=agent_info)
#     env_samples = EnvSamples(observation=observation, reward=env_reward, prev_reward=prev_reward,
#                              done=done, env_info=env_info)
#     samples = Samples(agent=agent_samples, env=env_samples)
#     loss = dreamer.loss(samples)
#
#     # Check we have a single-element FloatTensor with a gradient
#     assert isinstance(loss, torch.FloatTensor)
#     assert loss.requires_grad
#     assert loss.shape == ()
#
#     # Check it still works if we pass in discrete actions
#     num_actions = 6
#     dreamer = make_dreamer(num_actions)
#     action = torch.randint(0, num_actions, (batch_t, batch_b))
#     prev_action = torch.randint(0, num_actions, (batch_t, batch_b))
#     agent_samples = AgentSamples(action=action, prev_action=prev_action, agent_info=agent_info)
#     env_samples = EnvSamples(observation=observation, reward=env_reward, prev_reward=prev_reward,
#                              done=done, env_info=env_info)
#     samples = Samples(agent=agent_samples, env=env_samples)
#     loss = dreamer.loss(samples)
#     assert isinstance(loss, torch.FloatTensor)
#     assert loss.requires_grad
#     assert loss.shape == ()
#
#
# def test_value_loss():
#     action_dim = 2
#     batch = 2
#     horizon = 2
#     dreamer = make_dreamer(action_dim)
#     vec_dim = dreamer.agent.model.deterministic_size + dreamer.agent.model.stochastic_size
#     imag_feat = torch.randn(horizon + 1, batch, vec_dim)
#     discount = torch.randn(horizon, batch, 1)
#     returns = torch.randn(horizon, batch, 1)
#     value_loss = dreamer.value_loss(imag_feat, discount, returns)
#
#     # Check we have a single-element FloatTensor with a gradient
#     assert isinstance(value_loss, torch.FloatTensor)
#     assert value_loss.requires_grad
#     assert value_loss.shape == ()
#     value_loss = value_loss.item()
#
#     # Check that it's linear in discount
#     assert value_loss * 2 == dreamer.value_loss(imag_feat, discount * 2, returns).item()
#
#     # Check that if you compute it independently for each batch element and each timestep we're just taking the mean.
#     manual_mean_value_loss = np.mean([
#         dreamer.value_loss(imag_feat[[0, 2], :1], discount[:1, :1], returns[:1, :1]).item(),
#         dreamer.value_loss(imag_feat[1:, :1], discount[1:, :1], returns[1:, :1]).item(),
#         dreamer.value_loss(imag_feat[[0, 2], 1:], discount[:1, 1:], returns[:1, 1:]).item(),
#         dreamer.value_loss(imag_feat[1:, 1:], discount[1:, 1:], returns[1:, 1:]).item(),
#     ])
#     assert abs(value_loss - manual_mean_value_loss) < .01, (value_loss, manual_mean_value_loss)
#
#
# def test_actor_loss():
#     action_dim = 1
#     batch = 2
#     horizon = 2
#     vec_dim = 1
#     dreamer = make_dreamer(action_dim)
#     discount = torch.randn(horizon, batch, vec_dim)
#     returns = torch.randn(horizon, batch, vec_dim)
#     returns.requires_grad = True
#     actor_loss = dreamer.actor_loss(discount, returns)
#
#     # Check we have a single-element FloatTensor with a gradient
#     assert isinstance(actor_loss, torch.FloatTensor)
#     assert actor_loss.requires_grad
#     assert actor_loss.shape == ()
#     actor_loss = actor_loss.item()
#
#     # Check that it's linear in both arguments
#     assert actor_loss * 2 == dreamer.actor_loss(discount * 2, returns).item()
#     assert actor_loss * 2 == dreamer.actor_loss(discount, returns * 2).item()
#
#     # Check that if you compute it independently for each batch element and each timestep we're just taking the mean.
#     manual_mean_actor_loss = np.mean([
#         dreamer.actor_loss(discount[:1, :1], returns[:1, :1]).item(),
#         dreamer.actor_loss(discount[1:, :1], returns[1:, :1]).item(),
#         dreamer.actor_loss(discount[:1, 1:], returns[:1, 1:]).item(),
#         dreamer.actor_loss(discount[1:, 1:], returns[1:, 1:]).item(),
#     ])
#     assert abs(actor_loss - manual_mean_actor_loss) < .01, (actor_loss, manual_mean_actor_loss)
#
#
# def test_model_loss():
#     action_dim = 2
#     dreamer = make_dreamer(action_dim)
#     batch = 2
#     horizon = 2
#     stoch_dim = dreamer.agent.model.stochastic_size
#     deter_dim = dreamer.agent.model.deterministic_size
#     img_dim = (3, 64, 64)
#     observation = torch.randn(horizon, batch, *img_dim)
#     prior = make_rssm_state(horizon, batch, stoch_dim, deter_dim)
#     post = make_rssm_state(horizon, batch, stoch_dim, deter_dim)
#     reward = torch.randn(horizon, batch, 1)
#
#     # Check we have a single-element FloatTensor with a gradient
#     model_loss = dreamer.model_loss(observation, prior, post, reward)
#     assert isinstance(model_loss, torch.Tensor)
#     assert model_loss.requires_grad
#     assert model_loss.shape == ()
#     # Check that making prior and post the same decreases the loss
#     model_loss_same = dreamer.model_loss(observation, prior, prior, reward)
#     assert model_loss_same.item() < model_loss.item()
#
#     # Check that if you compute it independently for each batch element and each timestep we're just taking the mean.
#     manual_mean_model_loss = np.mean([
#         dreamer.model_loss(observation[:1, :1], prior[:1, :1], post[:1, :1], reward[:1, :1]).item(),
#         dreamer.model_loss(observation[1:, :1], prior[1:, :1], post[1:, :1], reward[1:, :1]).item(),
#         dreamer.model_loss(observation[:1, 1:], prior[:1, 1:], post[:1, 1:], reward[:1, 1:]).item(),
#         dreamer.model_loss(observation[1:, 1:], prior[1:, 1:], post[1:, 1:], reward[1:, 1:]).item(),
#     ])
#     assert abs(model_loss.item() - manual_mean_model_loss) < .01, (model_loss.item(), manual_mean_model_loss)
#
#
# def test_compute_returns():
#     np.random.seed(0)
#     action_dim = 2
#     horizon = 3
#     batch = 3
#     reward = torch.randn(horizon, batch, 1)
#     value = torch.randn(horizon, batch, 1)
#     discount = torch.randn(horizon, batch, 1)
#     lambda_ = 0.9
#
#     dreamer = make_dreamer(action_dim)
#     returns = dreamer.compute_return(reward[:-1], value[:-1], discount[:-1],
#                                      bootstrap=value[-1], lambda_=lambda_)
#
#     assert isinstance(returns, torch.FloatTensor)
#     assert returns.shape == (horizon - 1, batch, 1)
#
#     # When lambda is 1, the value array does not affect the results at all (other than the every end)
#     lambda_ = 1
#     returns1 = dreamer.compute_return(reward[:-1], value[:-1], discount[:-1],
#                                       bootstrap=value[-1], lambda_=lambda_).data.numpy()
#     returns2 = dreamer.compute_return(reward[:-1], value[:-1] * 100, discount[:-1],
#                                       bootstrap=value[-1], lambda_=lambda_).data.numpy()
#     assert np.array_equal(returns1, returns2)
#
#     # When lambda is 0, the return is the target
#     lambda_ = 0
#     target = reward[:-1] + discount[:-1] * value[1:] * (1 - lambda_)
#     returns = dreamer.compute_return(reward[:-1], value[:-1], discount[:-1],
#                                      bootstrap=value[-1], lambda_=lambda_).data.numpy()
#     assert np.array_equal(returns, target)
#
#     # When the discount is 0, the cumulative reward is just the rewards
#     lambda_ = 0.5
#     discount = torch.zeros(horizon, 1, 1)
#     returns = dreamer.compute_return(reward[:-1], value[:-1], discount[:-1],
#                                      bootstrap=value[-1], lambda_=lambda_).data.numpy()
#     assert np.array_equal(returns, reward[:-1])
#
#     # When the discount is 1 and lambda is 1, the cumulative reward is just the sum of rewards.
#     lambda_ = 1
#     reward = torch.ones(horizon, 1, 1)
#     value = torch.zeros(horizon, 1, 1)
#     discount = torch.ones(horizon, 1, 1)
#     returns = dreamer.compute_return(reward[:-1], value[:-1], discount[:-1],
#                                      bootstrap=value[-1], lambda_=lambda_).data.numpy()
#     expected_returns = np.arange(horizon - 1, 0, -1).reshape((horizon - 1, 1, 1))
#     assert np.array_equal(returns, expected_returns)
