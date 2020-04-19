# import torch
#
# from dreamer.agents.atari_dreamer_agent import AtariDreamerAgent
# from rlpyt.envs.atari.atari_env import AtariEnv
#
# def test_exploration():
#     # Test with a categorical variable and epsilon_greedy
#
#     # No noise
#     agent = AtariDreamerAgent(train_noise=0, eval_noise=0.5, expl_type="epsilon_greedy", expl_min=0)
#     agent.initialize(AtariEnv().spaces)
#     agent.sample_mode(0)
#     action = torch.tensor([-1])  # Not a valid action, but it will let us check the action has been changed.
#     new_action = agent.exploration(action)
#     assert new_action.shape == action.shape
#     assert new_action.item() == -1
#
#     # Decaying noise
#     agent = AtariDreamerAgent(train_noise=1, eval_noise=0.5, expl_type="epsilon_greedy", expl_decay=5, expl_min=0)
#     agent.initialize(AtariEnv().spaces)
#     agent.sample_mode(0)
#     action = torch.tensor([-1])  # Not a valid action, but it will let us check the action has been changed.
#     new_action = agent.exploration(action)
#     assert new_action.item() > -1
#
#     agent.sample_mode(10)
#     new_action = agent.exploration(action)  # We'd expect no exploration now
#     assert new_action.item() == -1
#
#     agent = AtariDreamerAgent(train_noise=1, eval_noise=0.5, expl_type="epsilon_greedy", expl_decay=5, expl_min=1)
#     agent.initialize(AtariEnv().spaces)
#     agent.sample_mode(10)
#     new_action = agent.exploration(action)  # If expl_min is high, we expect high exploration even after many steps
#     assert new_action.item() > -1
#
#     # Different modes
#     agent = AtariDreamerAgent(train_noise=1, eval_noise=0, expl_type="epsilon_greedy", expl_min=0)
#     agent.initialize(AtariEnv().spaces)
#     agent.train_mode(0)
#     new_action = agent.exploration(action)
#     assert new_action.item() > -1
#
#     agent.eval_mode(0)
#     new_action = agent.exploration(action)
#     assert new_action.item() == -1
#
#     agent = AtariDreamerAgent(train_noise=0, eval_noise=1, expl_type="epsilon_greedy", expl_min=0)
#     agent.initialize(AtariEnv().spaces)
#     agent.train_mode(0)
#     new_action = agent.exploration(action)
#     assert new_action.item() == -1
#
#     agent.eval_mode(0)
#     new_action = agent.exploration(action)
#     assert new_action.item() > -1
#
#     # 0-dimensional action
#     zero_dim_action = torch.tensor(-2)
#     new_action = agent.exploration(zero_dim_action)
#     assert new_action.shape == zero_dim_action.shape
#     assert new_action.item() > -2
#
#     # Test with a continuous variable (completely random noise)
#     agent = AtariDreamerAgent(train_noise=0.5, eval_noise=0, expl_type="completely_random", expl_min=0)
#     agent.initialize(AtariEnv().spaces)
#     agent.train_mode(0)
#     action = torch.randn(100)
#     new_action = agent.exploration(action)
#     assert new_action.shape == action.shape
#     assert torch.min(new_action).item() >= -1
#     assert torch.max(new_action).item() <= 1
#     assert not torch.all(torch.eq(new_action, action))
#
#     agent = AtariDreamerAgent(train_noise=0, eval_noise=0, expl_type="completely_random", expl_min=0)
#     agent.initialize(AtariEnv().spaces)
#     agent.train_mode(0)
#     new_action = agent.exploration(action)
#     assert torch.all(torch.eq(new_action, action))
#
#     # Test with a continuous variable (additive gaussian)
#     agent = AtariDreamerAgent(train_noise=0, expl_type="additive_gaussian", expl_min=0)
#     agent.initialize(AtariEnv().spaces)
#     agent.train_mode(0)
#     action = torch.rand(100)
#     new_action = agent.exploration(action)
#     assert new_action.shape == action.shape
#     assert torch.all(torch.eq(new_action, action))
#
#     agent = AtariDreamerAgent(train_noise=1, expl_type="additive_gaussian", expl_min=0)
#     agent.initialize(AtariEnv().spaces)
#     agent.train_mode(0)
#     action = torch.randn(100)
#     new_action = agent.exploration(action)
#     assert not torch.all(torch.eq(new_action, action))
#     assert torch.min(new_action).item() >= -1
#     assert torch.max(new_action).item() <= 1
#
