import gym
import pytest
from counterfactuals.bots import *
from gym_minigrid.minigrid import MiniGridEnv

# Make sure all the bots run for at least one step
@pytest.mark.parametrize('bot_class', [DoubleForwardBot,
                                       StraightNotRightBot,
                                       LeftNotRightBot,
                                       NoMemoryBot,
                                       EpsilonRandomBot,
                                       RandomBot,
                                       RandomStraightBot,
                                       GreyBlindBot,
                                       WallBlindBot,
                                       RandomExplorationBot,
                                       MyopicBot,
                                       ExploreFirstBot])
def test_bots(bot_class):
    env = gym.make("BabyAI-GoToLocal-v0")
    env.reset()
    bot = bot_class(env)
    action = bot.replan()
    assert action in list(MiniGridEnv.Actions)

def test_policy():
    env = gym.make("BabyAI-GoToLocal-v0")
    state = env.reset()
    bot = DoubleForwardBot(env)
    policy = Policy(DoubleForwardBot, env)
    # The DoubleForwardBot bot is deterministic, so we should get the same response by calling forward
    # and calling the bot's "replan" directly.
    bot_action = bot.replan()
    policy_action = policy(state)
    assert bot_action == policy_action

    # Call reset and confirm we've created a new bot
    old_bot = policy.bot
    assert old_bot is old_bot
    policy.reset()
    new_bot = policy.bot
    assert not old_bot is new_bot
