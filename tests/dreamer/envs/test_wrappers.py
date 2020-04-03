import gym
import wrappers

env = gym.make('AirRaid-ram-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

name = 'AirRaid'
env = wrappers.ActionRepeat(env, 2)
env = wrappers.Atari(
        name, 4, (64, 64), grayscale=False,
         life_done=True, sticky_actions=True)

env = gym.make('Pendulum-v0')
env = wrappers.NormalizeActions(env)
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
