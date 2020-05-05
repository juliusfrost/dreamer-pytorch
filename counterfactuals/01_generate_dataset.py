import argparse
import copy
import os

import gym
from babyai.bot import Bot  # in case we want to use the oracle to generate optimal trajectories
from tqdm import tqdm

from counterfactuals import bots
from counterfactuals.dataset import TrajectoryDataset, TrajectoryStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', default=os.path.join('..', 'data', 'pretrained'))
    parser.add_argument('--episodes', default=1000, type=int)
    parser.add_argument('--env-name', default='BabyAI-GoToLocal-v0', type=str)
    parser.add_argument('--bot-name', default='RandomExplorationBot', type=str)
    parser.add_argument('--max-steps', default=50)
    args = parser.parse_args()

    env = gym.make(args.env_name, room_size=15)
    dataset = TrajectoryDataset(args.save_dir)

    if hasattr(bots, args.bot_name):
        bot_class = getattr(bots, args.bot_name)
    elif env.bot_name.lower() == "bot":
        bot_class = Bot
    else:
        raise ValueError("Invalid bot name")
    policy = bots.Policy(bot_class, env)

    for episode in tqdm(range(args.episodes)):
        obs = env.reset()
        if hasattr(env, 'max_steps'):
            env.max_steps = args.max_steps
        if hasattr(policy, "reset"):
            policy.reset()
        done = False
        step = 0
        while not done:
            action = policy(obs)
            next_obs, reward, done, info = env.step(action)
            dataset.append(episode, step,
                           TrajectoryStep(copy.deepcopy(env), obs, action, reward, done, info))
            obs = next_obs
            step += 1

    dataset.save()


if __name__ == '__main__':
    main()
