import argparse
import copy
import os

import gym
from tqdm import tqdm

from counterfactuals.dataset import TrajectoryDataset, TrajectoryStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', default=os.path.join('..', 'data', 'pretrained'))
    parser.add_argument('--episodes', default=1000)
    parser.add_argument('--env-name', default='babyai')  # TODO: fix with proper name
    args = parser.parse_args()

    env = gym.make(args.env_name)
    dataset = TrajectoryDataset(args.save_dir)

    policy = None  # TODO: implement

    for episode in tqdm(range(args.episodes)):
        obs = env.reset()
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
