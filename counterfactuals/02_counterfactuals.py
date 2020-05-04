import argparse
import os

from babyai.bot import Bot

from counterfactuals import bots
from counterfactuals.dataset import TrajectoryDataset
from counterfactuals.user import Interface


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', default=os.path.join('..', 'data', 'pretrained'))
    parser.add_argument('--save-dir', default=os.path.join('..', 'data', 'counterfactual'))
    parser.add_argument('--bot-name', default='RandomExplorationBot', type=str)
    args = parser.parse_args()

    dataset = TrajectoryDataset(args.load_dir)
    print(f'loading data from {args.load_dir}...')
    dataset.load()
    env = next(dataset.trajectory_generator()).state[0]

    if hasattr(bots, args.bot_name):
        bot_class = getattr(bots, args.bot_name)
    elif env.bot_name.lower() == "bot":
        bot_class = Bot
    else:
        raise ValueError("Invalid bot name")

    def policy_generator(env):
        return bots.Policy(bot_class, env)

    counterfactual_dataset = TrajectoryDataset(args.save_dir)

    print('starting user interface...')
    Interface(policy_generator, dataset, counterfactual_dataset)

    counterfactual_dataset.save()


if __name__ == '__main__':
    main()
