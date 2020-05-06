import argparse
import copy

from babyai.bot import Bot

from counterfactuals import bots
from counterfactuals.dataset import TrajectoryDataset, TrajectoryStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', type=str, required=True,
                        help='Trajectory dataset to continue from (likely from user)')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Save location (useful to specify which bot continued it in the path)')
    parser.add_argument('--bot-name', default='Bot', type=str,
                        help='Bot that continues from the last state in the trajectory')
    args = parser.parse_args()

    dataset = TrajectoryDataset()
    print(f'loading data from {args.load_dir}...')
    dataset.load(args.load_dir)
    env = next(dataset.trajectory_generator()).state[0]

    if hasattr(bots, args.bot_name):
        bot_class = getattr(bots, args.bot_name)
    elif env.bot_name.lower() == "bot":
        bot_class = Bot
    else:
        raise ValueError("Invalid bot name")

    def policy_generator(env):
        return bots.Policy(bot_class, env)

    trajectories = [t for t in dataset.trajectory_generator()]  # get all the trajectories from the generator first
    for t in trajectories:
        episode = t.episode
        env = copy.deepcopy(t.state[-1])
        done = t.done[-1]
        step = t.step[-1]
        policy = policy_generator(env)
        obs = t.observation[-1]
        count = 0
        while not done:
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            next_traj_step = TrajectoryStep(copy.deepcopy(env), obs, action, reward, done, info)
            step += 1
            dataset.append(episode, step, next_traj_step)
            count += 1
        print(f'Took policy {count} more steps to finish.')

    dataset.sort()
    dataset.save(args.save_dir)


if __name__ == '__main__':
    main()
