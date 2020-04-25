import datetime
import os

from rlpyt.samplers.collections import TrajInfo
from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.logging.context import logger_context

from dreamer.agents.dmc_dreamer_agent import DMCDreamerAgent
from dreamer.algos.dreamer_algo import Dreamer
from dreamer.envs.dmc import DeepMindControl
from dreamer.envs.time_limit import TimeLimit
from dreamer.envs.action_repeat import ActionRepeat
from dreamer.envs.normalize_actions import NormalizeActions
from dreamer.envs.wrapper import make_wapper


def build_and_train(log_dir, game="cartpole_balance", run_ID=0, cuda_idx=None, eval=False):
    action_repeat = 2
    factory_method = make_wapper(
        DeepMindControl,
        [ActionRepeat, NormalizeActions, TimeLimit],
        [dict(amount=action_repeat), dict(), dict(duration=1000 / action_repeat)])
    sampler = SerialSampler(
        EnvCls=factory_method,
        TrajInfoCls=TrajInfo,
        env_kwargs=dict(name=game),
        eval_env_kwargs=dict(name=game),
        batch_T=1,
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )
    algo = Dreamer()  # Run with defaults.
    agent = DMCDreamerAgent(train_noise=0.3, eval_noise=0, expl_type="additive_gaussian",
                              expl_min=None, expl_decay=None)
    runner_cls = MinibatchRlEval if eval else MinibatchRl
    runner = runner_cls(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=5e6,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx),
    )
    config = dict(game=game)
    name = "dreamer_" + game
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last", override_prefix=True,
                        use_summary_writer=True):
        runner.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='DMC game', default='cartpole_balance')
    parser.add_argument('--run-ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda-idx', help='gpu to use ', type=int, default=None)
    default_log_dir = os.path.join(
        os.path.dirname(__file__),
        'data',
        'local',
        datetime.datetime.now().strftime("%Y%m%d"))
    parser.add_argument('--log-dir', type=str, default=default_log_dir)
    args = parser.parse_args()
    log_dir = os.path.abspath(args.log_dir)
    i = args.run_ID
    while os.path.exists(os.path.join(log_dir, 'run_' + str(i))):
        print(f'run {i} already exists. ')
        i += 1
    print(f'Using run id = {i}')
    args.run_ID = i
    build_and_train(
        log_dir,
        game=args.game,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )
