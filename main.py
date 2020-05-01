import datetime
import os
import argparse
import torch

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.logging.context import logger_context

from dreamer.agents.atari_dreamer_agent import AtariDreamerAgent
from dreamer.algos.dreamer_algo import Dreamer
from dreamer.envs.atari import AtariEnv, AtariTrajInfo
from dreamer.envs.wrapper import make_wapper
from dreamer.envs.one_hot import OneHotAction
from dreamer.envs.time_limit import TimeLimit
from rlpyt.runners.sync_rl import SyncRl, SyncRlEval

from rlpyt.utils.launching.affinity import make_affinity


def build_and_train(log_dir, game="pong", run_ID=0, num_gpus=1, eval=False, save_model='last', load_model_path=None):
    params = torch.load(load_model_path) if load_model_path else {}
    agent_state_dict = params.get('agent_state_dict')
    optimizer_state_dict = params.get('optimizer_state_dict')

    action_repeat = 2
    env_kwargs = dict(
        name=game,
        action_repeat=action_repeat,
        size=(64, 64),
        grayscale=False,
        life_done=True,
        sticky_actions=True,
    )
    factory_method = make_wapper(
        AtariEnv,
        [OneHotAction, TimeLimit],
        [dict(), dict(duration=1000 / action_repeat)])
    sampler = SerialSampler(
        EnvCls=factory_method,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        env_kwargs=env_kwargs,
        eval_env_kwargs=env_kwargs,
        batch_T=1,
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )
    algo = Dreamer(horizon=10, kl_scale=0.1, use_pcont=True, initial_optim_state_dict=optimizer_state_dict)
    agent = AtariDreamerAgent(train_noise=0.4, eval_noise=0, expl_type="epsilon_greedy",
                              expl_min=0.1, expl_decay=2000 / 0.3, initial_model_state_dict=agent_state_dict,
                              model_kwargs=dict(use_pcont=True))

    runner_cls = SyncRlEval if eval else SyncRl
    affinity = make_affinity(  # TODO: are any of the other args we can pass in here important?
        n_cpu_core=6,  # Use 6 cores across all experiments.
        cpu_per_run=6,
        n_gpu=num_gpus,  # Use num_gpu gpus across all experiments.
        gpu_per_run=num_gpus,  # How many GPUs to parallelize one run across.
    )
    if not isinstance(affinity, list):
        affinity = [affinity]

    runner = runner_cls(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=5e6,
        log_interval_steps=1e3,
        affinity=affinity,
    )
    config = dict(game=game)
    name = "dreamer_" + game
    with logger_context(log_dir, run_ID, name, config, snapshot_mode=save_model, override_prefix=True,
                        use_summary_writer=True):
        runner.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Atari game', default='pong')
    parser.add_argument('--run-ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--num-gpus', help='number of gpu to use; specify which ones using CUDA_VISIBLE_DEVICES=...', type=int, default=None)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save-model', help='save model', type=str, default='last',
                        choices=['all', 'none', 'gap', 'last'])
    parser.add_argument('--load-model-path', help='load model from path', type=str)  # path to params.pkl

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
        num_gpus=args.num_gpus,
        eval=args.eval,
        save_model=args.save_model,
        load_model_path=args.load_model_path
    )
