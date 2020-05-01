import datetime
import os
import argparse
import torch

from rlpyt.samplers.collections import TrajInfo
from rlpyt.runners.sync_rl import SyncRl, SyncRlEval
from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.affinity import make_affinity

from dreamer.agents.dmc_dreamer_agent import DMCDreamerAgent
from dreamer.algos.dreamer_algo import Dreamer
from dreamer.envs.dmc import DeepMindControl
from dreamer.envs.time_limit import TimeLimit
from dreamer.envs.action_repeat import ActionRepeat
from dreamer.envs.normalize_actions import NormalizeActions
from dreamer.envs.wrapper import make_wapper


def build_and_train(log_dir, game="cartpole_balance", run_ID=0, num_gpus=None, num_cpus=6,
                    eval=False, save_model='last', load_model_path=None):
    params = torch.load(load_model_path) if load_model_path else {}
    agent_state_dict = params.get('agent_state_dict')
    optimizer_state_dict = params.get('optimizer_state_dict')

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
    algo = Dreamer(initial_optim_state_dict=optimizer_state_dict)  # Run with defaults.
    agent = DMCDreamerAgent(train_noise=0.3, eval_noise=0, expl_type="additive_gaussian",
                              expl_min=None, expl_decay=None, initial_model_state_dict=agent_state_dict)
    runner_cls = SyncRlEval if eval else SyncRl
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
        print("Using %i GPUs" % num_gpus)
    affinity = make_affinity(  # TODO: are any of the other args we can pass in here important?
        n_cpu_core=args.num_cpus,  # Use 6 cores across all experiments.
        n_gpu=num_gpus,  # Use num_gpu gpus across all experiments.
        gpu_per_run=num_gpus,  # How many GPUs to parallelize one run across.
        # async_sample=True,  # True if asynchronous sampling / optimization.
        # sample_gpu_per_run=1,
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
    parser.add_argument('--game', help='DMC game', default='cartpole_balance')
    parser.add_argument('--run-ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--num-gpus', help='number of gpu to use; specify which ones using CUDA_VISIBLE_DEVICES=...', type=int, default=None)
    parser.add_argument('--num-cpus', help='number of cpu cores to use', type=int, default=6)
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
        num_cpus=args.num_cpus,
        eval=args.eval,
        save_model=args.save_model,
        load_model_path=args.load_model_path,
    )
