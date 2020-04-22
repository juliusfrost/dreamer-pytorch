import platform

from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.logging.context import logger_context

from dreamer.agents.atari_dreamer_agent import AtariDreamerAgent
from dreamer.algos.dreamer_algo import Dreamer
from dreamer.envs.modified_atari import AtariEnv, AtariTrajInfo
from dreamer.envs.one_hot import OneHotAction
from dreamer.envs.wrapper import make_wapper


def build_and_train(log_dir, game="pong", run_ID=0, cuda_idx=None, eval=False):
    env_kwargs = dict(
        game=game,
        frame_shape=(64, 64),  # dreamer uses this, default is 80, 104
        frame_skip=2,  # because dreamer action repeat = 2
        num_img_obs=1,  # get only the last observation. returns black and white frame
        repeat_action_probability=0.25  # Atari-v0 repeat action probability = 0.25
    )
    factory_method = make_wapper(AtariEnv, [OneHotAction], [{}])
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
    algo = Dreamer(
        batch_size=1,
        batch_length=5,
        train_every=10,
        train_steps=1,
        prefill=10,
        horizon=5,
        replay_size=100,
        log_video=False,
        kl_scale=0.1)
    agent = AtariDreamerAgent(
        train_noise=0.4, eval_noise=0, expl_type="epsilon_greedy", expl_min=0.1, expl_decay=2000/0.3)
    runner_cls = MinibatchRlEval if eval else MinibatchRl
    runner = runner_cls(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=20,
        log_interval_steps=10,
        affinity=dict(cuda_idx=cuda_idx),
    )
    config = dict(game=game)
    name = "dreamer_" + game
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last", override_prefix=True,
                        use_summary_writer=True):
        runner.train()


def test_main():
    if platform.system() == 'Darwin':  # if mac-os
        return
    logdir = 'data/tests/'
    build_and_train(logdir)
