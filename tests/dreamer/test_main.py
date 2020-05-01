from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
from rlpyt.samplers.serial.sampler import SerialSampler

from dreamer.agents.atari_dreamer_agent import AtariDreamerAgent
from dreamer.algos.dreamer_algo import Dreamer
from dreamer.envs.atari import AtariEnv, AtariTrajInfo
from dreamer.envs.one_hot import OneHotAction
from dreamer.envs.time_limit import TimeLimit
from dreamer.envs.wrapper import make_wapper


def build_and_train(game="pong", run_ID=0, cuda_idx=None, eval=False):
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
    algo = Dreamer(
        batch_size=1,
        batch_length=5,
        train_every=10,
        train_steps=2,
        prefill=10,
        horizon=5,
        replay_size=100,
        log_video=False,
        kl_scale=0.1,
        use_pcont=True,
    )
    agent = AtariDreamerAgent(
        train_noise=0.4, eval_noise=0, expl_type="epsilon_greedy", expl_min=0.1, expl_decay=2000 / 0.3,
        model_kwargs=dict(use_pcont=True))
    runner_cls = MinibatchRlEval if eval else MinibatchRl
    runner = runner_cls(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=20,
        log_interval_steps=10,
        affinity=dict(cuda_idx=cuda_idx),
    )
    runner.train()


def test_main():
    build_and_train()
