# Counterfactual Policies -- coming soon
# Dreamer PyTorch

[![tests](https://github.com/juliusfrost/dreamer-pytorch/workflows/tests/badge.svg)](https://github.com/juliusfrost/dreamer-pytorch/actions)
[![codecov](https://codecov.io/gh/juliusfrost/dreamer-pytorch/branch/master/graph/badge.svg?token=DN9RKIRS7C)](https://codecov.io/gh/juliusfrost/dreamer-pytorch)

Dream to Control: Learning Behaviors by Latent Imagination

Paper: https://arxiv.org/abs/1912.01603  
Project Website: https://danijar.com/project/dreamer/   
TensorFlow 2 implementation: https://github.com/danijar/dreamer  
TensorFlow 1 implementation: https://github.com/google-research/dreamer  

## Installation

### Install packages

#### rlpyt

1. Clone the [rlpyt](https://github.com/astooke/rlpyt) framework with
`git clone https://github.com/astooke/rlpyt.git` and copy the `rlpyt` sub-directory into this directory.  
2. Or install directly from github with 
`pip install git+https://github.com/astooke/rlpyt.git`

Other rlpyt requirements
- psutil: `pip install psutil`  
- pyprind: `pip install pyprind`

(you may need --user flag when installing with pip)

Install PyTorch according to their [website](https://pytorch.org/get-started/locally/)

Install gym with `pip install gym`

Install tensorboard with `pip install tensorboard` or install [TensorFlow](https://www.tensorflow.org/install)

To run tests, install pytest: `pip install pytest`

For any other requirements do `pip install -r requirements.txt`

#### atari
```bash
pip install atari_py
```
opencv:  
1. anaconda: https://anaconda.org/conda-forge/opencv
2. pip: `pip install opencv-python`

#### DeepMind Control
Only linux is supported. Follow instructions on [dm_control](https://github.com/deepmind/dm_control)
```bash
pip install mujoco_py
pip install dm_control
```

You must have a [mujoco license](https://www.roboti.us/license.html)

## Running Experiments

To run experiments on Atari, run `python main.py`, and add any extra arguments you would like.
For example, to run with a single gpu set `--cuda-idx 0`.

To run experiments on DeepMind Control, run `python main_dmc.py`. You can also set any extra arguments here.

Experiments will automatically be stored in `data/local/yyyymmdd/run_#`  
You can use tensorboard to keep track of your experiment.
Run `tensorboard --logdir=data`.

If you have trouble reproducing any results, please raise a GitHub issue with your logs and results.
Otherwise, if you have success, please share your trained model weights with us and with the broader community!

## Testing

To run tests:
```bash
pytest tests
```

If you want additional code coverage information:
```bash
pytest tests --cov=dreamer
```

### Code structure
- `main.py` run atari experiment
- `main_dmc.py` run deepmind control experiment 
- `dreamer` dreamer code
  - `agents` agent code used in sampling
    - `atari_dreamer_agent.py` Atari agent
    - `dmc_dreamer_agent.py` DeepMind Control agent
    - `dreamer_agent.py` basic sampling agent, exploration, contains shared methods
  - `algos` algorithm specific code
    - `dreamer_algo.py` optimization algorithm, loss functions, hyperparameters
    - `replay.py` replay buffer
  - `envs` environment specific code
    - `action_repeat.py` action repeat wrapper. ported from tf2 dreamer
    - `atari.py` Atari environments. ported from tf2 dreamer
    - `dmc.py` DeepMind Control Suite environment. ported from tf2 dreamer
    - `env.py` base classes for environment
    - `modified_atari.py` unused atari environment from rlpyt
    - `normalize_actions.py` normalize actions wrapper. ported from tf2 dreamer
    - `one_hot.py` one hot action wrapper. ported from tf2 dreamer
    - `time_limit.py` Time limit wrapper. ported from tf2 dreamer
    - `wrapper.py` Base environment wrapper class
  - `experiments` currently not used
  - `models` all models used in the agent
    - `action.py` Action model
    - `agent.py` Summarizes all models for agent module
    - `dense.py` Dense fully connected models. Used for Reward Model, Value Model, Discount Model.
    - `distribution.py` Distributions, TanH Bijector
    - `observation.py` Observation Model
    - `rnns.py` Recurrent State Space Model
  - `utils` utility functions
    - `logging.py` logging videos
    - `module.py`  freezing parameters


### Contributing
Read `CONTRIBUTING.md`.
Contact juliusf@bu.edu or send in a pull request.
Join our [slack link](https://join.slack.com/t/dreamer-pytorch/shared_invite/zt-dobz7kf6-_tpAv1H9qkk8Ukov1Uy9qQ)
(valid until 5/19/2020)
