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

### Styling

Use PEP8 style for python syntax. (ctrl-alt-l in PyCharm)  


### Contributing
Contact juliusf@bu.edu or send in a pull request.
Join our [slack link](https://join.slack.com/t/dreamer-pytorch/shared_invite/zt-dobz7kf6-_tpAv1H9qkk8Ukov1Uy9qQ)
(valid until 5/19/2020)