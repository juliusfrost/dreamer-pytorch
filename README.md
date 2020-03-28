# Dreamer PyTorch

[![tests](https://github.com/juliusfrost/dreamer-pytorch/workflows/tests/badge.svg)](https://github.com/juliusfrost/dreamer-pytorch/actions)
[![codecov](https://codecov.io/gh/juliusfrost/dreamer-pytorch/graph/badge.svg)](https://codecov.io/gh/juliusfrost/dreamer-pytorch)

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


(you may need --user flag when installing with pip)

To run tests, install pytest: `pip install pytest`

Install PyTorch according to their [website](https://pytorch.org/get-started/locally/)

Install gym with `pip install gym`

#### atari
```bash
pip install atari_py
```
opencv:  
1. anaconda: https://anaconda.org/conda-forge/opencv
2. pip: `pip install opencv-python`

psutil: `pip install psutil`  
pyprind: `pip install pyprind`

#### mujoco
```bash
pip install mujoco_py
```
You must have a [mujoco license](https://www.roboti.us/license.html)


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
