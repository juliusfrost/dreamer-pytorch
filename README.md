# Dreamer PyTorch

Dream to Control: Learning Behaviors by Latent Imagination

Paper: https://arxiv.org/abs/1912.01603  
Project Website: https://danijar.com/project/dreamer/   
TensorFlow 2 implementation: https://github.com/danijar/dreamer  
TensorFlow 1 implementation: https://github.com/google-research/dreamer  

## Installation

Within this directory, clone the [rlpyt](https://github.com/astooke/rlpyt) framework with
`git clone https://github.com/astooke/rlpyt.git`  
Or install directly from github with 
```bash
pip install git+https://github.com/astooke/rlpyt.git
```


### Install packages

(you may need --user flag when installing with pip)

To run tests, install pytest: `pip install pytest`

[Install PyTorch according to their website](https://pytorch.org/get-started/locally/)

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
