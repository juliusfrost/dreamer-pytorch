from collections import namedtuple
from rlpyt.utils.collections import namedarraytuple

EnvInfo = namedtuple('EnvInfo', ['discount', 'game_score', 'traj_done'])
StateObs = namedarraytuple('StateObs', ['image', 'state'])
