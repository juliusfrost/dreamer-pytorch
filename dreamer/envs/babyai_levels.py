from babyai.levels.levelgen import RoomGridLevel
from babyai.levels.verifier import GoToInstr, ObjDesc
from gym_minigrid.minigrid import Lava

class Level_GoToLocalAvoidLava(RoomGridLevel):
    """
    Go to an object, inside a single room with no doors, no distractors
    """

    def __init__(self, room_size=8, num_dists=8, num_lava=5, seed=None):
        self.num_dists = num_dists
        self.num_lava = num_lava
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        non_lava_objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        lavas = self.add_lava(num_lava=self.num_lava)
        objs = non_lava_objs + lavas
        self.check_objs_reachable()
        obj = self._rand_elem(non_lava_objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


    def add_lava(self, i=None, j=None, num_lava=1):
        lavas = []
        for _ in range(num_lava):
            # Add the object to a random room if no room specified
            room_i = i
            room_j = j
            if room_i == None:
                room_i = self._rand_int(0, self.num_cols)
            if room_j == None:
                room_j = self._rand_int(0, self.num_rows)

            obj, pos = self.place_in_room(room_i, room_j, Lava())
            lavas.append(obj)
        return lavas
