from babyai.bot import Bot, ExploreSubgoal
from gym_minigrid.minigrid import MiniGridEnv
import copy
import numpy as np
import random
import torch.nn as nn

"""
File contains bots with certain suboptimal biases.

Each bot takes in a BabyAI env, which for consistency with the original Bot class is called a mission.
"""


class Policy(nn.Module):
    """Wrapper around the bot so it uses the expected policy interface."""
    def __init__(self, bot_class, env):
        super().__init__()
        self.bot_class = bot_class
        self.env = env
        self.reset()

    def forward(self, _):
        """Ignore the current state passed in and return the bot's next suggested action."""
        return self.bot.replan()

    def reset(self):
        """When we reset the environment, we also have to reset the bot."""
        self.bot = self.bot_class(self.env)


class DoubleForwardBot(Bot):
    """
    Bot which repeats its action whenever it steps forward.

    Seems to succeed every time on level GoTo.
    """

    def __init__(self, mission):
        super().__init__(mission)
        self.forward_next_turn = False
        self.past_action = None

    def replan(self, action_taken=None):
        past_action = action_taken if action_taken is not None else self.past_action
        proposed_action = super().replan(past_action)
        # If we went forward last turn, go forward again.
        if self.forward_next_turn:
            action = MiniGridEnv.Actions.forward
            self.forward_next_turn = False
        # Otherwise take the recommended action
        else:
            action = proposed_action
            if action == MiniGridEnv.Actions.forward:
                self.forward_next_turn = True
        self.past_action = action
        return action


class StraightNotRightBot(Bot):
    """
    Bot which can't turn right. If it tries to turn right,
    it will go straight instead.
    It doesn't incorporate this disability into its planner.

    Seems to succeed every time on level GoTo.
    """

    def __init__(self, mission):
        super().__init__(mission)
        self.past_action = None

    def replan(self, action_taken=None):
        past_action = action_taken if action_taken is not None else self.past_action
        proposed_action = super().replan(past_action)
        # If we're about to turn right, go forward instead
        if proposed_action == MiniGridEnv.Actions.right:
            action = MiniGridEnv.Actions.forward
        else:
            action = proposed_action
        self.past_action = action
        return action


class LeftNotRightBot(Bot):
    """
    Bot which can't turn right. If it tries to turn right,
    it will go left instead.
    It doesn't incorporate this disability into its planner.

    Seems to succeed every time on level GoTo.
    """

    def __init__(self, mission):
        super().__init__(mission)
        self.past_action = None

    def replan(self, action_taken=None):
        past_action = action_taken if action_taken is not None else self.past_action
        proposed_action = super().replan(past_action)
        # If we're about to turn right, go left instead
        if proposed_action == MiniGridEnv.Actions.right:
            action = MiniGridEnv.Actions.left
        else:
            action = proposed_action
        self.past_action = action
        return action


class NoMemoryBot(Bot):  # Seems to fail every time
    """
    Bot which restarts its planning algorithm from scratch every timestep.

    Seems to fail every time on level GoTo.
    """

    def __init__(self, mission):
        super().__init__(mission)
        self.mission = mission

    def replan(self, action_taken=None):
        # Create an entirely new bot each time we need to plan
        bot = Bot(self.mission)
        action = bot.replan()
        return action


class EpsilonRandomBot(Bot):
    """
    Bot which takes a random action some random percent of the time.

    Succeeds almost every time on level GoTo.
    """

    def __init__(self, mission, epsilon=0.3):
        super().__init__(mission)
        self.epsilon = epsilon
        self.past_action = None

    def replan(self, action_taken=None):
        past_action = action_taken if action_taken is not None else self.past_action
        action = super().replan(past_action)
        # With probability epsilon, choose an action randomly.  Otherwise to the suggested action.
        if np.random.uniform() < self.epsilon:
            action = random.choice([MiniGridEnv.Actions.left, MiniGridEnv.Actions.right, MiniGridEnv.Actions.forward])
        self.past_action = action
        return action


class RandomBot(Bot):  # Seems to fail every time
    """
    Bot which takes a random action all the time.

    Fails almost every time on all but the simplest environments.
    """

    def __init__(self, mission):
        pass

    def replan(self, action_taken=None):
        action = random.choice(list(MiniGridEnv.Actions))
        return action


class RandomStraightBot(Bot):
    """
    Bot which chooses a random direction and heads in it for between 1 and 4 timesteps.

    Fails almost every time on all but the simplest environments.
    """

    def __init__(self, mission):
        self.curr_action = None
        self.steps_remaining = 0

    def replan(self, action_taken=None):
        # If we're not in the middle of a string of actions, choose an action and how many steps to take it for
        if self.steps_remaining == 0:
            self.curr_action = random.choice(list(MiniGridEnv.Actions))
            self.steps_remaining = random.choice([0, 1, 2, 3])
        else:
            # Otherwise, take the current action again.
            self.steps_remaining -= 1
        return self.curr_action


class GreyBlindBot(Bot):
    """
    Bot which cannot see gray obstacles.
    """
    def __init__(self, mission):
        """Keep a copy of the real mission, which will get updated as the agent takes actions.
            Also keep a modifid mission copy with gray obstacles blanked out. """
        self.original_mission = mission
        mission_copy = self.update_mission()
        self.past_action = None
        super().__init__(mission_copy)

    def update_mission(self):
        """Make a modifid mission copy with gray obstacles blanked out."""
        mission_copy = copy.deepcopy(self.original_mission)
        # Remove gray objects
        for i, item in enumerate(mission_copy.grid.grid):
            if item is not None and item.color and item.color == 'grey' and not item.type == "wall":
                mission_copy.grid.grid[i] = None
        # If the goal is gray, remove that too
        if mission_copy.instrs.desc.color == 'grey':
            mission_copy.instrs.desc.obj_poss = []
            # for i, j in mission_copy.instrs.desc.obj_poss:
            #     mission_copy.instrs.desc.
            #     # check if vis_mask exists (it won't the first timestep)
            #     if hasattr(self, 'vis_mask'):
            #         self.vis_mask[i, j] = False
        return mission_copy

    def replan(self, action_taken=None):
        past_action = action_taken if action_taken is not None else self.past_action
        mission_copy = self.update_mission()
        self.mission = mission_copy
        # Since the goal may sometimes be to find a gray object and we don't
        # have any gray objects, we may get an assertion error that there is
        # nowhere left to explore.  If this happens, reset the bot.
        try:
            action = super().replan(past_action)
        except AssertionError:
            self.__init__(self.original_mission)
            action = self.replan()
        self.past_action = past_action
        return action


class WallBlindBot(Bot):
    """
    Bot which cannot see walls until it is right in front of them.

    Has about 0.27 success rate on GoTo.
    """

    def __init__(self, mission):
        self.original_mission = mission
        self.wall_visibliity = self.initialize_wall_visibility()
        mission_copy = self.update_mission()
        self.past_action = None
        super().__init__(mission_copy)

    def initialize_wall_visibility(self):
        """"Create a list representing the visibility of all items.  Start with everything visible except walls."""
        wall_visibility = []
        for item in self.original_mission.grid.grid:
            if item is None or not item.type == 'wall':
                wall_visibility.append(True)
            else:
                wall_visibility.append(False)
        return wall_visibility

    def update_wall_visibility(self):
        """If the agent is right in front of a wall, make that wall visible."""
        i, j = self.original_mission.front_pos
        index = self.original_mission.grid.grid[j * self.original_mission.width + i]
        self.wall_visibliity[index] = True
        pass

    def update_mission(self):
        """Create a copy of the real mission in with walls with visibility==False are blanked out."""
        mission_copy = copy.deepcopy(self.original_mission)
        for i, (item, visibility) in enumerate(zip(mission_copy.grid.grid, self.wall_visibliity)):
            if item is not None and item.type == 'wall' and visibility is False:
                mission_copy.grid.grid[i] = None
        return mission_copy

    def replan(self, action_taken=None):
        """Update the missiong grid to hide walls, then suggest a new action."""
        past_action = action_taken if action_taken is not None else self.past_action
        mission_copy = self.update_mission()
        self.mission = mission_copy
        action = super().replan(past_action)
        self.past_action = past_action
        return action




class RandomExplorationBot(Bot):
    """
    Bot which explores randomly when its goal is not in sight

    Has about 0.76 success on GoTo.
    """

    def __init__(self, mission):
        super().__init__(mission)
        self.past_action = None

    def replan(self, action_taken=None):
        past_action = action_taken if action_taken is not None else self.past_action
        proposed_action = super().replan(past_action)
        # Stack holds current subtasks.  When the subtask is exploration, choose randomly instead.
        if self.stack and self.stack[-1].is_exploratory():
            action = random.choice([MiniGridEnv.Actions.left, MiniGridEnv.Actions.right, MiniGridEnv.Actions.forward])
        else:
            action = proposed_action
        self.past_action = action
        return action


class MyopicBot(Bot):
    """
    Bot with a field of view of 5 squares (not 7)

    Has 1.0 success rate on GoTo.
    """

    def __init__(self, mission):
        self.mission_copy = copy.deepcopy(mission)
        self.mission_copy.agent_view_size = 5
        self.past_action = None
        super().__init__(self.mission_copy)

    def replan(self, action_taken=None):
        past_action = action_taken if action_taken is not None else self.past_action
        # The mission copy does not update automatically, so we step it here so it's consistent with the real one.
        if past_action is not None:
            self.mission_copy.step(past_action)
        action = super().replan(past_action)
        self.past_action = action
        return action


class ExploreFirstBot(Bot):
    """
    Bot which explores the entire grid before it does anything else.

    Gets about .28 success on GoTo.
    """

    def __init__(self, mission):
        super().__init__(mission)
        self.explored_everything = False
        self.past_action = None

    def replan(self, action_taken=None):
        past_action = action_taken if action_taken is not None else self.past_action
        # Find the action we'd take if we weren't exploring first.
        proposed_action = super().replan(past_action)
        action = None
        if not self.explored_everything:
            try:
                # Copy the real stack so we can replace it unmodified later
                original_stack = self.stack
                self.stack = []
                # Add exploration onto the stack.  We need to do this stack iteration
                # since the ExplorationSubgoal often creates another subgoal which it
                # puts on the stack (e.g. getting to a particular location).
                # This loop should alwyas produce an action unless all states are explored.
                self.stack = [ExploreSubgoal(self)]
                while self.stack:
                    subgoal = self.stack[-1]
                    action = subgoal.replan_before_action()
                    if action is not None:
                        break
                if action is None:
                    raise ValueError("Action should not be none.")
                self.stack = original_stack
            except AssertionError:  # Raised when there's nothing left to explore
                self.explored_everything = True
                self.stack = original_stack
        if action is None:
            action = proposed_action
        self.past_action = past_action
        return action
