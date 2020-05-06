import copy

from gym_minigrid.window import Window

from counterfactuals.dataset import TrajectoryDataset, Trajectory, TrajectoryStep


class Navigator:
    def __init__(self, index, min_index, max_index):
        """Used as a simple way to constrain movement through time"""
        self.index = index
        self.min = min_index
        self.max = max_index

    def forward(self):
        if self.index + 1 > self.max:
            raise ValueError(f'Tried to go past the maximum steps {self.max}')
        self.index += 1
        end = self.index == self.max
        return end

    def backward(self):
        if self.index - 1 < self.min:
            raise ValueError(f'Tried to go past the minimum steps {self.min}')
        self.index -= 1
        end = self.index == self.min
        return end

    def step(self):
        raise NotImplementedError


class TrajectoryNavigator(Navigator):
    """Navigates the prior trajectory data"""

    def __init__(self, trajectory: Trajectory):
        self.trajectory = trajectory
        self.index = 0
        self.episode = trajectory.episode
        self._step = trajectory.step
        self._state = trajectory.state
        self._observation = trajectory.observation
        self._action = trajectory.action
        self._reward = trajectory.reward
        self._done = trajectory.done
        self._info = trajectory.info
        super().__init__(0, self._step[0], self._step[-1])

    def step(self):
        state = self._state[self.index]
        obs = self._observation[self.index]
        action = self._action[self.index]
        reward = self._reward[self.index]
        done = self._done[self.index]
        info = self._info[self.index]
        return TrajectoryStep(state, obs, action, reward, done, info)


class CounterfactualNavigator(Navigator):
    """Creates a counterfactual trajectory based on user action. Keeps track of a stack of states"""

    def __init__(self, episode: int, step: int, start: TrajectoryStep, policy_factory=None, max_steps=100):
        self.policy_factory = policy_factory
        self.episode = episode
        self.start = start

        self.stack = [(episode, step, start)]
        super().__init__(step, step, step + max_steps)

    def current(self):
        cur = self.stack[-1]
        episode: int = cur[0]
        step: int = cur[1]
        traj_step: TrajectoryStep = cur[2]
        return episode, step, traj_step

    def step(self):
        return self.current()[2]

    def forward(self, action):
        end = super().forward()
        episode, step, traj_step = self.current()
        if traj_step.done:
            raise ValueError('Trajectory already finished! Cannot perform further actions!')
        env = copy.deepcopy(traj_step.state)
        obs, reward, done, info = env.step(getattr(env.actions, action))
        next_traj_step = TrajectoryStep(env, obs, action, reward, done, info)
        self.stack.append((episode, step + 1, next_traj_step))
        return end or done

    def backward(self):
        end = super().backward()
        self.stack.pop(-1)
        return end

    def rollout(self):
        episode, step, traj_step = self.current()
        done = traj_step.done
        env = copy.deepcopy(traj_step.state)
        policy = self.policy_factory(env)
        obs = traj_step.observation
        count = 0
        while not done:
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            next_traj_step = TrajectoryStep(copy.deepcopy(env), obs, action, reward, done, info)
            step += 1
            self.stack.append((episode, step, next_traj_step))
            count += 1
        print(f'Took policy {count} more steps to finish.')

    def store(self, dataset: TrajectoryDataset):
        if len(self.stack) - 1 == 0:
            raise Exception('Not enough steps in counterfactual trajectory to store! Dataset entry cant be generated!')
        for i in range(1, len(self.stack)):
            episode, step, traj_step = self.stack[i]
            dataset.append(episode, step, traj_step)


class Interface:
    """
    User interface for generating counterfactual states and actions.
    For each trajectory:
        The user can use the `a` and `d` keys to go backward and forward in time to a specific state they wish to
        generate a counterfactual explanation from. Once a specific state is found, the user presses `w` to launch the
        counterfactual mode.
        In the counterfactual mode, the user can control the agent using the keyboard. Once satisfied with their
        progress, the user can give up control to the bot to finish the episode by pressing `w`.

        The keys for controlling the agent are summarized here:

        escape: quit the program

        view mode:
            d: next time step
            a: previous time step
            w: select time step and go to counterfactual mode

        counterfactual mode:
            left:          turn counter-clockwise
            right:         turn clockwise
            up:            go forward
            space:         toggle
            pageup or x:   pickup
            pagedown or z: drop
            enter or q:    done (should not use)
            a:             undo action
            w:             roll out to the end of the episode and move to next episode
    """

    def __init__(self, original_dataset: TrajectoryDataset, counterfactual_dataset: TrajectoryDataset,
                 policy_factory=None):
        self.policy_factory = policy_factory
        self.dataset = original_dataset
        self.counterfactual_dataset = counterfactual_dataset
        self.trajectory_generator = self.dataset.trajectory_generator()
        self.navigator: TrajectoryNavigator = None
        self.window = None
        self.is_counterfactual = False
        self.run()

    def run(self):
        for i, trajectory in enumerate(self.trajectory_generator):
            self.saved = False
            self.is_counterfactual = False
            self.navigator = TrajectoryNavigator(trajectory)
            self.window = Window(f'Trajectory {i}')
            self.window.reg_key_handler(self.key_handler)
            self.reset()
            self.window.show(block=True)
            if not self.saved:
                raise Exception('Continued without saving the trajectory!')

    def redraw(self):
        step: TrajectoryStep = self.navigator.step()
        # if not self.agent_view:
        env = step.state
        img = env.render('rgb_array', tile_size=32)
        # else:
        # img = step.observation['image']
        # TODO later: figure out when to use the observation instead.

        self.window.show_img(img)

    def step(self, action=None):
        if action is None:
            self.navigator.forward()
        else:
            assert isinstance(self.navigator, CounterfactualNavigator)
            self.navigator.forward(action)
        self.redraw()

    def backward(self):
        self.navigator.backward()
        self.redraw()

    def reset(self):
        env = self.navigator.step().state

        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)
            self.window.set_caption(env.mission)

        self.redraw()

    def select(self):
        new_navigator = CounterfactualNavigator(
            self.navigator.episode, self.navigator.index, self.navigator.step(), policy_factory=self.policy_factory)
        self.navigator = new_navigator
        self.is_counterfactual = True
        print(f'Starting counterfactual trajectory from {self.navigator.index}')
        self.redraw()

    def save_trajectory(self):
        assert isinstance(self.navigator, CounterfactualNavigator)
        self.navigator.store(self.counterfactual_dataset)
        self.saved = True

    def key_handler(self, event):
        print('pressed', event.key)

        if event.key == 'escape':
            self.window.close()
            exit()
            return

        # if event.key == 'backspace':
        #     self.reset()
        #     return
        if self.is_counterfactual:
            if event.key == 'left':
                self.step('left')
                return
            if event.key == 'right':
                self.step('right')
                return
            if event.key == 'up':
                self.step('forward')
                return

            # Spacebar
            if event.key == ' ':
                self.step('toggle')
                return
            if event.key == 'pageup' or event.key == 'x':
                self.step('pickup')
                return
            if event.key == 'pagedown' or event.key == 'z':
                self.step('drop')
                return

            if event.key == 'enter' or event.key == 'q':
                self.step('done')
                return

            if event.key == 'w':
                if self.policy_factory is not None:
                    self.navigator.rollout()
                self.save_trajectory()
                self.window.close()

            if event.key == 'a':
                self.backward()
                return

        if not self.is_counterfactual:
            if event.key == 'd':
                self.step()
                return

            if event.key == 'a':
                self.backward()
                return

            if event.key == 'w':
                self.select()
                return
