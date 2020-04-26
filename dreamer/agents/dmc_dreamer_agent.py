from dreamer.agents.dreamer_agent import DreamerAgent
from dreamer.models.agent import AtariDreamerModel


class DMCDreamerAgent(DreamerAgent):

    def __init__(self, ModelCls=AtariDreamerModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

    def make_env_to_model_kwargs(self, env_spaces):
        if isinstance(env_spaces.observation, tuple):
            img_obs_shape = env_spaces.observation.image.shape
            state_obs_shape = env_spaces.observation.state.shape[0]
            return dict(image_shape=img_obs_shape,
                        state_size=state_obs_shape,
                        output_size=env_spaces.action.shape[0],
                        action_shape=env_spaces.action.shape[0],
                        action_dist='tanh_normal')
        else:
            return dict(image_shape=env_spaces.observation.shape,
                        output_size=env_spaces.action.shape[0],
                        action_shape=env_spaces.action.shape[0],
                        action_dist='tanh_normal')
