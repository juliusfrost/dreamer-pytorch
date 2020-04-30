from dreamer.agents.dreamer_agent import DreamerAgent
from dreamer.models.agent import AgentModel


class ContinuousDreamerAgent(DreamerAgent):

    def __init__(self, ModelCls=AgentModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(image_shape=env_spaces.observation.shape,
                    action_shape=env_spaces.action.shape[0],
                    action_dist='tanh_normal',
                    obs_low=env_spaces.observation.low,
                    obs_high=env_spaces.observation.high,)
