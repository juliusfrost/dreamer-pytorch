from dreamer.agents.dreamer_agent import DreamerAgent
from dreamer.models.agent import MinigridDreamerModel


class MinigridDreamerAgent(DreamerAgent):

    def __init__(self, ModelCls=MinigridDreamerModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(action_shape=env_spaces.action.shape,
                    image_shape=env_spaces.observation.image.shape,
                    state_size=env_spaces.observation.state.shape,
                    )
