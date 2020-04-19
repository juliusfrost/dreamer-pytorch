from dreamer.agents.dreamer_agent import DreamerAgent
from dreamer.models.agent import AtariDreamerModel


class AtariDreamerAgent(DreamerAgent):

    def __init__(self, ModelCls=AtariDreamerModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(image_shape=env_spaces.observation.shape,
                    action_shape=env_spaces.action.shape,
                    action_dist='one_hot')
