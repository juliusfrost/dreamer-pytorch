import torch
import torch.optim as optim
from rlpyt.utils.tensor import to_onehot, infer_leading_dims, restore_leading_dims

from dreamer.models.rnns import get_feat, get_dist, RSSMState

from rlpyt.algos.base import RlAlgorithm
from rlpyt.samplers.collections import Samples
from rlpyt.utils.quick_args import save__init__args


class Dreamer(RlAlgorithm):

    def __init__(
            self,  # Hyper-parameters
            batch_size=50,
            batch_length=50,
            train_every=1000,
            train_steps=100,
            pretrain=100,
            model_lr=6e-4,
            value_lr=8e-5,
            actor_lr=8e-5,
            grad_clip=100.0,
            dataset_balance=False,
            discount=0.99,
            discount_lambda=0.95,
            horizon=15,
            action_dist='tanh_normal',
            action_init_std=5.0,
            expl='additive_gaussian',
            expl_amount=0.3,
            expl_decay=0.0,
            expl_min=0.0,
            optim_kwargs=None,
            free_nats=3,  # TODO: Do we actually need this??
            kl_scale=1,
            type=torch.float,
    ):
        super().__init__()

        self._batch_size = batch_size
        del batch_size  # Property.
        save__init__args(locals())
        self.update_counter = 0
        # dummy model parameters
        self.model_params = (torch.zeros(0, requires_grad=True),)
        self.optimizer = optim.Adam(self.model_params)
        self.type = type

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples, world_size=1, rank=0):
        self.agent = agent

    def async_initialize(self, agent, sampler_n_itr, batch_spec, mid_batch_reset, examples, world_size=1):
        self.agent = agent

    def optim_initialize(self, rank=0):
        pass

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        self.loss(samples)

    def loss(self, samples: Samples):
        """
        Compute the loss for a batch of data.  This includes computing the model and reward losses on the given data,
        as well as using the dynamics model to generate additional rollouts, which are used for the actor and value
        components of the loss.
        :param samples: rlpyt Samples object, containing a batch of data.  All vectors are [timestep, batch, vector_dim]
        :return: FloatTensor containing the loss
        """
        model = self.agent.model

        # Extract tensors from the Samples object
        # They all have the batch_t dimension first, but we'll put the batch_b dimension first.
        # Also, we convert all tensors to floats so they can be fed into our models.

        lead_dim, batch_t, batch_b, img_shape = infer_leading_dims(samples.env.observation, 3)
        # squeeze batch sizes to single batch dimension for imagination roll-out
        batch_size = batch_t * batch_b

        observation = samples.env.observation
        # normalize image
        observation = observation.type(self.type) / 255.0 - 0.5
        # embed the image
        embed = model.observation_encoder(observation)

        # get action
        action = samples.agent.action
        # make actions one-hot
        action = to_onehot(action, model.action_size, dtype=self.type)

        reward = samples.env.reward

        # if we want to continue the the agent state from the previous time steps, we can do it like so:
        # prev_state = samples.agent.agent_info.prev_state[0]
        prev_state = model.representation.initial_state(batch_b)
        # Rollout model by taking the same series of actions as the real model
        post, prior = model.rollout.rollout_representation(batch_t, embed, action, prev_state)
        # Flatten our data (so first dimension is batch_t * batch_b = batch_size)
        # since we're going to do a new rollout starting from each state visited in each batch.
        flat_post = RSSMState(
            mean=post.mean.reshape(batch_size, -1),
            std=post.std.reshape(batch_size, -1),
            stoch=post.stoch.reshape(batch_size, -1),
            deter=post.deter.reshape(batch_size, -1)
        )
        flat_action = action.reshape(batch_size, -1)
        # Rollout the policy for self.horizon steps. Variable names with imag_ indicate this data is imagined not real.
        # imag_feat shape is [horizon, batch_t * batch_b, feature_size]
        imag_dist, _ = model.rollout.rollout_policy(self.horizon, model.policy, flat_action, flat_post)

        # Use state features (deterministic and stochastic) to predict the image and reward
        imag_feat = get_feat(imag_dist)  # [horizon, batch_t * batch_b, feature_size]
        # Assumes these are normal distributions. In the TF code it's be mode, but for a normal distribution mean = mode
        # If we want to use other distributions we'll have to fix this.
        imag_reward = model.reward_model(imag_feat).mean
        value = model.value_model(imag_feat).mean

        # Compute the exponential discounted sum of rewards
        discount_arr = self.discount * torch.ones_like(imag_reward)
        returns = self.compute_return(imag_reward[:-1], value[:-1], discount_arr[:-1],
                                      bootstrap=value[-1], lambda_=self.discount_lambda)
        discount = torch.cumprod(discount_arr[:-1], 1).detach()

        # Compute losses for each component of the model
        model_loss = self.model_loss(observation, prior, post, reward)
        actor_loss = self.actor_loss(discount, returns)
        value_loss = self.value_loss(imag_feat, discount, returns)
        return self.model_lr * model_loss + self.actor_lr * actor_loss + self.value_lr * value_loss

    def model_loss(self, observation: torch.Tensor, prior: RSSMState, post: RSSMState, reward: torch.Tensor):
        """
        Compute the model loss for a bunch of data. All vectors are [batch_t, batch_x, vector_dim]
        """
        model = self.agent.model
        feat = get_feat(post)
        image_pred = model.observation_decoder(feat)
        reward_pred = model.reward_model(feat)
        reward_loss = -torch.mean(reward_pred.log_prob(reward))
        image_loss = -torch.mean(image_pred.log_prob(observation))
        prior_dist = get_dist(prior)
        post_dist = get_dist(post)
        div = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
        div = torch.clamp(div, -float('inf'), self.free_nats)
        model_loss = self.kl_scale * div + reward_loss + image_loss
        return model_loss

    def compute_return(self,
                       reward: torch.Tensor,
                       value: torch.Tensor,
                       discount: torch.Tensor,
                       bootstrap: torch.Tensor,
                       lambda_: float):
        """
        Compute the discounted reward for a batch of data.
        reward, value, and discount are all shape [horizon - 1, batch, 1] (last element is cut off)
        Bootstrap is [batch, 1]
        """
        next_values = torch.cat([value[1:], bootstrap[None]], 0)
        target = reward + discount * next_values * (1 - lambda_)
        timesteps = list(range(reward.shape[0] - 1, -1, -1))
        outputs = []
        accumulated_reward = bootstrap
        for t in timesteps:
            inp = target[t]
            discount_factor = discount[t]
            accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
            outputs.append(accumulated_reward)
        returns = torch.flip(torch.stack(outputs), [0])
        return returns

    def actor_loss(self, discount: torch.Tensor, returns: torch.Tensor):
        """
        Compute loss for the agent/actor model. All vectors are [batch, horizon]
        """
        actor_loss = -torch.mean(discount * returns)
        return actor_loss

    def value_loss(self, imag_feat: torch.Tensor, discount: torch.Tensor, returns: torch.Tensor):
        """
        Compute loss for the value model. All vectors are [batch, horizon, vector_dim]
        """
        value_pred = self.agent.model.value_model(imag_feat[:-1])
        target = returns.detach()
        log_prob = value_pred.log_prob(target)
        value_loss = -torch.mean(discount * log_prob.unsqueeze(2))
        return value_loss
