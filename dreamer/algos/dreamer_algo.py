import torch
from rlpyt.algos.base import RlAlgorithm
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.tensor import to_onehot, infer_leading_dims
from tqdm import tqdm

from dreamer.algos.replay import initialize_replay_buffer, samples_to_buffer
from dreamer.models.rnns import get_feat, get_dist, RSSMState

LossInfo = namedarraytuple('LossInfo', ['model_loss', 'actor_loss', 'value_loss'])
OptInfo = namedarraytuple("OptInfo", ['loss', 'model_loss', 'actor_loss', 'value_loss'])


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
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            initial_optim_state_dict=None,
            replay_size=int(1e6),
            replay_ratio=8,
            n_step_return=1,
            prioritized_replay=False,  # not implemented yet
            ReplayBufferCls=None,  # Leave None to select by above options.
            updates_per_sync=1,  # For async mode only. (not implemented)
            free_nats=3,
            kl_scale=1,
            type=torch.float,
            prefill=5000,
    ):
        super().__init__()
        if optim_kwargs is None:
            optim_kwargs = {}
        self._batch_size = batch_size
        del batch_size  # Property.
        save__init__args(locals())
        self.update_counter = 0

        self.optimizer = None
        self.learning_rate = model_lr
        self.model_weight = model_lr / self.learning_rate
        self.value_weight = value_lr / self.learning_rate
        self.actor_weight = actor_lr / self.learning_rate
        self.type = type

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples, world_size=1, rank=0):
        self.agent = agent
        self.n_itr = n_itr
        self.batch_spec = batch_spec
        self.mid_batch_reset = mid_batch_reset
        self.replay_buffer = initialize_replay_buffer(self, examples, batch_spec)
        self.optim_initialize(rank)

    def async_initialize(self, agent, sampler_n_itr, batch_spec, mid_batch_reset, examples, world_size=1):
        self.agent = agent
        self.n_itr = sampler_n_itr
        self.batch_spec = batch_spec
        self.mid_batch_reset = mid_batch_reset
        self.replay_buffer = initialize_replay_buffer(self, examples, batch_spec, async_=True)

    def optim_initialize(self, rank=0):
        self.rank = rank
        self.optimizer = self.OptimCls(self.agent.model.parameters(), lr=self.learning_rate, **self.optim_kwargs)
        if self.initial_optim_state_dict is not None:
            self.optimizer.load_state_dict(self.initial_optim_state_dict)

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        itr = itr if sampler_itr is None else sampler_itr
        if samples is not None:
            self.replay_buffer.append_samples(samples_to_buffer(samples))

        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        if itr < self.prefill:
            return opt_info
        if itr % self.train_every != 0:
            return opt_info
        for i in tqdm(range(self.train_steps), desc='Imagination'):
            self.optimizer.zero_grad()
            samples_from_replay = self.replay_buffer.sample_batch(self._batch_size, self.batch_length)
            observation = samples_from_replay.all_observation[:-1]  # [t, t+batch_length+1] -> [t, t+batch_length]
            action = samples_from_replay.all_action[1:]  # [t-1, t+batch_length] -> [t, t+batch_length]
            reward = samples_from_replay.all_reward[1:]  # [t-1, t+batch_length] -> [t, t+batch_length]
            loss_inputs = buffer_to((observation, action, reward), self.agent.device)
            loss, loss_info = self.loss(*loss_inputs)
            loss.backward()
            self.optimizer.step()
            opt_info.loss.append(loss.item())
            opt_info.model_loss.append(loss_info.model_loss.item())
            opt_info.actor_loss.append(loss_info.actor_loss.item())
            opt_info.value_loss.append(loss_info.value_loss.item())

        return opt_info

    def loss(self, observation, action, reward):
        """
        Compute the loss for a batch of data.  This includes computing the model and reward losses on the given data,
        as well as using the dynamics model to generate additional rollouts, which are used for the actor and value
        components of the loss.
        :param observation: size(batch_t, batch_b, c, h ,w)
        :param action: size(batch_t, batch_b) if categorical
        :param reward: size(batch_t, batch_b, 1)
        :return: FloatTensor containing the loss
        """
        model = self.agent.model

        # Extract tensors from the Samples object
        # They all have the batch_t dimension first, but we'll put the batch_b dimension first.
        # Also, we convert all tensors to floats so they can be fed into our models.

        lead_dim, batch_t, batch_b, img_shape = infer_leading_dims(observation, 3)
        # squeeze batch sizes to single batch dimension for imagination roll-out
        batch_size = batch_t * batch_b

        # normalize image
        observation = observation.type(self.type) / 255.0 - 0.5
        # embed the image
        embed = model.observation_encoder(observation)

        # if we want to continue the the agent state from the previous time steps, we can do it like so:
        # prev_state = samples.agent.agent_info.prev_state[0]
        prev_state = model.representation.initial_state(batch_b, device=action.device, dtype=action.dtype)
        # Rollout model by taking the same series of actions as the real model
        post, prior = model.rollout.rollout_representation(batch_t, embed, action, prev_state)
        # Flatten our data (so first dimension is batch_t * batch_b = batch_size)
        # since we're going to do a new rollout starting from each state visited in each batch.

        # detach gradient here since the actor and value gradients do not need to propagate through representation
        flat_post = buffer_method(post, 'reshape', batch_size, -1)
        flat_post = buffer_method(flat_post, 'detach')
        flat_action = action.reshape(batch_size, -1).detach()
        # Rollout the policy for self.horizon steps. Variable names with imag_ indicate this data is imagined not real.
        # imag_feat shape is [horizon, batch_t * batch_b, feature_size]
        imag_dist, _ = model.rollout.rollout_policy(self.horizon, model.policy, flat_action, flat_post)

        # Use state features (deterministic and stochastic) to predict the image and reward
        imag_feat = get_feat(imag_dist)  # [horizon, batch_t * batch_b, feature_size]
        # Assumes these are normal distributions. In the TF code it's be mode, but for a normal distribution mean = mode
        # If we want to use other distributions we'll have to fix this.
        # We calculate the target here so no grad necessary
        imag_reward = model.reward_model(imag_feat).mean
        value = model.value_model(imag_feat).mean

        # Compute the exponential discounted sum of rewards
        discount_arr = self.discount * torch.ones_like(imag_reward)
        returns = self.compute_return(imag_reward[:-1], value[:-1], discount_arr[:-1],
                                      bootstrap=value[-1], lambda_=self.discount_lambda)
        discount = torch.cumprod(discount_arr[:-1], 1)

        # Compute losses for each component of the model
        model_loss = self.model_loss(observation, prior, post, reward)
        actor_loss = self.actor_loss(discount, returns)
        value_loss = self.value_loss(imag_feat, discount, returns)
        loss = self.model_weight * model_loss + self.actor_weight * actor_loss + self.value_weight * value_loss
        loss_info = LossInfo(model_loss, actor_loss, value_loss)
        return loss, loss_info

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
        target = returns.detach()  # stop gradients here
        log_prob = value_pred.log_prob(target)
        value_loss = -torch.mean(discount * log_prob.unsqueeze(2))
        return value_loss
