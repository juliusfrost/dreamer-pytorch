import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dreamer.models.observation import observe

def _train(self, data, log_images):
    #comes from _build_model function
    embed = self._encode(data)
    #self._dynamics defined in _build_model as a class
    post, prior = self._dynamics.observe(embed, data['action'])
    feat = self._dynamics.get_feat(post)
    #_decode is defined in _build_model function
    image_pred = self._decode(feat)
    #_reward is defined in _build_model function
    reward_pred = self._reward(feat)
    #I do not know what tools.AttrDict() does. I need to come back to it.
    likes = tools.AttrDict()
    likes.image = tf.reduce_mean(image_pred.log_prob(data['image']))
        likes.reward = tf.reduce_mean(reward_pred.log_prob(data['reward']))
        if self._c.pcont:
            pcont_pred = self._pcont(feat)
            pcont_target = self._c.discount * data['discount']
            likes.pcont = tf.reduce_mean(pcont_pred.log_prob(pcont_target))
            likes.pcont *= self._c.pcont_scale
        prior_dist = self._dynamics.get_dist(prior)
        post_dist = self._dynamics.get_dist(post)
        div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
        div = tf.maximum(div, self._c.free_nats)
        model_loss = self._c.kl_scale * div - sum(likes.values())
        model_loss /= float(self._strategy.num_replicas_in_sync)

    with tf.GradientTape() as actor_tape:
      imag_feat = self._imagine_ahead(post)
      reward = self._reward(imag_feat).mode()
      if self._c.pcont:
        pcont = self._pcont(imag_feat).mean()
      else:
        pcont = self._c.discount * tf.ones_like(reward)
      value = self._value(imag_feat).mode()
      returns = tools.lambda_return(
          reward[:-1], value[:-1], pcont[:-1],
          bootstrap=value[-1], lambda_=self._c.disclam, axis=0)
      discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
          [tf.ones_like(pcont[:1]), pcont[:-2]], 0), 0))
      actor_loss = -tf.reduce_mean(discount * returns)
      actor_loss /= float(self._strategy.num_replicas_in_sync)

    with tf.GradientTape() as value_tape:
      value_pred = self._value(imag_feat)[:-1]
      target = tf.stop_gradient(returns)
      value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))
      value_loss /= float(self._strategy.num_replicas_in_sync)

    model_norm = self._model_opt(model_tape, model_loss)
    actor_norm = self._actor_opt(actor_tape, actor_loss)
    value_norm = self._value_opt(value_tape, value_loss)

    if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
      if self._c.log_scalars:
        self._scalar_summaries(
            data, feat, prior_dist, post_dist, likes, div,
            model_loss, value_loss, actor_loss, model_norm, value_norm,
            actor_norm)
      if tf.equal(log_images, True):
        self._image_summaries(data, embed, image_pred)








def get_feat(self, state):
    return torch.cat([state['stoch'], state['deter']], -1)