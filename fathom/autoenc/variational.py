#!/usr/bin/env python

import tensorflow as tf

import numpy as np
from fathom.nn import default_runstep
from fathom.autoenc.autoenc import xavier_init, AutoencBase

# heavily based on tensorflow.models.autoencoder
class Autoenc(AutoencBase):
  """Variational Autoencoder."""
  def build_inference(self, inputs, transfer_function=tf.nn.softplus, scale=0.1):
    with self.G.as_default():
      self.transfer = transfer_function

      self.training_scale = scale

      network_weights = self._initialize_weights()
      self.weights = network_weights

      self.z_mean = tf.add(tf.matmul(inputs, self.weights['w1']), self.weights['b1'])
      self.z_log_sigma_sq = tf.add(tf.matmul(inputs, self.weights['log_sigma_w1']), self.weights['log_sigma_b1'])

      # sample from gaussian distribution
      eps = tf.random_normal(tf.pack([tf.shape(self.xs)[0], self.n_hidden]), 0, 1, dtype = tf.float32)
      self.z = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

      self.reconstruction = tf.add(tf.matmul(self.z, self.weights['w2']), self.weights['b2'])

      # for unsupervised model, loss is part of testing as well
      self.build_loss(self.inputs, self.outputs)

    return self.reconstruction

  def build_loss(self, inputs, reconstruction):
    with self.G.as_default():
      # cost
      reconstr_loss = 0.5 * tf.reduce_sum(tf.pow(tf.sub(self.reconstruction, self.xs), 2.0))
      latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                         - tf.square(self.z_mean)
                                         - tf.exp(self.z_log_sigma_sq), 1)
      self.loss_op = tf.reduce_mean(reconstr_loss + latent_loss)
    return self.loss_op

  def _initialize_weights(self):
    all_weights = dict()
    all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
    all_weights['log_sigma_w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
    all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
    all_weights['log_sigma_b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
    all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
    all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
    return all_weights

  def generate(self, hidden = None):
    if hidden is None:
      hidden = np.random.normal(size=self.weights["b1"])
    return self.session.run(self.reconstruction, feed_dict={self.z_mean: hidden})

class AutoencFwd(Autoenc):
  forward_only = True

if __name__ == "__main__":
  m = Autoenc()
  m.setup()
  m.run(runstep=default_runstep, n_steps=10)
  m.teardown()
