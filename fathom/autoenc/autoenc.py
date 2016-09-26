#!/usr/bin/env python

from math import sqrt
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from fathom.nn import NeuralNetworkModel, default_runstep
import fathom.imagenet.mnist as input_data


# TODO: create an unsupervised parent class

def standard_scale(X_train, X_test):
  preprocessor = prep.StandardScaler().fit(X_train)
  X_train = preprocessor.transform(X_train)
  X_test = preprocessor.transform(X_test)
  return X_train, X_test

# heavily based on tensorflow.models.autoencoder
class AutoencBase(NeuralNetworkModel):
  """Basic Autoencoder (denoising optional)."""
  def load_data(self):
    # Grab the dataset from the internet, if necessary
    self.mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    self.X_train, self.X_test = standard_scale(self.mnist.train.images, self.mnist.test.images)

  def build_hyperparameters(self):
    # Parameters
    self.learning_rate = 0.001
    self.batch_size = 128
    if self.init_options:
      self.batch_size = self.init_options.get('batch_size', self.batch_size)
    self.display_step = 1

    # Network Parameters
    self.n_hidden = 200

    # TODO: remove this data-specific stuff
    self.n_input = 784 # MNIST data input (img shape: 28*28)

    if not self.forward_only:
      self.scale = tf.placeholder(tf.float32)
    #self.keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

  def build_inputs(self):
    # tf Graph input
    self.xs = tf.placeholder(tf.float32, [None, self.n_input])

  @property
  def inputs(self):
    return self.xs

  @property
  def outputs(self):
    return self.reconstruction

  # TODO: remove labels methods upon creating unsupervised parent class
  def build_labels(self):
    # inputs are the ground truth
    pass

  @property
  def labels(self):
    # inputs are the ground truth
    return self.inputs

  def run(self, runstep=None, n_steps=1):
    self.load_data()

    with self.G.as_default():
      # %% We'll train in minibatches and report accuracy:
      self.epochs = 20
      self.display_step = 1

      if self.forward_only:
        self.epochs = 1

      for epoch in xrange(self.epochs):
        # TODO: re-enable options and metadata, which slow down the run

        total_batch = self.mnist.train.num_examples // self.batch_size

        avg_cost = 0
        for batch_i in range(total_batch):
          if batch_i >= n_steps:
            return
          #batch_xs = self.mnist.train.next_batch(self.batch_size)
          batch_xs = get_random_block_from_data(self.X_train, self.batch_size)

          # TODO: summary nodes

          if not self.forward_only:
            # train on batch
            _, loss_value = runstep(
                self.session,
                [self.train, self.loss],
                feed_dict={self.xs: batch_xs, self.scale: self.training_scale},
                #options=run_options, run_metadata=values
            )
          else:
            # run forward on train batch
            _ = runstep(
                self.session,
                self.outputs,
                feed_dict={self.xs: batch_xs}
            )

        if not self.forward_only:
          avg_cost += loss_value * self.mnist.train.num_examples * self.batch_size

          if epoch % self.display_step == 0:
            print('epoch:', epoch, 'cost:', avg_cost)

      print("Total cost:", self.calc_total_cost(self.X_test))

  def noisy_input(self, inputs, scale, dist=tf.random_normal):
    """Add scaled noise to input for denoising autoencoder."""
    with self.G.as_default():
      return inputs + scale * dist((self.n_input,))

  def build_inference(self, inputs, transfer_function=tf.nn.softplus, scale=0.1, denoising=True):
    with self.G.as_default():
      self.transfer = transfer_function

      self.training_scale = scale

      network_weights = self._initialize_weights()
      self.weights = network_weights

      if denoising and not self.forward_only:
        # add white noise to the input so the autoencoder learns to reconstruct from noise
        self.hidden = self.transfer(
          tf.matmul(self.noisy_input(self.xs, self.scale), self.weights['w1']) + self.weights['b1'])
      else:
        # learn to reconstruct the input alone
        self.hidden = self.transfer(tf.add(tf.matmul(self.xs, self.weights['w1']), self.weights['b1']))

      self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

      # for an autoencoder, the cost/loss is not just part of training
      self.build_loss(self.inputs, self.reconstruction)

    return self.reconstruction

  def build_loss(self, inputs, reconstruction):
    with self.G.as_default():
      self.loss_op = 0.5 * tf.reduce_sum(tf.pow(tf.sub(reconstruction, inputs), 2.0))
    return self.loss_op

  @property
  def loss(self):
    return self.loss_op

  def build_train(self, total_loss):
    with self.G.as_default():
      opt = tf.train.AdamOptimizer()

      # Compute and apply gradients.
      self.train_op = opt.minimize(total_loss)#, global_step)

    return self.train_op

  @property
  def train(self):
    return self.train_op

  def _initialize_weights(self):
    all_weights = dict()
    all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
    all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
    all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
    all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
    return all_weights

  def calc_total_cost(self, X):
    return self.session.run(self.loss, feed_dict = {self.xs: X, self.scale: self.training_scale})

  def transform(self, X):
    return self.session.run(self.hidden, feed_dict={self.xs: X, self.scale: self.training_scale})

  def generate(self, hidden = None):
    if hidden is None:
      hidden = np.random.normal(size=self.weights["b1"])
    return self.session.run(self.reconstruction, feed_dict={self.hidden: hidden})

  def reconstruct(self, X):
    return self.session.run(self.reconstruction, feed_dict={self.xs: X, self.scale: self.training_scale})

def xavier_init(fan_in, fan_out, constant = 1):
  low = -constant * sqrt(6.0 / (fan_in + fan_out))
  high = constant * sqrt(6.0 / (fan_in + fan_out))
  return tf.random_uniform((fan_in, fan_out),
                           minval = low, maxval = high,
                           dtype = tf.float32)

def get_random_block_from_data(data, batch_size):
  start_index = np.random.randint(0, len(data) - batch_size)
  return data[start_index:(start_index + batch_size)]

class AutoencBaseFwd(AutoencBase):
  forward_only = True

if __name__ == "__main__":
  m = AutoencBase()
  m.setup()
  m.run(runstep=default_runstep)
  m.teardown()
