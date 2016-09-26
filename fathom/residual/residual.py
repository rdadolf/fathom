#!/usr/bin/env python

from math import sqrt
from collections import namedtuple
import tensorflow as tf
from fathom.nn import default_runstep
from fathom.imagenet import imagenet

# Code heavily based on Parag Mital's TensorFlow tutorials.
class Residual(imagenet.ImagenetModel):
  """Residual Network."""
  def build_hyperparameters(self):
    # Parameters
    self.learning_rate = 0.01
    self.training_iters = 200000
    self.batch_size = 16
    if self.init_options:
      self.batch_size = self.init_options.get('batch_size', self.batch_size)
    self.display_step = 1

    self.dropout = 0.8 # Dropout, probability to keep units
    self.keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

  def build_inference(self, images):
    with self.G.as_default():
      LayerBlock = namedtuple(
        'LayerBlock', ['num_repeats', 'num_filters', 'bottleneck_size'])
      blocks = [
        LayerBlock(3, 128, 32),
        LayerBlock(3, 256, 64),
        LayerBlock(3, 512, 128),
        LayerBlock(3, 1024, 256)
      ]

      # %%
      input_shape = images.get_shape().as_list()
      if len(input_shape) == 2:
        ndim = int(sqrt(input_shape[1]))
        if ndim * ndim != input_shape[1]:
          raise ValueError('input_shape should be square')
        images = tf.reshape(images, [-1, ndim, ndim, 1])

      # %%
      # First convolution expands to 64 channels and downsamples
      net = conv2d(images, 64, k_h=7, k_w=7,
        name='conv1',
        activation=tf.nn.relu)

      # %%
      # Max pool and downsampling
      net = tf.nn.max_pool(
        net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

      # %%
      # Setup first chain of resnets
      net = conv2d(net, blocks[0].num_filters, k_h=1, k_w=1,
        stride_h=1, stride_w=1, padding='VALID', name='conv2')

      # %%
      # Loop through all res blocks
      for block_i, block in enumerate(blocks):
        for repeat_i in range(block.num_repeats):
          name = 'block_%d/repeat_%d' % (block_i, repeat_i)
          conv = conv2d(net, block.bottleneck_size, k_h=1, k_w=1,
            padding='VALID', stride_h=1, stride_w=1,
            activation=tf.nn.relu,
            name=name + '/conv_in')

          conv = conv2d(conv, block.bottleneck_size, k_h=3, k_w=3,
            padding='SAME', stride_h=1, stride_w=1,
            activation=tf.nn.relu,
            name=name + '/conv_bottleneck')

          conv = conv2d(conv, block.num_filters, k_h=1, k_w=1,
            padding='VALID', stride_h=1, stride_w=1,
            activation=tf.nn.relu,
            name=name + '/conv_out')

          net = conv + net

        try:
          # upscale to the next block size
          next_block = blocks[block_i + 1]
          net = conv2d(net, next_block.num_filters, k_h=1, k_w=1,
            padding='SAME', stride_h=1, stride_w=1, bias=False,
            name='block_%d/conv_upscale' % block_i)
        except IndexError:
          pass

      # %%
      net = tf.nn.avg_pool(net,
          ksize=[1, net.get_shape().as_list()[1],
            net.get_shape().as_list()[2], 1],
          strides=[1, 1, 1, 1], padding='VALID')
      net = tf.reshape(
          net,
          [-1, net.get_shape().as_list()[1] *
            net.get_shape().as_list()[2] *
            net.get_shape().as_list()[3]])

      self.logits = linear(net, self.n_classes, activation=tf.identity)

    # %%
    return self.logits

def conv2d(x, n_filters,
           k_h=5, k_w=5,
           stride_h=2, stride_w=2,
           stddev=0.02,
           activation=lambda x: x,
           bias=True,
           padding='SAME',
           name="Conv2D"):
  """2D Convolution with options for kernel size, stride, and init deviation.
  Parameters
  ----------
  x : Tensor
      Input tensor to convolve.
  n_filters : int
      Number of filters to apply.
  k_h : int, optional
      Kernel height.
  k_w : int, optional
      Kernel width.
  stride_h : int, optional
      Stride in rows.
  stride_w : int, optional
      Stride in cols.
  stddev : float, optional
      Initialization's standard deviation.
  activation : arguments, optional
      Function which applies a nonlinearity
  padding : str, optional
      'SAME' or 'VALID'
  name : str, optional
      Variable scope to use.
  Returns
  -------
  x : Tensor
      Convolved input.
  """
  with tf.variable_scope(name):
    w = tf.get_variable(
      'w', [k_h, k_w, x.get_shape()[-1], n_filters],
      initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(x, w, strides=[1, stride_h, stride_w, 1], padding=padding)
    if bias:
      b = tf.get_variable(
        'b', [n_filters],
        initializer=tf.truncated_normal_initializer(stddev=stddev))

      conv = conv + b
    return activation(conv)

def linear(x, n_units, scope=None, stddev=0.02,
           activation=lambda x: x):
  """Fully-connected network.
  Parameters
  ----------
  x : Tensor
      Input tensor to the network.
  n_units : int
      Number of units to connect to.
  scope : str, optional
      Variable scope to use.
  stddev : float, optional
      Initialization's standard deviation.
  activation : arguments, optional
      Function which applies a nonlinearity
  Returns
  -------
  x : Tensor
      Fully-connected output.
  """
  shape = x.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], n_units], tf.float32,
        tf.random_normal_initializer(stddev=stddev))
    return activation(tf.matmul(x, matrix))

class ResidualFwd(Residual):
  forward_only = True

if __name__ == "__main__":
  m = Residual()
  m.setup()
  m.run(runstep=default_runstep, n_steps=10)
  m.teardown()
