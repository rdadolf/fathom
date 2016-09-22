#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from fathom.util import FathomModel, Imagenet

# Building blocks

def conv2d(x, n_filters, k_h=1, k_w=1, stride_h=1, stride_w=1, stddev=0.02,
           bias=True, padding='SAME', name='Conv2D'):
  with tf.variable_scope(name):
    w = tf.get_variable('weights', [k_h, k_w, x.get_shape()[-1], n_filters],
      initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(x, w, strides=[1, stride_h, stride_w, 1], padding=padding)
    if bias:
      b = tf.get_variable(
        'b', [n_filters],
        initializer=tf.truncated_normal_initializer(stddev=stddev))

      conv = conv + b

    # FIXME: batch normalization?
    return activation(conv)

class Residual(FathomModel):
  def __init__(self, opts={}):
    super(Residual, self).__init__(opts)
    self.inputs = None
    self.outputs = None
    self.loss = None
    self.optimizer = None

    self.learning_rate = 0.01
    self.batch_size = opts.get('batch_size', 16)

    self.forward_only = opts.get('forward_only', False)

  def build_inference_graph(self, inputs):
    with self.G.as_default():
      # (# of blocks, # of channels)
      # FIXME: changed to what's in the paper. check this.
      stages = [ (3, 64),
                 (4, 128),
                 (6, 256),
                 (3, 512) ]

      # assume images are 224x224x3
      input_shape = inputs.get_shape().as_list()
      # (first dimension is batch size)
      assert input_shape[1]==224, 'Input images should be 224x224x3'
      assert input_shape[2]==224, 'Input images should be 224x224x3'
      assert input_shape[3]==3, 'Input images should be 224x224x3'
      # TODO: check reshape logic
      self.inputs = tf.reshape(inputs, [-1, 224, 224, 1])

      # First layer expands channels and pools
      n_channels = stages[0][1]
      net = conv2d(self.inputs, n_channels, k_h=7, k_w=7, stride_h=2, stride_w=2,
                   name='conv1', activation=tf.nn.relu)
      net = tf.nn.max_pool(net, [1,3,3,1], strides=[1,2,2,1], padding='SAME')

      # Generate all residual layers
      for stage_i, (n_blocks,n_channels) in enumerate(stages):
        for block_i in range(n_blocks):
          # Follow naming convention in the original paper
          name = 'conv%d_%d' % (stage_i+1, block_i+1)
          # downsample in the first conv of each stage (except the first)
          if block_i==0 and stage_i!=0:
            stride = 2
          else:
            stride = 1
          conv_a = conv2d(net, n_channels, k_h=3, k_w=3,
                          stride_h=stride, stride_w=stride, name=name+'a')
          conv_b = conv2d(conv_a, n_channels, k_h=3, k_w=3, name=name+'b')
          # projection shortcut for dimension matching
          # FIXME: double check this
          if block_i==0 and stage_i!=(len(stages)-1):
            new_n_channels = stages[stage_i+1][0]
            net = conv2d(net, new_n_channels, bias=False, name=name+'proj')
          net = conv_b + net

      # Global average pooling
      (k_h, k_w) = net.get_shape().as_list[1:3]
      net = tf.nn.avg_pool(net, ksize=[1, k_h, k_w, 1], strides=[1, 1, 1, 1],
                           padding='VALID')
      # Flatten
      width = np.prod(net.get_shape().as_list()[1:])
      net = tf.reshape(net, [-1, width])
      # FC
      with tf.variable_scope('FC'):
        w = tf.get_variable('weights', [width, self.n_classes], tf.float32,
                            tf.random_normal_initializer(stddev=0.02))
        self.outputs = tf.matmul(net, w)

      return self.outputs

  def build_training_graph(self, labels):
    pass # FIXME
  def build(self):
    pass # FIXME
  def run(self, n_steps=10):
    pass # FIXME

class ResidualFwd(Residual):
  forward_only = True

if __name__ == "__main__":
  m = Residual()
  m.setup()
  m.run()
  m.teardown()
