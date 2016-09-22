#!/usr/bin/env python
from math import sqrt
import tensorflow as tf
from fathom.util import FathomModel, Imagenet

# Building blocks

def conv(input_op, name, kw, kh, n_out, dw, dh):
  n_in = input_op.get_shape()[-1].value
  with tf.name_scope(name) as scope:
    kernel_init_val = tf.truncated_normal([kh, kw, n_in, n_out], dtype=tf.float32, stddev=0.1)
    kernel = tf.Variable(kernel_init_val, trainable=True, name='weights')
    conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
    bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
    biases = tf.Variable(bias_init_val, trainable=True, name='biases')
    z = tf.reshape(tf.nn.bias_add(conv, biases), [n_in] + conv.get_shape().as_list()[1:])
    z = tf.nn.bias_add(conv, biases)
    activation = tf.nn.relu(z, name=scope)
    return activation

def mpool(input_op, name, kh, kw, dh, dw):
  return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1],
                        padding='VALID', name=name)

def fc(input_op, name, n_out):
  n_in = input_op.get_shape()[-1].value
  with tf.name_scope(name):
    kernel = tf.Variable(tf.truncated_normal([n_in, n_out], dtype=tf.float32, stddev=0.1), name='w')
    biases = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32), name='b')
    activation = tf.nn.relu_layer(input_op, kernel, biases, name=name)
    return activation


class VGG(FathomModel):
  def __init__(self, opts={})
    super(VGG, self).__init__(opts)
    self.inputs = None
    self.outputs = None
    self.loss = None
    self.optimizer = None

    self.learning_rate = 0.0001
    self.training_iters = 200000
    self.batch_size = opts.get('batch_size', 8)

    self.forward_only = opts.get('forward_only', False)
    if self.forward_only:
      self.dropout = 1.
    else:
      self.dropout = 0.8 # probability to keep units

  def build_inference_graph(self, inputs):
    with self.G.as_default():
      # assume images are 224x224x3
      input_shape = inputs.get_shape().as_list()
      # (first dimension is batch size)
      assert input_shape[1]==224, 'Input images should be 224x224x3'
      assert input_shape[2]==224, 'Input images should be 224x224x3'
      assert input_shape[3]==3, 'Input images should be 224x224x3'
      # TODO: check reshape logic
      self.inputs = tf.reshape(inputs, [-1, 224, 224, 1] )

      # Layers 1-2 (outputs 112x112x64)
      conv_1 = conv(self.inputs, name="conv_1", kh=3, kw=3, n_out=64, dh=1, dw=1)
      conv_2 = conv(conv_1,  name="conv_2", kh=3, kw=3, n_out=64, dh=1, dw=1)
      pool_A = mpool(conv_2, name="pool_A", kh=2, kw=2, dw=2, dh=2)

      # Layers 3-4 (outputs 56x56x128)
      conv_3 = conv(pool_A,  name="conv_3", kh=3, kw=3, n_out=128, dh=1, dw=1)
      conv_4 = conv(conv_3,  name="conv_4", kh=3, kw=3, n_out=128, dh=1, dw=1)
      pool_B = mpool(conv_4, name="pool_B", kh=2, kw=2, dh=2, dw=2)

      # Layers 5-6 (outputs 28x28x256)
      conv_5 = conv(pool_B,  name="conv_5", kh=3, kw=3, n_out=256, dh=1, dw=1)
      conv_6 = conv(conv_5,  name="conv_6", kh=3, kw=3, n_out=256, dh=1, dw=1)
      # FIXME: 2 missing layers?
      pool_C = mpool(conv_6, name="pool_C", kh=2, kw=2, dh=2, dw=2)

      # Layers 7-9 (outputs 14x14x512)
      conv_7 = conv(pool_C,  name="conv_7", kh=3, kw=3, n_out=512, dh=1, dw=1)
      conv_8 = conv(conv_7,  name="conv_8", kh=3, kw=3, n_out=512, dh=1, dw=1)
      conv_9 = conv(conv_8,  name="conv_9", kh=3, kw=3, n_out=512, dh=1, dw=1)
      # FIXME: 1 missing layer?
      pool_D = mpool(conv_9, name="pool_D", kh=2, kw=2, dh=2, dw=2)

      # Layers 10-12 (outputs 7x7x512)
      conv_10 = conv(pool_D,  name="conv_10", kh=3, kw=3, n_out=512, dh=1, dw=1)
      conv_11 = conv(conv_10, name="conv_11", kh=3, kw=3, n_out=512, dh=1, dw=1)
      conv_12 = conv(conv_11, name="conv_12", kh=3, kw=3, n_out=512, dh=1, dw=1)
      # FIXME: 1 missing layer?
      pool_E = mpool(conv_12, name="pool_E",  kh=2, kw=2, dw=2, dh=2)

      # flatten
      shp = pool_E.get_shape().as_list()
      flattened_shape = shp[1] * shp[2] * shp[3]
      resh1 = tf.reshape(pool5, [self.batch_size, flattened_shape], name="resh1")

      # fully connected
      fc_13 = fc(resh1, name="fc_13", n_out=4096)
      fc_13_drop = tf.nn.dropout(fc_13, self.dropout, name="fc13_drop")

      fc_14 = fc(fc_13_drop, name="fc_14", n_out=4096)
      fc_14_drop = tf.nn.dropout(fc_14, self.dropout, name="fc14_drop")

      fc_15 = fc(fc_14_drop, name="fc_15", n_out=self.n_classes)

      self.outputs = fc_15

    return self.outputs

  def build_training_graph(self, labels):
    pass # FIXME
  def build(self):
    pass # FIXME
  def run(self, nsteps=10):
    pass # FIXME

class VGGFwd(VGG):
  self.forward_only = True

if __name__ == "__main__":
  m = VGG()
  m.setup()
  m.run()
  m.teardown()
