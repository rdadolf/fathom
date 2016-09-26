#!/usr/bin/env python

from math import sqrt
import tensorflow as tf
from fathom.nn import default_runstep
from fathom.imagenet import imagenet

class VGG(imagenet.ImagenetModel):
  """VGG Network."""
  def build_hyperparameters(self):
    # TODO: put these into runstep options or somewhere else
    # Parameters
    self.learning_rate = 0.0001
    self.training_iters = 200000
    self.batch_size = 8
    if self.init_options:
      self.batch_size = self.init_options.get('batch_size', self.batch_size)
    self.display_step = 1

    if not self.forward_only:
      self.dropout = 0.8 # Dropout, probability to keep units
    else:
      self.dropout = 1.

    self.keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

  def build_inference(self, images):
    with self.G.as_default():
      # fix dimensions
      input_shape = images.get_shape().as_list()
      if len(input_shape) == 2:
        ndim = int(sqrt(input_shape[1]))
        if ndim * ndim != input_shape[1]:
          raise ValueError('input_shape should be square')
        images = tf.reshape(images, [-1, ndim, ndim, 1])

      # assume images shape is 224x224x3

      # block 1 -- outputs 112x112x64
      conv1_1 = conv_op(images, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1)
      conv1_2 = conv_op(conv1_1,  name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1)
      pool1 = mpool_op(conv1_2,   name="pool1",   kh=2, kw=2, dw=2, dh=2)

      # block 2 -- outputs 56x56x128
      conv2_1 = conv_op(pool1,    name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1)
      conv2_2 = conv_op(conv2_1,  name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1)
      pool2 = mpool_op(conv2_2,   name="pool2",   kh=2, kw=2, dh=2, dw=2)

      # TODO: VGG pooling in later layers is too aggressive for MNIST
      using_imagenet = True
      if using_imagenet:
        # block 3 -- outputs 28x28x256
        conv3_1 = conv_op(pool2,    name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1)
        conv3_2 = conv_op(conv3_1,  name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1)
        pool3 = mpool_op(conv3_2,   name="pool3",   kh=2, kw=2, dh=2, dw=2)

        # block 4 -- outputs 14x14x512
        conv4_1 = conv_op(pool3,    name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1)
        conv4_2 = conv_op(conv4_1,  name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
        conv4_3 = conv_op(conv4_2,  name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
        pool4 = mpool_op(conv4_3,   name="pool4",   kh=2, kw=2, dh=2, dw=2)

        # block 5 -- outputs 7x7x512
        conv5_1 = conv_op(pool4,    name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1)
        conv5_2 = conv_op(conv5_1,  name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
        conv5_3 = conv_op(conv5_2,  name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1)
        pool5 = mpool_op(conv5_3,   name="pool5",   kh=2, kw=2, dw=2, dh=2)

      # flatten
      shp = pool5.get_shape().as_list() # pool2 if shrunk
      flattened_shape = shp[1] * shp[2] * shp[3]
      resh1 = tf.reshape(pool5, [self.batch_size, flattened_shape], name="resh1")

      # fully connected
      fc6 = fc_op(resh1, name="fc6", n_out=4096)
      fc6_drop = tf.nn.dropout(fc6, self.dropout, name="fc6_drop")

      fc7 = fc_op(fc6_drop, name="fc7", n_out=4096)
      fc7_drop = tf.nn.dropout(fc7, self.dropout, name="fc7_drop")

      fc8 = fc_op(fc7_drop, name="fc8", n_out=self.n_classes)

      self.logits = fc8

    return self.logits

# crudely based on https://github.com/huyng/tensorflow-vgg
# TODO: refactor these utility functions across convnet models to remove dependencies
def conv_op(input_op, name, kw, kh, n_out, dw, dh):
  n_in = input_op.get_shape()[-1].value

  with tf.name_scope(name) as scope:
    kernel_init_val = tf.truncated_normal([kh, kw, n_in, n_out], dtype=tf.float32, stddev=0.1)
    kernel = tf.Variable(kernel_init_val, trainable=True, name='w')
    conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
    bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
    biases = tf.Variable(bias_init_val, trainable=True, name='b')
    z = tf.reshape(tf.nn.bias_add(conv, biases), [n_in] + conv.get_shape().as_list()[1:])
    z = tf.nn.bias_add(conv, biases)
    activation = tf.nn.relu(z, name=scope)
    return activation

def fc_op(input_op, name, n_out):
  n_in = input_op.get_shape()[-1].value

  with tf.name_scope(name):
    kernel = tf.Variable(tf.truncated_normal([n_in, n_out], dtype=tf.float32, stddev=0.1), name='w')
    biases = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32), name='b')
    activation = tf.nn.relu_layer(input_op, kernel, biases, name=name)
    return activation

def mpool_op(input_op, name, kh, kw, dh, dw):
  return tf.nn.max_pool(input_op,
                        ksize=[1, kh, kw, 1],
                        strides=[1, dh, dw, 1],
                        padding='VALID',
                        name=name)

class VGGFwd(VGG):
  forward_only = True

if __name__ == "__main__":
  m = VGG()
  m.setup()
  m.run(runstep=default_runstep, n_steps=10)
  m.teardown()
