#!/usr/bin/env python
import tensorflow as tf

from fathom.imagenet import imagenet
from fathom.nn import default_runstep

def conv2d(name, l_input, w, b):
  return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)

def max_pool(name, l_input, k):
  return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def norm(name, l_input, lsize=4):
  return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

class AlexNet(imagenet.ImagenetModel):
  """Based on Aymeric Damien's TensorFlow example of AlexNet."""
  def build_inference(self, images):
    with self.G.as_default():
      # conv1
      with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)

      # pool1
      pool1 = tf.nn.max_pool(conv1,
                             ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1],
                             padding='VALID',
                             name='pool1')

      # TODO: lrn1
      lsize = 4
      norm1 = tf.nn.lrn(pool1, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

      # conv2
      with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)

      # pool2
      pool2 = tf.nn.max_pool(conv2,
                             ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1],
                             padding='VALID',
                             name='pool2')

      norm2 = tf.nn.lrn(pool2, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

      # conv3
      with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(norm2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)

      # conv4
      with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)

      # conv5
      with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)

      # pool5
      pool5 = tf.nn.max_pool(conv5,
                             ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1],
                             padding='VALID',
                             name='pool5')

      pool5_shape = pool5.get_shape().as_list()
      pool5_length = pool5_shape[1] * pool5_shape[2] * pool5_shape[3]

      wd1 = tf.Variable(tf.random_normal([pool5_length, 4096]))
      bd1 = tf.Variable(tf.random_normal([4096]))

      flattened_pool5 = tf.reshape(pool5, [self.batch_size, pool5_length])
      dense1 = tf.nn.relu(tf.nn.xw_plus_b(flattened_pool5, wd1, bd1), name='fc1')

      wd2 = tf.Variable(tf.random_normal([4096, 4096]))
      bd2 = tf.Variable(tf.random_normal([4096]))
      dense2 = tf.nn.relu(tf.nn.xw_plus_b(dense1, wd2, bd2), name='fc2')

      w_out = tf.Variable(tf.random_normal([4096, self.n_classes]))
      b_out = tf.Variable(tf.random_normal([self.n_classes]))

      self.logits = tf.nn.xw_plus_b(dense2, w_out, b_out)

    return self.logits

  def build_hyperparameters(self):
    self.learning_rate = 0.001
    self.training_iters = 200000
    self.batch_size = 64
    if self.init_options:
      self.batch_size = self.init_options.get('batch_size', self.batch_size)
    self.display_step = 1

    self.dropout = 0.8 # Dropout, probability to keep units

    # TODO: can this not be a placeholder?
    self.keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

class AlexNetFwd(AlexNet):
  forward_only = True

if __name__=='__main__':
  m = AlexNet()
  m.setup()
  m.run(runstep=default_runstep, n_steps=10)
  m.teardown()

