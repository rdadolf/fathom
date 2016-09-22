#!/usr/bin/env python
import tensorflow as tf
from fathom.util import FathomModel, Imagenet, runstep

# Building blocks

def conv2d(inputs, weight, stride, bias):
  weights = tf.Variable(tf.truncated_normal(weight, dtype=tf.float32, stddev=1.e-1), name='weights')
  conv = tf.nn.conv2d(inputs, weights, strides=strides)
  bias = tf.Variable(tf.constant(0.0, shape=bias, dtype=tf.float32), trainable=True, name='biases')
  return tf.nn.relu(bias)

def maxpool(inputs, kernel, stride):
  return tf.nn.max_pool(inputs, ksize=kernel, strides=stride, padding='VALID', name='maxpool')

def lrn(inputs, window=4):
  return tf.nn.lrn(inputs, window, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


class AlexNet(FathomModel):
  def __init__(self, opts={}):
    super(AlexNet, self).__init__(opts)
    self.inputs = None
    self.outputs = None
    self.loss = None
    self.optimizer = None

    self.learning_rate = 0.001
    self.training_iters = 200000
    self.batch_size = opts.get('batch_size', 64)
    self.dropout = 0.8 # Probability to keep units

    self.forward_only = opts.get('forward_only', False)

  def build_inference_graph(self, inputs):
    self.inputs = inputs

    with self.G.as_default():
      with tf.name_scope('layer1') as scope:
        conv1 = conv2d(self.inputs, kernel=[11,11,3,64], stride=[1,4,4,1], bias=[64])
        pool1 = maxpool(conv1, kernel=[1,3,3,1], stride=[1,2,2,1])
        norm1 = lrn(pool1, window=4)

      with tf.name_scope('layer2') as scope:
        conv2 = conv2d(norm1, kernel=[5,5,64,192], stride=[1,1,1,1], bias=[192])
        pool2 = maxpool(conv2, kernel=[1,3,3,1], stride=[1,2,2,1])
        norm2 = lrn(pool2, window=4)

      with tf.name_scope('layer3') as scope:
        conv3 = conv2d(norm2, kernel=[3,3,192,384], stride=[1,1,1,1], bias=[384])

      with tf.name_scope('layer4') as scope:
        conv4 = conv2d(conv3, kernel=[3,3,384,256], stride=[1,1,1,1], bias=[256])

      with tf.name_scope('layer5') as scope:
        conv5 = conv2d(conv4, kernel=[3,3,256,256], stride=[1,1,1,1], bias=[256])
        pool5 = maxpool(conv5, kernel=[1,3,3,1], stride=[1,2,2,1])
        pool5_size = np.prod(pool5.get_shape().as_list())

      # Fully-connected layers
      flat_pool5 = tf.reshape(pool5, [self.batch_size, pool5_length])
      wd1 = tf.Variable(tf.random_normal([pool5_size, 4096]))
      bd1 = tf.Variable(tf.random_normal([4096]))
      dense1 = tf.nn.relu(tf.nn.xw_plus_b(flat_pool5, wd1, bd1), name='fc1')

      wd2 = tf.Variable(tf.random_normal([4096, 4096]))
      bd2 = tf.Variable(tf.random_normal([4096]))
      dense2 = tf.nn.relu(tf.nn.xw_plus_b(dense1, wd2, bd2), name='fc2')

      w_out = tf.Variable(tf.random_normal([4096, self.n_classes]))
      b_out = tf.Variable(tf.random_normal([self.n_classes]))

      self.outputs = tf.nn.xw_plus_b(dense2, w_out, b_out)

    return self.outputs

  def build_training_graph(self, labels):
    with self.G.as_default():
      self.loss = tf.reduce_mean(tf.nn.sparse_sotfmax_cross_entropy_with_logits(self.outputs, labels)
      self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    return self.loss, self.optimizer

  def build(self):
    (inputs, labels ) = (None,None) # FIXME load ImageNet data
    self.build_inference_graph( inputs )
    if not self.forward_only:
      self.build_training_graph( labels )

  def run(self, runstep=runstep, nsteps=10):
    # FIXME: load data here?
    with self.G.as_default():
      for i in range(0,nsteps):
        batch_inputs, batch_labels = # FIXME: create minibatch

        _, loss = runstep(self.session,
          [self.optimizer, self.loss],
          {self.inputs: batch_inputs, self.labels: batch_labels} )

class AlexNetFwd(AlexNet):
  self.forward_only = True

if __name__=='__main__':
  m = AlexNet()
  m.setup()
  m.run()
  m.teardown()

