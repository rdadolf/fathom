#!/usr/bin/env python

"""Dominique Luna's implementation of End-to-End Memory Networks, refactored."""

from itertools import chain
import tensorflow as tf
import numpy as np
from sklearn import cross_validation
from fathom.nn import NeuralNetworkModel, default_runstep
from data_utils import load_task, vectorize_data

data_dir = "/data/babi/tasks_1-20_v1-2/en/"
task_id = 1

class MemNet(NeuralNetworkModel):
  def build_inference(self, inputs):
    with self.G.as_default():
      self.encoding_op = tf.constant(self.encoding(self.sentence_size, self.embedding_size), name="encoding")

      # variables
      #with tf.variable_scope(self.name):
      nil_word_slot = tf.zeros([1, self.embedding_size])
      A = tf.concat(0, [ nil_word_slot, self.initializer([self.vocab_size-1, self.embedding_size]) ])
      B = tf.concat(0, [ nil_word_slot, self.initializer([self.vocab_size-1, self.embedding_size]) ])
      self.A = tf.Variable(A, name="A")
      self.B = tf.Variable(B, name="B")

      self.TA = tf.Variable(self.initializer([self.memory_size, self.embedding_size]), name='TA')

      self.H = tf.Variable(self.initializer([self.embedding_size, self.embedding_size]), name="H")
      self.W = tf.Variable(self.initializer([self.embedding_size, self.vocab_size]), name="W")

      #with tf.variable_scope(self.name):
      q_emb = tf.nn.embedding_lookup(self.B, self.queries)
      u_0 = tf.reduce_sum(q_emb * self.encoding_op, 1)
      u = [u_0]
      m_emb = tf.nn.embedding_lookup(self.A, self.stories)
      m = tf.reduce_sum(m_emb * self.encoding_op, 2) + self.TA

      # hop
      for hop_number in range(self.hops):
        with tf.name_scope('Hop_'+str(hop_number)):
          u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
          dotted = tf.reduce_sum(m * u_temp, 2)

          # Calculate probabilities
          probs = tf.nn.softmax(dotted)

          probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
          c_temp = tf.transpose(m, [0, 2, 1])
          o_k = tf.reduce_sum(c_temp * probs_temp, 2)

          u_k = tf.matmul(u[-1], self.H) + o_k

          # nonlinearity
          if self.nonlin:
            u_k = nonlin(u_k)

          u.append(u_k)

      self.nil_vars = set([self.A.name, self.B.name])

      self._outputs = tf.matmul(u_k, self.W)

    return self._outputs

  @property
  def outputs(self):
    return self._outputs

  def build_loss(self, logits, labels):
    with self.G.as_default():
      with tf.name_scope('loss'):
        # Define loss
        # TODO: does this labels have unexpected state?
        self.loss_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(labels, tf.float32)))
    return self.loss_op

  @property
  def loss(self):
    return self.loss_op

  def build_train(self, total_loss):
    with self.G.as_default():
      self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

      # can't use opt.minimize because we need to clip the gradients
      grads_and_vars = self.opt.compute_gradients(self.loss)
      grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g,v in grads_and_vars]
      grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
      nil_grads_and_vars = []
      for g, v in grads_and_vars:
        if v.name in self.nil_vars:
          nil_grads_and_vars.append((zero_nil_slot(g), v))
        else:
          nil_grads_and_vars.append((g, v))

      self.train_op = self.opt.apply_gradients(nil_grads_and_vars, name="train_op")

    return self.train_op
  @property
  def train(self):
    return self.train_op

  def load_data(self):
    # single babi task
    # TODO: refactor all this running elsewhere
    # task data
    train, test = load_task(data_dir, task_id)

    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in train + test)))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    self.memory_size = 50

    self.max_story_size = max(map(len, (s for s, _, _ in train + test)))
    self.mean_story_size = int(np.mean(map(len, (s for s, _, _ in train + test))))
    self.sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in train + test)))
    self.query_size = max(map(len, (q for _, q, _ in train + test)))
    self.memory_size = min(self.memory_size, self.max_story_size)
    self.vocab_size = len(word_idx) + 1 # +1 for nil word
    self.sentence_size = max(self.query_size, self.sentence_size) # for the position

    print("Longest sentence length", self.sentence_size)
    print("Longest story length", self.max_story_size)
    print("Average story length", self.mean_story_size)

    # train/validation/test sets
    self.S, self.Q, self.A = vectorize_data(train, word_idx, self.sentence_size, self.memory_size)
    self.trainS, self.valS, self.trainQ, self.valQ, self.trainA, self.valA = cross_validation.train_test_split(self.S, self.Q, self.A, test_size=.1) # TODO: randomstate
    self.testS, self.testQ, self.testA = vectorize_data(test, word_idx, self.sentence_size, self.memory_size)

    print(self.testS[0])

    print("Training set shape", self.trainS.shape)

    # params
    self.n_train = self.trainS.shape[0]
    self.n_test = self.testS.shape[0]
    self.n_val = self.valS.shape[0]

    print("Training Size", self.n_train)
    print("Validation Size", self.n_val)
    print("Testing Size", self.n_test)

  def build_hyperparameters(self):
    with self.G.as_default():
      # TODO: put these into runstep options or somewhere else
      # Parameters
      self.learning_rate = 0.01
      self.batch_size = 32
      if self.init_options:
        self.batch_size = self.init_options.get('batch_size', self.batch_size)
      self.embedding_size = 20
      self.hops = 3
      self.max_grad_norm = 40.0
      self.nonlin = None
      self.encoding = position_encoding
      self.display_step = 10

  def build_inputs(self):
    self.load_data() # TODO: get static numbers for the things that currently require loading and move this to run

    with self.G.as_default():
      # inputs
      self.stories = tf.placeholder(tf.int32, [None, self.memory_size, self.sentence_size], name="stories")
      self.queries = tf.placeholder(tf.int32, [None, self.sentence_size], name="queries")

      self.initializer = tf.random_normal_initializer(stddev=0.1)

  @property
  def inputs(self):
    return self.stories, self.queries

  def build_labels(self):
    with self.G.as_default():
      self.answers = tf.placeholder(tf.int32, [None, self.vocab_size], name="answers")

  @property
  def labels(self):
    return self.answers

  def run(self, runstep=None, n_steps=1):
    # load babi data
    # vocab, memory, sentence sizes set here
    # TODO: get static data size numbers and don't load in inputs anymore
    #self.load_data()
    #tf.set_random_seed(random_state)

    start = 0
    assert self.batch_size<self.n_train, 'Batch size is larger than training data---something is horribly wrong'
    end = self.batch_size
    for _ in range(1, n_steps+1):
      s = self.trainS[start:end]
      q = self.trainQ[start:end]
      a = self.trainA[start:end]

      feed = {self.stories: s, self.queries: q, self.answers: a}

      if not self.forward_only:
        # Fit training using batch data
        _, _, _ = runstep(
            self.session,
            [self.train, self.loss, self.accuracy],
            feed_dict=feed,
        )
      else:
        _ = runstep(
            self.session,
            self.outputs,
            feed_dict=feed,
        )

      # FIXME: Should really be shuffling here.
      if end+self.batch_size>self.n_train:
        start,end = 0,self.batch_size
      else:
        start,end = end,end+self.batch_size

    acc = self.session.run(
        self.accuracy,
        feed_dict={self.stories: self.testS, self.queries: self.testQ, self.answers: self.testA}
    )

    print("Test accuracy: {:.5f}".format(acc))

def position_encoding(sentence_size, embedding_size):
  """
  Position Encoding described in section 4.1 [1]
  """
  encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
  ls = sentence_size+1
  le = embedding_size+1
  for i in range(1, le):
    for j in range(1, ls):
      encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
  encoding = 1 + 4 * encoding / embedding_size / sentence_size
  return np.transpose(encoding)

def zero_nil_slot(t, name=None):
  """
  Overwrites the nil_slot (first row) of the input Tensor with zeros.
  The nil_slot is a dummy slot and should not be trained and influence
  the training algorithm.
  """
  with tf.op_scope([t], name, "zero_nil_slot") as name:
    t = tf.convert_to_tensor(t, name="t")
    s = tf.shape(t)[1]
    z = tf.zeros(tf.pack([1, s]))
    return tf.concat(0, [z, tf.slice(t, [1, 0], [-1, -1])], name=name)

def add_gradient_noise(t, stddev=1e-3, name=None):
  """
  Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].
  The input Tensor `t` should be a gradient.
  The output will be `t` + gaussian noise.
  0.001 was said to be a good fixed value for memory networks [2].
  """
  with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
    t = tf.convert_to_tensor(t, name="t")
    gn = tf.random_normal(tf.shape(t), stddev=stddev)
    return tf.add(t, gn, name=name)

class MemNetFwd(MemNet):
  forward_only = True

if __name__=='__main__':
  m = MemNet()
  m.setup()
  m.run(runstep=default_runstep, n_steps=100)
  m.teardown()

