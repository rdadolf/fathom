#!/usr/bin/env python

import numpy as np
import tensorflow as tf

#from tensorflow.models.rnn import rnn, rnn_cell
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import variable_scope as vs
try:
  from tensorflow.python.ops.rnn_cell import linear as _linear
except ImportError:
  from tensorflow.python.ops.rnn_cell import _linear

from fathom.nn import NeuralNetworkModel, default_runstep

from preproc import load_timit, timit_hdf5_filepath
from phoneme import index2phoneme_dict


def clipped_relu(inputs, clip=20):
  """Similar to tf.nn.relu6, but can clip at 20 as in Deep Speech."""
  return tf.minimum(tf.nn.relu(inputs), clip)


class ClippedReluRNNCell(tf.nn.rnn_cell.RNNCell):
  """Basic RNN cell with clipped ReLU rather than tanh activation."""

  def __init__(self, num_units, input_size=None):
    self._num_units = num_units

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Basic RNN: output = new_state = clipped_relu(W * input + U * state + B)."""
    with vs.variable_scope(scope or type(self).__name__):
      output = clipped_relu(_linear([inputs, state], self._num_units, True))
    return output, output


# TODO: show label error rate
# TODO: avoid labels and blank off-by-one error due to padding zeros
class Speech(NeuralNetworkModel):
  """RNN for speech recognition."""
  def __init__(self, device=None, init_options=None):
    super(Speech,self).__init__(device=device, init_options=init_options)

  #def inference(self, inputs, n_hidden=2048):
  def build_inference(self, inputs, n_hidden=1024):
    with self.G.as_default():
      self.n_hidden = n_hidden

      # Architecture of Deep Speech [Hannun et al. 2014]
      outputs_1 = self.mlp_layer(inputs, self.n_coeffs, self.n_hidden)
      outputs_2 = self.mlp_layer(outputs_1, self.n_hidden, self.n_hidden)
      outputs_3 = self.mlp_layer(outputs_2, self.n_hidden, self.n_hidden)
      outputs_4 = self.bidirectional_layer(outputs_3, n_input=self.n_hidden, n_hidden=self.n_hidden, n_output=self.n_hidden)
      outputs_5 = self.mlp_layer(outputs_3, self.n_hidden, self.n_labels)

      self._outputs = outputs_5

      # transpose in preparation for CTC loss
      self.logits_t = tf.transpose(self._outputs, perm=[1,0,2])

      return outputs_5

  @property
  def outputs(self):
    return self._outputs

  @property
  def loss(self):
    return self.loss_op

  def build_loss(self, logits, labels):
    with self.G.as_default():
      # NOTE: CTC does the softmax for us, according to the code

      # CTC loss requires sparse labels
      self.sparse_labels = self.ctc_label_dense_to_sparse(self.labels, self.seq_lens)

      # CTC
      self.loss_op = tf.contrib.ctc.ctc_loss(
        inputs=self.logits_t,
        labels=self.sparse_labels,
        sequence_length=self.seq_lens
      )

      return self.loss_op

  def build_train(self, loss):
    # TODO: buckets
    with self.G.as_default():
      self.train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
      return self.train_op

  @property
  def train(self):
    return self.train_op

  def mlp_layer(self, inputs, n_input, n_output):
    with self.G.as_default():
      # layer sees inputs as (batch_size, max_time, n_input)
      W = tf.Variable(tf.zeros([n_input, n_output]))
      b = tf.Variable(tf.zeros([n_output]))

      W_batch_multiples = tf.constant([self.batch_size, 1, 1], dtype=tf.int32)
      W_batch = tf.tile(tf.expand_dims(W, 0), W_batch_multiples)

      # TODO: is tiling a bias vector over batch and frames correct?
      b_batch_multiples = tf.constant([self.batch_size, self.max_frames, 1], dtype=tf.int32)
      b_batch = tf.tile(tf.expand_dims(tf.expand_dims(b, 0), 0), b_batch_multiples)

      # TODO: change batch_matmul to an averaging reshape so that batching happens and dimensions are easier
      outputs = tf.add(tf.batch_matmul(inputs, W_batch), b_batch)

      return clipped_relu(outputs)

  def bidirectional_layer(self, inputs, n_input, n_hidden, n_output):
    """Bidirectional RNN layer."""
    with self.G.as_default():
      fw_cell = ClippedReluRNNCell(n_hidden)
      bw_cell = ClippedReluRNNCell(n_hidden)

      # input shape: (batch_size, max_time, n_input)
      inputs = tf.transpose(inputs, perm=[1, 0, 2])  # permute max_time and batch_size
      inputs = tf.reshape(inputs, [-1, n_input]) # (max_time*batch_size, n_input)

      inputs = tf.split(0, self.max_frames, inputs) # max_time * (batch_size, n_hidden)

      # optional initial states
      istate_fw = tf.placeholder("float", [None, n_hidden])
      istate_bw = tf.placeholder("float", [None, n_hidden])

      # TODO: support both tanh (default) and clipped_relu
      outputs, _, _ = tf.nn.bidirectional_rnn(fw_cell, bw_cell, inputs, initial_state_fw=istate_fw, initial_state_bw=istate_bw)

      # TODO: is this the right output?
      return outputs[-1]

  def ctc_label_dense_to_sparse( self, labels, label_lengths ):
    """Mike Henry's implementation, with some minor modifications."""
    with self.G.as_default():
      label_shape = tf.shape( labels )
      num_batches_tns = tf.pack( [label_shape[0]] )
      max_num_labels_tns = tf.pack( [label_shape[1]] )

      def range_less_than(previous_state, current_input):
        return tf.expand_dims( tf.range( label_shape[1] ), 0 ) < current_input

      init = tf.cast( tf.fill( max_num_labels_tns, 0 ), tf.bool )
      dense_mask = functional_ops.scan(range_less_than, label_lengths , initializer=init, parallel_iterations=1)
      dense_mask = dense_mask[ :, 0, : ]

      label_array = tf.reshape( tf.tile( tf.range( 0, label_shape[1] ), num_batches_tns ), label_shape )
      label_ind = tf.boolean_mask( label_array, dense_mask )

      batch_array = tf.transpose( tf.reshape( tf.tile( tf.range( 0,  label_shape[0] ), max_num_labels_tns ), tf.reverse( label_shape,[True]) ) )
      batch_ind = tf.boolean_mask( batch_array, dense_mask )

      indices = tf.transpose( tf.reshape( tf.concat( 0, [batch_ind, label_ind] ), [2,-1] ) )
      vals_sparse = tf.gather_nd( labels, indices )
      return tf.SparseTensor( tf.to_int64(indices), vals_sparse, tf.to_int64( label_shape ) )

  def build_hyperparameters(self):
    self.n_labels = 61 + 1 # add blank
    self.max_frames = 1566 # TODO: compute dynamically
    self.max_labels = 75
    self.n_coeffs = 26
    self.batch_size = 32
    if self.init_options:
      self.batch_size = self.init_options.get('batch_size', self.batch_size)

  def build_inputs(self):
    with self.G.as_default():
      # NOTE: ctc_loss requires a transpose
      # tf.transpose(inputs,perm=[1,0,2])
      self._inputs = tf.placeholder(tf.float32, [None, self.max_frames, self.n_coeffs], name="inputs")

  @property
  def inputs(self):
    return self._inputs

  def build_labels(self):
    with self.G.as_default():
      self._labels = tf.placeholder(tf.int32, [None, self.max_labels], name="labels")
      self.seq_lens = tf.placeholder(tf.int32, [None], name="seq_lens")

  @property
  def labels(self):
    return self._labels

  def build(self):
    super(Speech, self).build()

    with self.G.as_default():
      self.decode_op = self.decoding()

  def load_data(self):
    self.train_spectrograms, self.train_labels, self.train_seq_lens = load_timit(timit_hdf5_filepath, train=True)
    # TODO: load test

  def get_random_batch(self):
    """Get random batch from np.arrays (not tf.train.shuffle_batch)."""
    n_examples = self.train_spectrograms.shape[0]
    random_sample = np.random.randint(n_examples, size=self.batch_size)
    return self.train_spectrograms[random_sample, :, :], self.train_labels[random_sample, :], self.train_seq_lens[random_sample]

  def decoding(self):
    """Predict labels from learned sequence model."""
    # TODO: label error rate on validation set
    decoded, _ = tf.contrib.ctc.ctc_greedy_decoder(self.logits_t, self.seq_lens)
    sparse_decode_op = decoded[0] # single-element list
    self.decode_op = tf.sparse_to_dense(sparse_decode_op.indices, sparse_decode_op.shape, sparse_decode_op.values)
    return self.decode_op

  def run(self, runstep=None, n_steps=1, *args, **kwargs):
    print("Loading spectrogram features...")
    self.load_data()

    with self.G.as_default():
      print 'Starting run...'
      for _ in xrange(n_steps):
        spectrogram_batch, label_batch, seq_len_batch = self.get_random_batch()

        if not self.forward_only:
          _, _ = runstep(self.session,
              [self.train_op, self.loss_op],
              feed_dict={self.inputs: spectrogram_batch, self.labels: label_batch, self.seq_lens: seq_len_batch})
        else:
          # run forward-only on train batch
          _ = runstep(self.session,
              self.outputs,
              feed_dict={self.inputs: spectrogram_batch, self.labels: label_batch, self.seq_lens: seq_len_batch})

        # decode the same batch, for debugging
        decoded = self.session.run(self.decode_op,
            feed_dict={self.inputs: spectrogram_batch, self.labels: label_batch, self.seq_lens: seq_len_batch})

        # print some decoded examples
        if False:
          print(' '.join(self.labels2phonemes(decoded[0])))
          # TODO: fix dtypes in dataset (labels are accidentally floats right now)
          print(' '.join(self.labels2phonemes(np.array(label_batch[0,:], dtype=np.int32))))

  def labels2phonemes(self, decoded_labels):
    """Convert a list of label indices to a list of corresponding phonemes."""
    return [index2phoneme_dict[label] for label in decoded_labels]

class SpeechFwd(Speech):
  forward_only = True

if __name__=='__main__':
  m = Speech()
  m.setup()
  m.run(runstep=default_runstep, n_steps=10)
  m.teardown()
