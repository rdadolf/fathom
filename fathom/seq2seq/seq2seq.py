#!/usr/bin/env python
import tensorflow as tf
import numpy as np

import math
import random
import sys
import time

from fathom.nn import NeuralNetworkModel, default_runstep

import data_utils

class Seq2Seq(NeuralNetworkModel):
  """Based on TensorFlow example of sequence-to-sequence translation."""
  def build_inputs(self):
    # Feeds for inputs.
    self.encoder_inputs = []
    self.decoder_inputs = []
    for i in xrange(self.buckets[-1][0]):  # Last bucket is the biggest one.
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
    for i in xrange(self.buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))

  @property
  def inputs(self):
    return self.encoder_inputs, self.decoder_inputs

  @property
  def labels(self):
    return self.target_weights

  def build_labels(self):
    # Our targets are decoder inputs shifted by one.
    self.targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]

    self.target_weights = []
    for i in xrange(self.buckets[-1][1] + 1):
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))

  def build_evaluation(self):
    pass

  def build_inference(self, xs):
    with self.G.as_default():
      # If we use sampled softmax, we need an output projection.
      output_projection = None
      softmax_loss_function = None
      # Sampled softmax only makes sense if we sample less than vocabulary size.
      num_samples = self.num_samples
      if num_samples > 0 and num_samples < self.target_vocab_size:
        with tf.device("/cpu:0"):
          w = tf.get_variable("proj_w", [self.size, self.target_vocab_size])
          w_t = tf.transpose(w)
          b = tf.get_variable("proj_b", [self.target_vocab_size])
        output_projection = (w, b)

        def sampled_loss(inputs, labels):
          with tf.device("/cpu:0"):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                                              self.target_vocab_size)
        softmax_loss_function = sampled_loss

      # Create the internal multi-layer cell for our RNN.
      single_cell = tf.nn.rnn_cell.GRUCell(self.size)
      if self.use_lstm:
        single_cell = tf.nn.rnn_cell.BasicLSTMCell(self.size)
      cell = single_cell
      if self.num_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers)

      # The seq2seq function: we use embedding for the input and attention.
      def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
        return tf.nn.seq2seq.embedding_attention_seq2seq(
            encoder_inputs, decoder_inputs, cell,
            num_encoder_symbols=self.source_vocab_size,
            num_decoder_symbols=self.target_vocab_size,
            embedding_size=self.size,
            output_projection=output_projection,
            feed_previous=do_decode)

      # Training outputs and losses.
      if self.forward_only:
        self._outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
            self.encoder_inputs, self.decoder_inputs, self.targets,
            self.target_weights, self.buckets, lambda x, y: seq2seq_f(x, y, True),
            softmax_loss_function=softmax_loss_function)
        # If we use output projection, we need to project outputs for decoding.
        if output_projection is not None:
          for b in xrange(len(self.buckets)):
            self._outputs[b] = [
                tf.matmul(output, output_projection[0]) + output_projection[1]
                for output in self._outputs[b]
            ]
      else:
        self._outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
            self.encoder_inputs, self.decoder_inputs, self.targets,
            self.target_weights, self.buckets,
            lambda x, y: seq2seq_f(x, y, False),
            softmax_loss_function=softmax_loss_function)

    return self._outputs

  @property
  def loss(self):
    return self.losses

  @property
  def train(self):
    return self.updates

  def build_loss(self, logits, labels):
    with self.G.as_default():
      # TODO: how to handle this in seq2seq? refactoring needed
      self.loss_op = self.losses
    return self.losses

  def build_train(self, losses):
    # TODO: modify total_loss to handle buckets
    self.updates = None
    with self.G.as_default():
      # Gradients and SGD update operation for training the model.
      params = tf.trainable_variables()
      if not self.forward_only:
        self.gradient_norms = []
        self.updates = []
        self.opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        for b in xrange(len(self.buckets)):
          gradients = tf.gradients(self.losses[b], params)
          clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                           self.max_gradient_norm)
          self.gradient_norms.append(norm)
          self.updates.append(self.opt.apply_gradients(
              zip(clipped_gradients, params), global_step=self.global_step))

    return self.updates # note: this is per-bucket

  def load_data(self):
    # TODO: make configurable
    self.data_dir = "/data/WMT15/"

    print("Preparing WMT data in %s" % self.data_dir)
    en_train, fr_train, en_dev, fr_dev, _, _ = data_utils.prepare_wmt_data(
        self.data_dir, self.en_vocab_size, self.fr_vocab_size)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % self.max_train_data_size)
    self.dev_set = self.read_data(en_dev, fr_dev)
    self.train_set = self.read_data(en_train, fr_train, self.max_train_data_size)
    train_bucket_sizes = [len(self.train_set[b]) for b in xrange(len(self._buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    self.train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

  def read_data(self, source_path, target_path, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
      source_path: path to the files with token-ids for the source language.
      target_path: path to the file with token-ids for the target language;
        it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
      max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).

    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in self._buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
      with tf.gfile.GFile(target_path, mode="r") as target_file:
        source, target = source_file.readline(), target_file.readline()
        counter = 0
        while source and target and (not max_size or counter < max_size):
          counter += 1
          if counter % 100000 == 0:
            print("  reading data line %d" % counter)
            sys.stdout.flush()
          source_ids = [int(x) for x in source.split()]
          target_ids = [int(x) for x in target.split()]
          target_ids.append(data_utils.EOS_ID)
          for bucket_id, (source_size, target_size) in enumerate(self._buckets):
            if len(source_ids) < source_size and len(target_ids) < target_size:
              data_set[bucket_id].append([source_ids, target_ids])
              break
          source, target = source_file.readline(), target_file.readline()
    return data_set

  @property
  def outputs(self):
    return self._outputs

  def build_hyperparameters(self):
    # data-specific
    self.en_vocab_size = 40000
    self.fr_vocab_size = 40000
    self.max_train_data_size = 1 # 0 is no limit

    # We use a number of buckets and pad to the closest one for efficiency.
    # See seq2seq_model.Seq2SeqModel for details of how they work.
    self._buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    # Parameters
    self.source_vocab_size = self.en_vocab_size
    self.target_vocab_size = self.fr_vocab_size
    self.buckets = self._buckets # FIXME: better bucket names
    self.num_samples = 512
    self.size = 256
    self.num_layers = 3
    self.use_lstm = True # else GRU

    self.batch_size = 64
    if self.init_options:
      self.batch_size = self.init_options.get('batch_size', self.batch_size)

    self.display_step = 1
    self.global_step = tf.Variable(0, trainable=False)
    if not self.forward_only:
      self.learning_rate = tf.Variable(0.5, trainable=False)
      self.learning_rate_decay_factor = 0.99
      self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * self.learning_rate_decay_factor)
      self.max_gradient_norm = 5.0

  def run(self, runstep=None, n_steps=1):
    # Grab the dataset from the internet, if necessary
    self.load_data()

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
      if current_step >= n_steps:
        return
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(self.train_buckets_scale))
                       if self.train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = self.get_batch(
          self.train_set, bucket_id)
      output_feeds, input_feeds = self.step_feeds(encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, self.forward_only)

      outputs = runstep(
        self.session,
        output_feeds,
        input_feeds,
        #options=run_options, run_metadata=values
      )

      # TODO: do this in a runstep
      if not self.forward_only:
        _, step_loss, _ = outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
      else:
        _, step_loss, _ = None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

      step_time += (time.time() - start_time) / self.display_step
      loss += step_loss / self.display_step
      current_step += 1

      if not self.forward_only:
        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % self.display_step == 0:
          # Print statistics for the previous epoch.
          perplexity = math.exp(loss) if loss < 300 else float('inf')
          with self.session.as_default():
            print ("global step %d learning rate %.4f step-time %.2f perplexity "
                   "%.2f" % (self.global_step.eval(), self.learning_rate.eval(),
                             step_time, perplexity))
          # Decrease learning rate if no improvement was seen over last 3 times.
          if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
            self.session.run(self.learning_rate_decay_op)
          previous_losses.append(loss)
          # Save checkpoint and zero timer and loss.
          #checkpoint_path = os.path.join(self.train_dir, "translate.ckpt")
          #self.saver.save(sess, checkpoint_path, global_step=self.global_step)
          step_time, loss = 0.0, 0.0
          # Run evals on development set and print their perplexity.
          for bucket_id in xrange(len(self._buckets)):
            if len(self.dev_set[bucket_id]) == 0:
              print("  eval: empty bucket %d" % (bucket_id))
              continue
            encoder_inputs, decoder_inputs, target_weights = self.get_batch(
                self.dev_set, bucket_id)
            output_feeds, input_feeds = self.step_feeds(encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, True)

            outputs = self.session.run(
              output_feeds,
              input_feeds,
              #options=run_options, run_metadata=values
            )

            # TODO: do this in a runstep
            if not self.forward_only:
              _, eval_loss, _ = outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.

              if False: # FIXME: remove this temporary
                eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
            else:
              _, eval_loss, _ = None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

          sys.stdout.flush()

  def step_feeds(self, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only):
    """Construct feeds for given inputs.

    Args:
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
      #print("encoder", len(encoder_inputs[l]), self.encoder_inputs[l].get_shape())
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      #print("decoder", len(decoder_inputs[l]), self.decoder_inputs[l].get_shape())
      input_feed[self.target_weights[l].name] = target_weights[l]
      #print("target", len(target_weights[l]), self.target_weights[l].get_shape())

    # Since our targets are decoder inputs shifted by one, we need one more.
    #last_target = self.decoder_inputs[decoder_size].name
    last_target = self.decoder_inputs[decoder_size]
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self._outputs[bucket_id][l])

    return output_feed, input_feed

  def get_batch(self, data, bucket_id):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # Encoder inputs are padded and then reversed.
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights

class Seq2SeqFwd(Seq2Seq):
  forward_only = True

if __name__=='__main__':
  m = Seq2Seq()
  m.setup()
  m.run(runstep=default_runstep, n_steps=10)
  m.teardown()

