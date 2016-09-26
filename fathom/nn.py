#!/usr/bin/env python

import tensorflow as tf
from abc import ABCMeta, abstractmethod, abstractproperty

class GenericModel(object):
  __metaclass__ = ABCMeta
  def __init__(self, device=None, init_options=None):
    self.device=device

  @abstractmethod
  def model(self):
    'Return a reference to the native representation of the model.'
    pass
  def setup(self, setup_options=None):
    '(Optional) Prepare the model for running.'
    pass
  @abstractmethod
  def run(self, runstep=None, n_steps=1, *args, **kwargs):
    'Run the model.'
    pass
  def teardown(self):
    '(Optional) Clean up after a model run.'
    pass

def default_runstep(session, sink_ops, *options, **kw_options):
  return session.run(sink_ops, *options, **kw_options)


class NeuralNetworkModel(GenericModel):
  __metaclass__ = ABCMeta
  forward_only = False

  def __init__(self, device=None, init_options=None):
    super(NeuralNetworkModel,self).__init__(device=device, init_options=init_options)

    self.G = tf.Graph()
    self.session = None

    # e.g., for batch_size
    self.init_options = init_options

    with self.G.device(device):
      with self.G.as_default():
        self.build()

    with self.G.as_default():
      self.init = tf.initialize_all_variables()

  @abstractmethod
  def load_data(self):
    """Load dataset (possibly downloading it)."""
    pass

  @abstractmethod
  def build_inputs(self):
    """Construct graph's input placeholders."""
    pass

  @abstractmethod
  def build_labels(self):
    """Construct graph's label placeholders."""
    pass

  @abstractproperty
  def inputs(self):
    pass

  @abstractproperty
  def labels(self):
    pass

  @abstractmethod
  def build_hyperparameters(self):
    """Set hard-coded hyperparameters."""
    pass

  @abstractproperty
  def outputs(self):
    """Network outputs before loss function."""
    pass

  @abstractproperty
  def loss(self):
    """Loss function."""
    pass

  @abstractproperty
  def train(self):
    """Training/optimization operation."""
    pass

  def build_evaluation(self):
    """Evaluation metrics (e.g., accuracy)."""
    self.correct_pred = tf.equal(tf.argmax(self.outputs, 1), tf.argmax(self.labels, 1))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

  def build(self):
    """Build computation graph."""
    with self.G.as_default():
      self.global_step = tf.Variable(0, trainable=False)

      self.build_hyperparameters()

      self.build_inputs()
      self.build_labels()

      self.build_inference(self.inputs)

      if not self.forward_only:
        self.build_loss(self.outputs, self.labels)
        self.build_train(self.loss_op)

      self.build_evaluation()

  @abstractmethod
  def build_inference(self, inputs):
    """Build inference.

    Args:
      inputs: Images, for example.

    Returns:
      Logits.
    """
    pass

  @abstractmethod
  def build_loss(self, outputs, labels):
    """Add loss to trainable variables.
    Args:
      outputs: Outputs from inference().
      labels: Labels from inputs. 1-D tensor of shape [batch_size].

    Returns:
      Loss tensor of type float.
    """
    pass

  @abstractmethod
  def build_train(self, total_loss, global_step):
    """Train model.

    Create optimizer and apply to all trainable variables.

    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting number of training steps processed.

    Returns:
      train_op: op for training.
    """
    pass

  def model(self):
    return self.G

  def setup(self, setup_options=None):
    """Make session and launch queue runners."""
    super(NeuralNetworkModel,self).setup(setup_options=setup_options)
    with self.G.as_default():
      # Start a new session and initialize the network
      if setup_options is not None:
        self.session = tf.Session(config=tf.ConfigProto(**setup_options))
      else:
        self.session = tf.Session()
      # Start the input data loaders
      self.coord = tf.train.Coordinator()
      self.session.run(self.init)
      # Start the input data loaders
      self.threads = tf.train.start_queue_runners(sess=self.session,coord=self.coord)

  def teardown(self):
    """Close session and join queue runners."""
    self.coord.request_stop()
    self.coord.join(self.threads, stop_grace_period_secs=10)
    if self.session is not None:
      self.session.close()
      self.session = None

