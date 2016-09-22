from abc import ABCMeta, abstractmethod, abstractproperty
import tensorflow as tf

class FathomModel(object):
  __metaclass__ = ABCMeta
  def __init__(self, opts={}):
    self.G = tf.Graph()
    self.session = None
    self.init = None
    self.coord = None
    self.threads = None
    self.forward = False

  @abstractmethod # FIXME?
  def build(self):
    with self.G.as_default():
      self.build_graph()
      self.init = tf.initialize_all_variables()

  def setup(self, setup_options={}):
    with self.G.as_default():
      self.load_data()
      self.session = tf.Session(config=tf.ConfigProto(**setup_options))
      self.coord = tf.train.Coordinator()
      self.session.run(self.init)
      self.threads = tf.train.start_queue_runners(sess=self.session, coord=self.coord)
      
  @abstractmethod
  def run(self): pass

  def teardown(self):
    self.coord.request_stop()
    self.coord.join(self.threads, stop_grace_period_secs=10)
    if self.session is not None:
      self.session.close()
      self.session = None

  @abstractmethod
  def build_graph(self): pass
  @abstractmethod
  def load_data(self): pass

  @abstractproperty
  def inputs(self): pass # preprocessed input nodes
  @abstractproperty
  def outputs(self): pass # inference output nodes

################################################################################
# Datasets Utilities

from dataset import Dataset
from imagenet_preprocessing import distorted_inputs

class Imagenet(Dataset):
  train_examples_per_epoch = 1281167
  valid_examples_per_epoch = 50000
  image_size = 224
  image_channels = 3
  batch_size = 64
  n_classes = 1001 # one background class

  def build_inputs(self, graph):
    n_inputs = image_size*image_size*image_channels
    op_images = tf.placeholder(tf.float32, [None, image_size, image_size, channels])
    op_batch_images, op_batch_labels = distorted_inputs(self, batch_size=self.batch_size)
    op_labels = tf.placeholder(tf.int64, [None])
    return (op_inputs, op_labels, op_batch_images, op_batch_labels)

  
