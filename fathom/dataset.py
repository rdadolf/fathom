#!/usr/bin/env python

import os
import tensorflow as tf

class Dataset(object):
  """Simple wrapper for a dataset.

  Inspired by David Dao's TensorFlow models code.
  """
  def __init__(self, subset, record_dir):
    """
    record_dir: Directory with TFRecords.
    """
    self.subset = subset
    self.record_dir = record_dir

  def data_files(self):
    return tf.gfile.Glob(os.path.join(self.record_dir, "{}-*".format(self.subset)))

  def record_queue(self):
    """Return a TensorFlow queue of TFRecords."""
    return tf.train.string_input_producer(self.data_files())

  def reader(self):
    return tf.TFRecordReader()

