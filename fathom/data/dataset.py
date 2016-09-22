#!/usr/bin/env python

import os
import tensorflow as tf

# Imagenet (alexnet, residual)
# MNIST (autoenc)
# DeepQ
# Memnet
# seq2seq
# speech

class Dataset(object):
  """Base class for datasets."""
  def num_classes(self):
    pass

  def num_train_examples(self):
    pass

  def num_test_examples(self):
    pass

  def train_test_data(self):
    pass

  def next_batch(self):
    pass

