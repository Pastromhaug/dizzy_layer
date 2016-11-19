from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

import tensorflow as tf

def DizzyLayer(X):
    n = int(X.get_shape()[0])
    n_prime = int(n*(n-1)/2)
    thetas = tf.Variable(tf.random_uniform([n_prime, 1], 0, 2*math.pi), name="thetas")
    X_split = [X[k, :] for k in range(n)]
    indices = [(a, b) for b in range(n) for a in range(b)]
    for k in range(n_prime):
        (a, b) = indices[k]
        theta = thetas[k]
        c = tf.cos(theta)
        s = tf.sin(theta)
        v_1 =  c*X_split[a]+s*X_split[b]
        v_2 = -s*X_split[a]+c*X_split[b]
        X_split[a] = v_1
        X_split[b] = v_2
    out = tf.pack(X_split)
    return out


class DizzyRNNCell(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units, input_size=None, activation=tanh):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Most basic RNN: output = new_state = activation(W * input + U * state + B)."""
    with vs.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
      X = tf.transpose(state)
      state_out = tf.transpose(DizzyLayer(X));
      input_out = tf.nn.rnn_cell._linear([inputs], self._num_units, True)
      output = tf.abs(state_out + input_out)
    return output, output

class DizzyRNNCell2(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units, input_size=None, activation=tanh):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Most basic RNN: output = new_state = activation(W * input + U * state + B)."""
    with vs.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
      output = self._activation(tf.nn.rnn_cell._linear([inputs, state], self._num_units, True))
    return output, output
