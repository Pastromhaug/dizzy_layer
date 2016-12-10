import tensorflow as tf
import numpy as np
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs

from utils.linearTransformWithBias import linearTransformWithBias

class BasicRNNCellGauss(tf.nn.rnn_cell.RNNCell):
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
        output = self._activation(linearTransformWithBias([inputs, state], self._num_units, True))
    return output, output
