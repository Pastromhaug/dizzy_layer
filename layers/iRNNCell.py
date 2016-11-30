import tensorflow as tf
import numpy as np

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from utils.linearTransformIdentityInit import linearTransformIdentityInit
from utils.linearTransformWithBias import linearTransformWithBias

class IRNNCell(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units, input_size=None, bottom=True):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._bottom = bottom

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Most basic RNN: output = new_state = activation(W * input + U * state + B)."""
    with vs.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
        state_out = linearTransformIdentityInit(state, self._num_units, True)
        state_bias = vs.get_variable(
            "state_bias", [self._num_units],
            dtype=tf.float32,
            initializer=init_ops.constant_initializer(dtype=tf.float32))
        state_out = state_out + state_bias
        state_act = tf.nn.relu(state_out)

        if self._bottom == True:
            input_act = linearTransformWithBias([inputs], self._num_units, True)
        else:
            input_out = linearTransformIdentityInit(inputs, self._num_units, True)
            input_bias = vs.get_variable(
                "input_bias", [self._num_units],
                dtype=tf.float32,
                initializer=init_ops.constant_initializer(dtype=tf.float32))
            input_out = input_out + input_bias
            input_act = tf.nn.relu(input_out)

    output = state_act + input_act
    return output, output
