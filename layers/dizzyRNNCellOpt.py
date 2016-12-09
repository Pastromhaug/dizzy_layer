import tensorflow as tf
import numpy as np

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from utils.linearTransformWithBias import linearTransformWithBias
from utils.rotationTransform import rotationTransform

class DizzyRNNCellOpt(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""
  def __init__(self, num_units, bottom=True):
        self._num_units = num_units
        self._bottom = bottom

  @property
  def state_size(self):
        return self._num_units

  @property
  def output_size(self):
        return self._num_units

  def __call__(self, inputs, state, scope=None):

        with vs.variable_scope(scope or type(self).__name__):

            t_state = tf.transpose(state)
            t_inputs = tf.transpose(inputs)
            if self._bottom == True:
                [state_out] = rotationTransform([("StateL", t_state)], self._num_units, scope)
                input_out = linearTransformWithBias([inputs],
                    self._num_units, bias=False, scope=scope)
            else:
                [state_out, input_out] = \
                    rotationTransform([("StateL", t_state), ("InputL", t_inputs)],
                    self._num_units, scope)
                input_out = tf.transpose(input_out)
            state_out = tf.transpose(state_out)


            bias = vs.get_variable(
                "Bias", [self._num_units],
                dtype=tf.float32,
                initializer=init_ops.constant_initializer(dtype=tf.float32))

            output = tf.abs(state_out + input_out + bias)
        return output, output
