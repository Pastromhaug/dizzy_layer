import tensorflow as tf
import numpy as np

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from utils.linearTransformWithBias import linearTransformWithBias
from utils.rotationPreprocess import rotationPreprocess
from utils.rotationTransform import rotationTransform

class DizzyRNNCellOpt(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""
  def __init__(self, num_units, bottom=True):
        self._num_units = num_units
        self._indices = [(a, b) for b in range(self._num_units) for a in range(b)]
        self._num_params = num_units*(num_units-1)/2
        cos_list,  sin_list, nsin_list, cos_idxs, sin_idxs, nsin_idxs = rotationPreprocess(self._num_units, self._num_params)
        self._cos_list = cos_list
        self._sin_list = sin_list
        self._nsin_list = nsin_list
        self._cos_idxs = cos_idxs
        self._sin_idxs = sin_idxs
        self._nsin_idxs = nsin_idxs
        self._bottom = bottom

  @property
  def state_size(self):
        return self._num_units

  @property
  def output_size(self):
        return self._num_units

  def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or type(self).__name__):

            state_out = rotationTransform(tf.transpose(state), self._num_units, self._num_params,
                self._cos_list,  self._sin_list, self._nsin_list,
                self._cos_idxs, self._sin_idxs, self._nsin_idxs)
            state_out = tf.transpose(state_out)

            if self._bottom == True:
                input_out = linearTransformWithBias([inputs], self._num_units, bias=False)
            else:
                input_out = rotationTransform(tf.transpose(inputs), self._num_units, self._num_params,
                    self._cos_list,  self._sin_list, self._nsin_list,
                    self._cos_idxs, self._sin_idxs, self._nsin_idxs)
                input_out = tf.transpose(input_out)

            bias = vs.get_variable(
                "Bias", [self._num_units],
                dtype=tf.float32,
                initializer=init_ops.constant_initializer(dtype=tf.float32))

            output = tf.nn.relu(state_out + input_out + bias)
        return output, output
