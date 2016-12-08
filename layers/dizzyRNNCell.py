import tensorflow as tf
import numpy as np

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from utils.linearTransformWithBias import linearTransformWithBias
from utils.rotationPreprocess import rotationPreprocess
from utils.rotationTransform import rotationTransform
from utils.diagonalTransform import diagonalTransform

class DizzyRNNCell(tf.nn.rnn_cell.RNNCell):
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

    def get_sigmas(self):
        return self._sigmas

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or type(self).__name__):
            state_out = tf.transpose(state)

            if self._bottom == True:
                [state_out] = rotationTransform([("StateL", state_out)],
                                                self._num_units, scope)
                [state_out], self._sigmas = \
                        diagonalTransform([("State", state_out)],
                                          self._num_units, scope=scope)
                [state_out] = \
                        rotationTransform([("StateR", state_out)],
                                          self._num_units, scope)
                input_out = linearTransformWithBias([inputs],
                    self._num_units, bias=False, scope=scope)
            else:
                input_out = tf.transpose(inputs)
                [state_out, input_out] = \
                        rotationTransform([("StateL", input_out),
                                           ("InputL", input_out)],
                                          self._num_units, scope=scope)
                [state_out, input_out], self._sigmas = \
                        diagonalTransform([("State", state_out),
                                           ("Input", input_out)],
                                          self._num_units, scope)
                [state_out, input_out] = \
                        rotationTransform([("StateR", state_out),
                                           ("InputR", input_out)],
                                          self._num_units, scope)
                input_out = tf.transpose(input_out)

            state_out = tf.transpose(state_out)

            bias = vs.get_variable(
                "Bias", [self._num_units],
                dtype=tf.float32,
                initializer=init_ops.constant_initializer(dtype=tf.float32))

            output = tf.abs(state_out + input_out + bias)
        return output, output
