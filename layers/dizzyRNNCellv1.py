import tensorflow as tf
import numpy as np

from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import tanh
from utils.linearTransformWithBias import linearTransformWithBias

class DizzyRNNCellV1(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units, input_size=None):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._indices = [(a, b) for b in range(self._num_units) for a in range(b)]

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
      state_out = tf.transpose(DizzyLayerV1(X, self._indices));
      input_out = linearTransformWithBias([inputs], self._num_units, True)
      output = tf.abs(state_out + input_out)
    return output, output

def DizzyLayerV1(X, indices):
    n = int(X.get_shape()[0])
    n_prime = int(n*(n-1)/2)
    thetas = tf.Variable(tf.random_uniform([n_prime, 1], 0, 2*math.pi), name="thetas")
    X_split = [X[k, :] for k in range(n)]
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
