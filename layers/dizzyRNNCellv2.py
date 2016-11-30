import tensorflow as tf
import numpy as np


from tensorflow.python.ops import variable_scope as vs
from utils.linearTransformWithBias import linearTransformWithBias

class DizzyRNNCellV2(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units, input_size=None):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._indices = [(a, b) for b in range(self._num_units) for a in range(b)]
    self._rot_list = gen_rot_list(self._num_units)

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
      state_out = tf.transpose(DizzyLayerV2(X, self._rot_list, self._num_units));
      input_out = linearTransformWithBias([inputs], self._num_units, True)
      output = tf.abs(state_out + input_out)
    return output, output

def gen_rot_list(n):
    arr = [[0] * n for i in range(n-1)]
    rot_list = [[] for i in range(n-1)]
    idx = 0
    for i in range(n-1):
        for j in range(i+1, n):
            while arr[idx][i] == 1:
                idx = (idx+1) % (n-1)
            arr[idx][i] = 1
            arr[idx][j] = 1
            rot_list[idx].append([i, j])
    return rot_list

def DizzyLayerV2(X, rot_list, n):
    n_prime = int(n*(n-1)/2)
    thetas = tf.Variable(tf.random_uniform([n_prime, 1], 0, 2*math.pi), name="thetas")

    results = [X]
    k = 0
    for sublist in rot_list:
        indices = []
        values = []
        for (a, b) in sublist:
            c = tf.cos(thetas[k])
            s = tf.sin(thetas[k])
            indices = indices + [[a, a], [a, b], [b, a], [b, b]]
            values = values + [c, s, -s, c]
            k += 1
        shape = [n, n]
        v = tf.pack(tf.squeeze(values))
        R = tf.SparseTensor(indices, v, shape)
        results.append(tf.sparse_tensor_dense_matmul(R, results[-1]))
    return results[-1]
