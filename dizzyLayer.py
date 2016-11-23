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

def DizzyLayerV3(X, rot_list, n):
    thetas = tf.Variable(tf.random_uniform([n/2, n-1], 0, 2*math.pi), name="thetas")
    

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

class DizzyRNNCellV3(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units, input_size=None, activation=tanh):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation
    self._indices = [(a, b) for b in range(self._num_units) for a in range(b)]
    self._rot_list = tf.constant(gen_rot_list(self._num_units))
    self._num_params = tf.constant(num_units*(num_units-2)/2)

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
      input_out = _linear([inputs], self._num_units, True)
      output = tf.abs(state_out + input_out)
    return output, output

class DizzyRNNCellV1(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units, input_size=None, activation=tanh):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation
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
      input_out = _linear([inputs], self._num_units, True)
      output = tf.abs(state_out + input_out)
    return output, output

class DizzyRNNCellV2(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units, input_size=None, activation=tanh):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation
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
      input_out = _linear([inputs], self._num_units, True)
      output = tf.abs(state_out + input_out)
    return output, output

class BasicRNNCell(tf.nn.rnn_cell.RNNCell):
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
        output = self._activation(_linear([inputs, state], self._num_units, True))
        # output = 2*tf.sqrt(tf.abs(_linear([inputs, state], self._num_units, True)))
    return output, output

def _linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with vs.variable_scope(scope or "Linear"):
    matrix = vs.get_variable(
        "Matrix", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = math_ops.matmul(args[0], matrix)
    else:
      res = math_ops.matmul(array_ops.concat(1, args), matrix)
    if not bias:
      return res
    bias_term = vs.get_variable(
        "Bias", [output_size],
        dtype=dtype,
        initializer=init_ops.constant_initializer(
            bias_start, dtype=dtype))
  return res + bias_term
