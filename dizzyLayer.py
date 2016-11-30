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
import numpy as np

def gen_rot_idx(n,np):
    np = int(np)
    cos_list = [[] for i in range(np*2)]
    sin_list = [[-1,-1] for i in range(np*2)]
    nsin_list = [[-1,-1] for i in range(np*2)]
    cos_idxs = [-1 for i in range(np*2)]
    sin_idxs = [-1 for i in range(np*2)]
    nsin_idxs = [-1 for i in range(np*2)]

    theta_num = 0

    head = -1
    tail = -1
    arr = [[-1] * n for i in range(n-1)]
    rot_list = [[] for i in range(n-1)]
    start_idx = 0
    idx = 0
    theta_idx = 0
    for i in range(n-1):
        for j in range(i+1, n):
            while arr[idx][i] > -1 or arr[idx][j] > -1:
                idx = (idx+1) % (n-1)
            arr[idx][i] = theta_idx
            arr[idx][j] = theta_idx
            cos_list[idx*n + i] = [idx*n + i,i]
            cos_list[idx*n + j] = [idx*n + j,j]
            sin_list[idx*n + i] = [idx*n + i,j]
            nsin_list[idx*n + j] = [idx*n + j,i]
            cos_idxs[idx*n + i] = theta_num
            cos_idxs[idx*n + j] = theta_num
            sin_idxs[idx*n + i] = theta_num
            nsin_idxs[idx*n + j] = theta_num
            theta_num += 1
        head = (head + 2) % (n - 1)
        tail = (tail + 1) % (n - 1)
        theta_idx += 1
        start_idx = (start_idx + 2) % (n - 1)
        idx = start_idx

    sin_list = [i for i in sin_list if i != [-1,-1]]
    nsin_list = [i for i in nsin_list if i != [-1,-1]]
    sin_idxs = [i for i in sin_idxs if i != -1]
    nsin_idxs = [i for i in nsin_idxs if i != -1]

    cos_list = tf.constant(cos_list, dtype=tf.int64)
    sin_list = tf.constant(sin_list, dtype=tf.int64)
    nsin_list = tf.constant(nsin_list, dtype=tf.int64)
    cos_idxs = tf.constant(cos_idxs, dtype=tf.int64)
    sin_idxs = tf.constant(sin_idxs, dtype=tf.int64)
    nsin_idxs = tf.constant(nsin_idxs, dtype=tf.int64)

    return cos_list, sin_list, nsin_list, cos_idxs, sin_idxs, nsin_idxs

def DizzyLayerV3(X, n, n_prime, cos_list,  sin_list, nsin_list, cos_idxs, sin_idxs, nsin_idxs):
    n_prime = int(n_prime)
    thetas = tf.Variable(tf.random_uniform([n_prime, 1], 0, 2*math.pi), name="thetas", dtype=tf.float32)
    cos = tf.cos(thetas)
    sin = tf.sin(thetas)
    nsin = tf.neg(sin)

    cos_thetas = tf.squeeze(tf.gather(cos, cos_idxs))
    sin_thetas = tf.squeeze(tf.gather(sin, sin_idxs))
    nsin_thetas = tf.squeeze(tf.gather(nsin, nsin_idxs))

    shape = tf.constant([2*n_prime, n])
    shape = tf.cast(shape, tf.int64)
    sparse_cos = tf.SparseTensor(indices=cos_list, values=cos_thetas, shape=shape)
    sparse_sin = tf.SparseTensor(indices=sin_list, values=sin_thetas, shape=shape)
    sparse_nsin = tf.SparseTensor(indices=nsin_list, values=nsin_thetas, shape=shape)

    full_rot = tf.sparse_add(sparse_cos, tf.sparse_add(sparse_sin, sparse_nsin))

    indices = full_rot.indices
    indices = tf.mod(indices, n)
    splt_indices = tf.split(0, n-1, indices)

    values = full_rot.values
    splt_values = tf.split(0, n-1, values)

    for i in range(n-1):
        shape = tf.cast(tf.constant([n,n]), tf.int64)
        curr_indices = splt_indices[i]
        curr_values = splt_values[i]
        sparse_rot = tf.SparseTensor(indices=curr_indices, values=curr_values, shape=shape)
        X = tf.sparse_tensor_dense_matmul(sparse_rot, X)

    return X



class DizzyRNNCellBottom(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""
  def __init__(self, num_units):
        self._num_units = num_units
        self._indices = [(a, b) for b in range(self._num_units) for a in range(b)]
        self._num_params = num_units*(num_units-1)/2
        cos_list,  sin_list, nsin_list, cos_idxs, sin_idxs, nsin_idxs = gen_rot_idx(self._num_units, self._num_params)
        self._cos_list = cos_list
        self._sin_list = sin_list
        self._nsin_list = nsin_list
        self._cos_idxs = cos_idxs
        self._sin_idxs = sin_idxs
        self._nsin_idxs = nsin_idxs

  @property
  def state_size(self):
        return self._num_units

  @property
  def output_size(self):
        return self._num_units

  def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or type(self).__name__):

            state_out = DizzyLayerV3(tf.transpose(state), self._num_units, self._num_params,
                self._cos_list,  self._sin_list, self._nsin_list,
                self._cos_idxs, self._sin_idxs, self._nsin_idxs)
            state_out = tf.transpose(state_out)

            state_bias = vs.get_variable(
                "State_Bias", [self._num_units],
                dtype=tf.float32,
                initializer=init_ops.constant_initializer(dtype=tf.float32),)
            state_out = state_out + state_bias

            input_out = _linear([inputs], self._num_units, True)
            output = tf.abs(state_out + input_out)
        return output, output


class DizzyRNNCellV3(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""
  def __init__(self, num_units):
        self._num_units = num_units
        self._indices = [(a, b) for b in range(self._num_units) for a in range(b)]
        self._num_params = num_units*(num_units-1)/2
        cos_list,  sin_list, nsin_list, cos_idxs, sin_idxs, nsin_idxs = gen_rot_idx(self._num_units, self._num_params)
        self._cos_list = cos_list
        self._sin_list = sin_list
        self._nsin_list = nsin_list
        self._cos_idxs = cos_idxs
        self._sin_idxs = sin_idxs
        self._nsin_idxs = nsin_idxs

  @property
  def state_size(self):
        return self._num_units

  @property
  def output_size(self):
        return self._num_units

  def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or type(self).__name__):

            state_out = DizzyLayerV3(tf.transpose(state), self._num_units, self._num_params,
                self._cos_list,  self._sin_list, self._nsin_list,
                self._cos_idxs, self._sin_idxs, self._nsin_idxs)
            state_out = tf.transpose(state_out)

            state_bias = vs.get_variable(
                "State_Bias", [self._num_units],
                dtype=tf.float32,
                initializer=init_ops.constant_initializer(dtype=tf.float32))
            state_out = state_out + state_bias

            input_out = DizzyLayerV3(tf.transpose(inputs), self._num_units, self._num_params,
                self._cos_list,  self._sin_list, self._nsin_list,
                self._cos_idxs, self._sin_idxs, self._nsin_idxs)
            input_out = tf.transpose(input_out)

            input_bias = vs.get_variable(
                "Input_Bias", [self._num_units],
                dtype=tf.float32,
                initializer=init_ops.constant_initializer(dtype=tf.float32))
            input_out = input_out + input_bias

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
