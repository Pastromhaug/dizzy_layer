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

from dizzyLayer import gen_rot_idx

class DizzyRNNCellV4(tf.nn.rnn_cell.RNNCell):
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

            input_out = DizzyLayerV3(tf.transpose(inputs), self._num_units, self._num_params,
                self._cos_list,  self._sin_list, self._nsin_list,
                self._cos_idxs, self._sin_idxs, self._nsin_idxs)
            input_out = tf.transpose(input_out)

            output = tf.abs(state_out + input_out)
        return output, output


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

            input_out = _linear([inputs], self._num_units, True)
            output = tf.abs(state_out + input_out)
        return output, output

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
