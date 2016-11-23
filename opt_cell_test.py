import tensorflow as tf
import numpy as np
import math

def gen_rot_idx(n,np):
    cos_list = [[] for i in range(np*2)]
    sin_list = [[-1,-1] for i in range(np*2)]
    nsin_list = [[-1,-1] for i in range(np*2)]
    cos_idxs = [-1 for i in range(np*2)]
    sin_idxs = [-1 for i in range(np*2)]
    nsin_idxs = [-1 for i in range(np*2)]

    arr = [[0] * n for i in range(n-1)]
    idx = 0
    theta_num = 0

    for i in range(n-1):
        for j in range(i+1, n):
            while arr[idx][i] == 1:
                idx = (idx+1) % (n-1)
            arr[idx][i] = 1
            arr[idx][j] = 1
            cos_list[idx*n + i] = [idx*n + i,i]
            cos_list[idx*n + j] = [idx*n + j,j]
            sin_list[idx*n + i] = [idx*n + i,j]
            nsin_list[idx*n + j] = [idx*n + j,i]
            cos_idxs[idx*n + i] = theta_num
            cos_idxs[idx*n + j] = theta_num
            sin_idxs[idx*n + i] = theta_num
            nsin_idxs[idx*n + j] = theta_num
            theta_num += 1

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

def DizzyLayerV3(X, n, n_prime, cos_list,  sin_list, nsin_list, cos_idxs, sin_idxs, nsin_idxs):
    thetas = tf.Variable(tf.random_uniform([n_prime, 1], 0, 2*math.pi), name="thetas")
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

    full_rot = tf.sparse_add(sparse_cos,
                    tf.sparse_add(sparse_sin, sparse_nsin))

    dense = tf.sparse_to_dense(sparse_indices=full_rot.indices,
        output_shape = full_rot.shape, sparse_values=full_rot.values)

    n_rots = tf.sparse_split(0, n-1, full_rot)
    for i in range(n-1):
        X = tf.sparse_tensor_dense_matmul(n_rots[i], X)
    #
    # return X

    return cos_thetas, sin_thetas, nsin_thetas, sparse_cos, dense, X,

class DizzyRNNCellV3(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units):
      self._num_units = num_units
      self._indices = [(a, b) for b in range(self._num_units) for a in range(b)]
      self._rot_list = tf.constant(gen_rot_list(self._num_units))
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

  def __call__(self):
    """Most basic RNN: output = new_state = activation(W * input + U * state + B)."""
    X = tf.Variable(tf.random_uniform([self._num_units, self._num_units], 0, 5), name="input")
    cos_thetas, sin_thetas, nsin_thetas, sparse_cos, dense, Y = DizzyLayerV3(X, self._num_units, self._num_params,
        self._cos_list,  self._sin_list, self._nsin_list,
        self._cos_idxs, self._sin_idxs, self._nsin_idxs)
    return (X, Y, dense, cos_thetas, sin_thetas, nsin_thetas, sparse_cos, self._cos_list,
        self._sin_list, self._nsin_list, self._cos_idxs, self._sin_idxs, self._nsin_idxs)



rnn_cell = DizzyRNNCellV3(6)
run_cell = rnn_cell()
sess = tf.Session()
sess.run(tf.initialize_all_variables())
(X, Y, dense, cos_thetas, sin_thetas, nsin_thetas, sparse_cos, cos_list, sin_list, nsin_list,
    cos_idxs, sin_idxs, nsin_idxs) = sess.run(run_cell)

print(cos_list)
print(sin_list)
print(nsin_list)
print(cos_idxs)
print(sin_idxs)
print(nsin_idxs)
print("")
print(cos_thetas)
print(sin_thetas)
print(nsin_thetas)
print(sparse_cos)
print(dense)
print(X)
print(Y)
