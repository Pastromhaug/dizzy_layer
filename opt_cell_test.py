import tensorflow as tf
import numpy as np
import math

def gen_rot_idx(n,np):
    cos_list = [[] for i in range(np*2)]
    sin_list = [[0,0] for i in range(np*2)]
    nsin_list = [[0,0] for i in range(np*2)]

    arr = [[0] * n for i in range(n-1)]
    idx = 0

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

    sin_list = [i for i in sin_list if i != [0,0]]
    nsin_list = [i for i in nsin_list if i != [0,0]]
    return tf.constant(cos_list), tf.constant(sin_list), tf.constant(nsin_list)

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

def DizzyLayerV3(rot_list, n):
    thetas = tf.Variable(tf.random_uniform([n-1, n/2], 0, 2*math.pi), name="thetas")
    cos = tf.cos(thetas)
    sin = tf.sin(thetas)
    neg_sin = -sin
    # gathered = tf.gather(cos[0],thetas[0])

    return thetas, cos, sin, neg_sin

class DizzyRNNCellV3(tf.nn.rnn_cell.RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units):
    self._num_units = num_units
    self._indices = [(a, b) for b in range(self._num_units) for a in range(b)]
    self._rot_list = tf.constant(gen_rot_list(self._num_units))
    self._num_params = num_units*(num_units-1)/2

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self):
    """Most basic RNN: output = new_state = activation(W * input + U * state + B)."""
    cos_list,  sin_list, nsin_list = gen_rot_idx(self._num_units, self._num_params)
    thetas, cos, sin, neg_sin = DizzyLayerV3(self._rot_list, self._num_units);
    return thetas, cos, sin, neg_sin, self._rot_list, cos_list, sin_list, nsin_list



rnn_cell = DizzyRNNCellV3(6)
run_cell = rnn_cell()
sess = tf.Session()
sess.run(tf.initialize_all_variables())
thetas, cos, sin, neg_sin, rot_list, cos_list, sin_list, nsin_list = sess.run(run_cell)
print(thetas)
print(sin)
print(cos)
print(neg_sin)
print(rot_list)
print(cos_list)
print(sin_list)
print(nsin_list)
