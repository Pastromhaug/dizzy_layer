import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import math
from numpy import linalg as LA

n = int(sys.argv[1])
num_rot = int(sys.argv[2])
np = int(n*(n-1)/2*num_rot/(n-1))
cos_list = [[] for i in range(np*2)]
sin_list = [[-1,-1] for i in range(np*2)]
nsin_list = [[-1,-1] for i in range(np*2)]
cos_idxs = [-1 for i in range(np*2)]
sin_idxs = [-1 for i in range(np*2)]
nsin_idxs = [-1 for i in range(np*2)]

indices = [[-1,-1] for i in range(np*4)]
values = [-1 for i in  range(np*4)]

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
        if idx >= num_rot:
            continue

        indices[idx*n*2 + 2*i] = [i,i]
        indices[idx*n*2 + 2*j+1] = [j,j]
        indices[idx*n*2 + 2*i+1] = [i,j]
        indices[idx*n*2 + 2*j] = [j,i]

        values[idx*n*2 + 2*i] = theta_num
        values[idx*n*2 + 2*j+1] = theta_num
        values[idx*n*2 + 2*i+1] = np + theta_num
        values[idx*n*2 + 2*j] = np*2 + theta_num

        theta_num += 1
    head = (head + 2) % (n - 1)
    tail = (tail + 1) % (n - 1)
    theta_idx += 1
    start_idx = (start_idx + 2) % (n - 1)
    idx = start_idx

print("indices")
print(indices)

print("values")
print(values)

indices = tf.constant(indices, dtype=tf.int64)
values = tf.constant(values, dtype=tf.int64)
print("indices")
print(indices)

print("values")
print(values)

n_prime = np
thetas = vs.get_variable(initializer=tf.random_uniform([n_prime, 1], 0, 2*math.pi), name="thetas"+str(i), dtype=tf.float32)
cos = tf.cos(thetas)
sin = tf.sin(thetas)
nsin = tf.neg(sin)
thetas_concat = tf.concat(0, [cos,sin,nsin])
print("thetas_concat")
print(thetas_concat)

gathered_thetas = tf.squeeze(tf.gather(thetas_concat, values))
shape = tf.constant([n, n], dtype=tf.int64)

splt_values = tf.split(0, num_rot, gathered_thetas)
splt_indices = tf.split(0, num_rot, indices)



var = vs.get_variable(initializer=tf.random_uniform([n, 1], 0, 5.0), name="var", dtype=tf.float32)
var_sq = tf.square(var)
var_sum = tf.reduce_sum(var_sq)
var_norm = tf.sqrt(var_sum)
var2 = var

testing = tf.constant([[1,5,2,3,4,2],[6,2,4,2,1,2],[6,2,3,2,1,3],[1,5,3,2,3,2],[67,2,2,5,2,3],[6,2,3,2,1,3]], dtype=tf.float32)
all_rots = []
for i in range(num_rot):
    curr_indices = splt_indices[i]
    curr_values = splt_values[i]
    sparse_rot = tf.SparseTensor(indices=curr_indices, values=curr_values, shape=shape)
    all_rots.append(sparse_rot)
    # var2 = tf.matmul(testing, var2)
    var2 = tf.sparse_tensor_dense_matmul(sparse_rot, var2)
var2_sq = tf.square(var2)
var2_sum = tf.reduce_sum(var2_sq)
var2_norm = tf.sqrt(var2_sum)



sess = tf.Session()
run_metadata = tf.RunMetadata()
sess.run(tf.initialize_all_variables())
(rot1, rot2, rot3, rot4, rot5) = sess.run(all_rots)#[var_norm, var2_norm, var, var2], run_metadata=run_metadata,)

print("rot1")
print(rot1)
print("rot2")
print(rot2)
print("rot3")
print(rot3)
print("rot4")
print(rot4)
print("rot5")
print(rot5)

# print("final_g")
# print(final_g)
