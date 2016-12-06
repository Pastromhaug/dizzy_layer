import tensorflow as tf
import numpy as np
import math

def rotationTransform(X, n, n_prime, cos_list,  sin_list, nsin_list, cos_idxs, sin_idxs, nsin_idxs):
    n_prime = int(n_prime)
    thetas = tf.Variable(tf.random_uniform([n_prime, 1], 0, 2*math.pi), name="thetas", dtype=tf.float32)
    cos = tf.cos(thetas)
    sin = tf.sin(thetas)
    nsin = tf.neg(sin)

    cos_thetas = tf.squeeze(tf.gather(cos, cos_idxs))
    sin_thetas = tf.squeeze(tf.gather(sin, sin_idxs))
    nsin_thetas = tf.squeeze(tf.gather(nsin, nsin_idxs))

    shape = tf.constant([2*n_prime, n], dtype=tf.int64)
    sparse_cos = tf.SparseTensor(indices=cos_list   , values=cos_thetas, shape=shape)
    sparse_sin = tf.SparseTensor(indices=sin_list, values=sin_thetas, shape=shape)
    sparse_nsin = tf.SparseTensor(indices=nsin_list, values=nsin_thetas, shape=shape)

    full_rot = tf.sparse_add(sparse_cos, tf.sparse_add(sparse_sin, sparse_nsin))

    indices = full_rot.indices
    indices = tf.mod(indices, n)
    splt_indices = tf.split(0, n-1, indices)

    values = full_rot.values
    splt_values = tf.split(0, n-1, values)

    shape = tf.constant([n,n], dtype=tf.int64)
    for i in range(n-1):
        curr_indices = splt_indices[i]
        curr_values = splt_values[i]
        sparse_rot = tf.SparseTensor(indices=curr_indices, values=curr_values, shape=shape)
        X = tf.sparse_tensor_dense_matmul(sparse_rot, X)
    return X
