import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import variable_scope as vs
from utils.rotationPreprocess import rotationPreprocess

def rotationTransform(X, n, scope, num_rots=None):
    num_rots = num_rots or (n-1)
    n_prime = int(n*(n-1)//2*num_rots/(n-1))
    outputs = []

    with vs.variable_scope(scope or "RotationTransform"):

        for i, (name, x) in enumerate(X):
            (indices, values_idxs) = rotationPreprocess(n, num_rots)
            thetas = vs.get_variable(initializer=tf.random_uniform([n_prime, 1], 0, 2*math.pi),
                    name="Thetas"+str(i)+name, dtype=tf.float32)

            cos = tf.cos(thetas)
            sin = tf.sin(thetas)
            nsin = tf.neg(sin)

            thetas_concat = tf.concat(0, [cos,sin,nsin])

            gathered_values = tf.squeeze(tf.gather(thetas_concat, values_idxs))
            shape = tf.constant([n, n], dtype=tf.int64)

            splt_values = tf.split(0, num_rots, gathered_values)
            splt_indices = tf.split(0, num_rots, indices)

            shape = tf.constant([n,n], dtype=tf.int64)
            for i in range(num_rots):
                curr_indices = splt_indices[i]
                curr_values = splt_values[i]
                sparse_rot = tf.SparseTensor(indices=curr_indices, values=curr_values, shape=shape)
                x = tf.sparse_tensor_dense_matmul(sparse_rot, x)
            outputs.append(x)
    return outputs
