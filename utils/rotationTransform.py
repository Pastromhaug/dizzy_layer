import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import variable_scope as vs
from utils.rotationPreprocess import rotationPreprocess

def rotationTransform(X, n, scope):

    outputs = []
    with vs.variable_scope(scope or "RotationTransform"):
        for i, x in enumerate(X):
            n_prime = int(n*(n-1)//2)
            (indices, values_idxs) = rotationPreprocess(n, n_prime)
            thetas = vs.get_variable(initializer=tf.random_uniform([n_prime, 1], 0, 2*math.pi), name="thetas"+str(i), dtype=tf.float32)
            cos = tf.cos(thetas)
            sin = tf.sin(thetas)
            nsin = tf.neg(sin)

            thetas_concat = tf.concat(0, [cos,sin,nsin])

            gathered_values = tf.squeeze(tf.gather(thetas_concat, values_idxs))
            shape = tf.constant([n, n], dtype=tf.int64)

            splt_values = tf.split(0, n-1, gathered_values)
            splt_indices = tf.split(0, n-1, indices)

            shape = tf.constant([n,n], dtype=tf.int64)
            for i in range(n-1):
                curr_indices = splt_indices[i]
                curr_values = splt_values[i]
                sparse_rot = tf.SparseTensor(indices=curr_indices, values=curr_values, shape=shape)
                x = tf.sparse_tensor_dense_matmul(sparse_rot, x)
            outputs.append(x)
    return outputs
