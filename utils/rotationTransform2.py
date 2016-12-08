import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import variable_scope as vs
from utils.rotationPreprocess2 import rotationPreprocess2

def rotationTransform2(X, n, scope):

    outputs = []
    with vs.variable_scope(scope or "RotationTransform"):
        for i, x in enumerate(X):
            n_prime = int(n*(n-1)//2)
            (cos_list,  sin_list, nsin_list, cos_idxs, sin_idxs, nsin_idxs) = \
                rotationPreprocess2(n, n_prime)
            thetas = vs.get_variable(initializer=tf.random_uniform([n_prime, 1], 0, 2*math.pi), name="thetas"+str(i), dtype=tf.float32)
            cos = tf.cos(thetas)
            sin = tf.sin(thetas)
            nsin = tf.neg(sin)

            cos_thetas = [tf.squeeze(tf.gather(cos, cos_idxs[j])) for j in range(n-1)]
            sin_thetas = [tf.squeeze(tf.gather(sin, sin_idxs[j])) for j in range(n-1)]
            nsin_thetas = [tf.squeeze(tf.gather(nsin, nsin_idxs[j])) for j in range(n-1)]

            shape = tf.constant([n, n], dtype=tf.int64)
            sparse_cos = [tf.SparseTensor(indices=cos_list[j], values=cos_thetas[j], shape=shape) for j in range(n-1)]
            sparse_sin = [tf.SparseTensor(indices=sin_list[j], values=sin_thetas[j], shape=shape) for j in range(n-1)]
            sparse_nsin = [tf.SparseTensor(indices=nsin_list[j], values=nsin_thetas[j], shape=shape) for j in range(n-1)]

            ones = tf.ones(shape=[n])
            I = tf.diag(ones)

            final_rot = I
            for i in range(n-1):
                cos_dense = tf.sparse_tensor_dense_matmul(sparse_cos[i], final_rot)
                sin_dense = tf.sparse_tensor_dense_matmul(sparse_sin[i], final_rot)
                nsin_dense = tf.sparse_tensor_dense_matmul(sparse_nsin[i], final_rot)
                final_rot = cos_dense + sin_dense + nsin_dense
            outputs.append(tf.matmul(x, final_rot))
        return outputs
