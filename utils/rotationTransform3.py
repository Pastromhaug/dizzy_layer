import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import variable_scope as vs
from utils.rotationPreprocess import rotationPreprocess

def rotationTransform3(X, n, scope):

    outputs = []
    with vs.variable_scope(scope or "RotationTransform"):
        for i, x in enumerate(X):
            n_prime = int(n*(n-1)//2)
            (cos_list,  sin_list, nsin_list, cos_idxs, sin_idxs, nsin_idxs) = \
                rotationPreprocess(n, n_prime)
            thetas = vs.get_variable(initializer=tf.random_uniform([n_prime, 1], 0, 2*math.pi), name="thetas"+str(i), dtype=tf.float32)
            cos = tf.cos(thetas)
            sin = tf.sin(thetas)
            nsin = tf.neg(sin)

            cos_thetas = tf.squeeze(tf.gather(cos, cos_idxs))
            sin_thetas = tf.squeeze(tf.gather(sin, sin_idxs))
            nsin_thetas = tf.squeeze(tf.gather(nsin, nsin_idxs))

            shape = tf.constant([n,n], dtype=tf.int64)
            for i in range(2):

                rot_cos_idxs = cos_list[n*i:n*(i+1)]
                # print(rot_cos_idxs)
                rot_sin_idxs = sin_list[n/2*i:n/2*(i+1)]
                # print(rot_sin_idxs)
                rot_nsin_idxs = nsin_list[n/2*i:n/2*(i+1)]
                # print(rot_nsin_idxs)

                rot_cos_vals = cos_thetas[n*i:n*(i+1)]
                # print(rot_cos_vals)
                rot_sin_vals = sin_thetas[n/2*i:n/2*(i+1)]
                # print(rot_sin_vals)
                rot_nsin_vals = nsin_thetas[n/2*i:n/2*(i+1)]
                # print(rot_nsin_vals)

                rot_cos = tf.SparseTensor(indices=rot_cos_idxs, values=rot_cos_vals, shape=shape)
                # print(rot_cos)
                rot_sin = tf.SparseTensor(indices=rot_sin_idxs, values=rot_sin_vals, shape=shape)
                # print(rot_sin)
                rot_nsin = tf.SparseTensor(indices=rot_nsin_idxs, values=rot_nsin_vals, shape=shape)
                # print(rot_nsin)
                # vals = tf.constant([1,5,2,3,4,6], dtype=tf.float32)
                # idxs = tf.constant([[1,3],[2,3],[2,4],[2,5],[3,5],[4,1]], dtype=tf.int64)
                # new_rot = tf.SparseTensor(indices=idxs, values=vals, shape=shape)
                # print(new_rot)

                x_cos = tf.sparse_tensor_dense_matmul(rot_cos, x)
                x_sin = tf.sparse_tensor_dense_matmul(rot_sin, x)
                x_nsin = tf.sparse_tensor_dense_matmul(rot_nsin, x)

                x = x_cos + x_sin + x_nsin
                # x = x_cos
            outputs.append(x)
    return outputs
