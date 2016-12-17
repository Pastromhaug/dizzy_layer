import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import variable_scope as vs
from utils.rotationPreprocess import rotationPreprocess

def buildRotations(n, rand_or_identity,num_rots=None):
    print("num_rots: %d" %num_rots)
    num_rots = num_rots or (n-1)
    n_prime = int(n*(n-1)//2*num_rots/(n-1))
    outputs = []

    with vs.variable_scope("Build_Rotations"):

        (indices, values_idxs) = rotationPreprocess(n, num_rots)
        if rand_or_identity:
            print("Initialization: Random")
            thetas = vs.get_variable(initializer=tf.random_uniform([n_prime, 1], 0, 2*math.pi),
                    name="Thetas_RandInit", dtype=tf.float32)
        else:
            print("Initialization: Identity")
            thetas = vs.get_variable(initializer=tf.zeros([n_prime, 1]),
                    name="Thetas_OnesInit", dtype=tf.float32)
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
            outputs.append(sparse_rot)
    print("buildRotations output length: %d" % len(outputs))
    return outputs
