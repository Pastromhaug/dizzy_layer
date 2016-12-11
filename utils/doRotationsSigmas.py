import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from utils.rotationPreprocess import rotationPreprocess

def doRotationsSigmas(X, rotations, num_units):
    with vs.variable_scope("Do_Rotations"):
        sigmas = vs.get_variable(
            "Sigmas", [num_units,1],
            dtype=tf.float32,
            initializer=init_ops.constant_initializer(value=1.0, dtype=tf.float32))
        sigma_spot = int(len(rotations)/2)
        for i, sparse_rot in enumerate(rotations):
            if i == sigma_spot:
                X = X * sigmas
            X = tf.sparse_tensor_dense_matmul(sparse_rot, X)
        return X
