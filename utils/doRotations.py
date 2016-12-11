import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import variable_scope as vs
from utils.rotationPreprocess import rotationPreprocess

def doRotations(X, rotations):
    with vs.variable_scope("Do_Rotations"):
        for sparse_rot in rotations:
            X = tf.sparse_tensor_dense_matmul(sparse_rot, X)
        return X
